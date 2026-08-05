"""Compare feature and training variants on the same split, same seed, same budget.

Every number printed is top-1 / top-3 on SWL-LSE's held-out test set. The point is to decide
what actually helps before changing the shipped model, rather than assuming.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

DATA = Path(__file__).parent / "data"

# Pose landmarks that frame the signing space. MediaPipe's pose indices.
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17

EPOCHS = 120
BATCH = 64
PATIENCE = 15
SEED = 7


def load_raw(split: str):
    bundle = np.load(DATA / f"{split}_raw.npz", allow_pickle=True)
    count = int(bundle["n"][0])
    samples = [
        (bundle[f"r{i}"], bundle[f"l{i}"], bundle[f"p{i}"]) for i in range(count)
    ]
    return samples, bundle["y"]


def palm_width(points: np.ndarray) -> float:
    width = float(np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP]))
    return width if width > 1e-6 else 1e-6


def body_frame(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """Origin and scale taken from the torso, not the image.

    This is the point of using pose at all: a wrist at "chin height" must read the same
    whether the signer is close to the camera or across the room. Shoulder width is the most
    stable body measurement MediaPipe gives, and it does not change when the arms move.
    """
    left, right = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
    centre = (left + right) / 2
    width = float(np.linalg.norm(left - right))
    return centre, (width if width > 1e-6 else 1e-6)


def hand_block(points: np.ndarray, side: str, pose: np.ndarray | None) -> np.ndarray:
    if not points.any():
        return np.zeros(66 if pose is None else 69, dtype=np.float32)

    wrist = points[WRIST]
    scale = palm_width(points)
    mirror = -1.0 if side == "left" else 1.0
    shape = (points - wrist) / scale
    shape[:, 0] *= mirror

    if pose is None:
        return np.concatenate([shape.reshape(-1), wrist]).astype(np.float32)

    centre, width = body_frame(pose)
    located = (wrist - centre) / width
    # Both: where the hand is relative to the body, and where it is in frame.
    return np.concatenate([shape.reshape(-1), located, wrist]).astype(np.float32)


def signature(sample, frames: int, use_pose: bool, deltas: bool) -> np.ndarray:
    right, left, pose = sample
    length = len(right)
    picks = [0] * frames if length == 1 else [
        int(round(s / (frames - 1) * (length - 1))) for s in range(frames)
    ]

    rows = []
    for index in picks:
        p = pose[index] if use_pose else None
        rows.append(np.concatenate([hand_block(right[index], "right", p),
                                    hand_block(left[index], "left", p)]))
    stacked = np.stack(rows)

    if deltas:
        # Frame-to-frame change, so the model is handed motion instead of inferring it.
        motion = np.diff(stacked, axis=0, prepend=stacked[:1])
        stacked = np.concatenate([stacked, motion], axis=1)

    return stacked.reshape(-1).astype(np.float32)


def build(samples, frames: int, use_pose: bool, deltas: bool) -> np.ndarray:
    return np.stack([signature(s, frames, use_pose, deltas) for s in samples])


def augment(x: torch.Tensor, width: int, strength: float) -> torch.Tensor:
    """Jitter and scale each example slightly, differently every epoch.

    With ~27 examples per class, the model memorises signers rather than signs. Noise at this
    level is the cheapest way to tell it which variations do not change the word.
    """
    if strength <= 0:
        return x
    scale = 1 + (torch.rand(x.shape[0], 1) - 0.5) * strength
    return x * scale + torch.randn_like(x) * (strength * 0.05)


class SignHead(nn.Module):
    def __init__(self, width: int, frames: int, classes: int, hidden: int = 128):
        super().__init__()
        self.frames, self.width = frames, width
        self.norm = nn.LayerNorm(width)
        self.gru = nn.GRU(width, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, classes)
        )

    def forward(self, x):
        out, _ = self.gru(self.norm(x.view(-1, self.frames, self.width)))
        return self.head(out.mean(dim=1))


def run(name: str, frames: int, use_pose: bool, deltas: bool, strength: float, cache: dict):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    key = (frames, use_pose, deltas)
    if key not in cache:
        cache[key] = {
            split: build(cache["samples"][split], frames, use_pose, deltas)
            for split in ("train", "val", "test")
        }
    built = cache[key]

    concepts = sorted(set(cache["labels"]["train"]))
    index = {c: i for i, c in enumerate(concepts)}
    to_y = lambda raw: torch.tensor([index[c] for c in raw])

    xt, yt = torch.tensor(built["train"]), to_y(cache["labels"]["train"])
    xv, yv = torch.tensor(built["val"]), to_y(cache["labels"]["val"])
    xs, ys = torch.tensor(built["test"]), to_y(cache["labels"]["test"])
    width = xt.shape[1] // frames

    model = SignHead(width, frames, len(concepts))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    def score(x, y):
        model.eval()
        with torch.no_grad():
            top3 = model(x).topk(3, dim=1).indices
        return ((top3[:, 0] == y).float().mean().item(),
                (top3 == y.unsqueeze(1)).any(dim=1).float().mean().item())

    best, best_state, stale = 0.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            opt.zero_grad()
            crit(model(augment(xt[batch], width, strength)), yt[batch]).backward()
            opt.step()
        sched.step()
        v1, _ = score(xv, yv)
        if v1 > best:
            best, stale = v1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    t1, t3 = score(xs, ys)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:38} top1 {t1:.3f}  top3 {t3:.3f}  ({params * 4 / 1024 / 1024:.1f} MB)")
    return t1, t3


def main() -> None:
    cache = {"samples": {}, "labels": {}}
    for split in ("train", "val", "test"):
        samples, labels = load_raw(split)
        cache["samples"][split] = samples
        cache["labels"][split] = labels

    variants = [
        ("baseline: 8 frames, hands only", 8, False, False, 0.0),
        ("+ pose-relative location", 8, True, False, 0.0),
        ("+ 16 frames", 16, True, False, 0.0),
        ("+ motion deltas", 16, True, True, 0.0),
        ("+ augmentation", 16, True, True, 0.15),
    ]
    for args in variants:
        run(*args, cache)


if __name__ == "__main__":
    main()
