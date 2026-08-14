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

from face import expression

DATA = Path(__file__).parent / "data"

# Pose landmarks that frame the signing space. MediaPipe's pose indices.
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17

EPOCHS = 120
BATCH = 64
PATIENCE = 15
SEED = 7

# SWL-LSE is 20.00 fps in every reference video. Face staleness is asked for in milliseconds,
# not frames, because the app's frame rate is nothing like the corpus's: holding a reading for
# two frames is 100 ms here and ~400 ms on a phone reaching 5 fps.
CORPUS_FPS = 20.0


def stale_face_index(index: int, hold_ms: float) -> int:
    """Which frame's face reading a pipeline running the face model less often would still hold."""
    if hold_ms <= 0:
        return index
    step = round(hold_ms / 1000 * CORPUS_FPS) + 1
    return (index // step) * step


def load_raw(split: str):
    bundle = np.load(DATA / f"{split}_raw.npz", allow_pickle=True)
    count = int(bundle["n"][0])
    has_face = "f0" in bundle
    samples = [
        (
            bundle[f"r{i}"],
            bundle[f"l{i}"],
            bundle[f"p{i}"],
            bundle[f"f{i}"] if has_face else None,
        )
        for i in range(count)
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


def torso_block(pose: np.ndarray) -> np.ndarray:
    """Orientation of the torso and of the head on top of it.

    The shoulder vector's angle gives the turn, and the depth difference between shoulders
    says which way the body is squared — LSE uses that to point at referents in space. The
    head offset is what makes a negating head-shake visible: it is a small value, but it
    swings across the frames of the signature.
    """
    left, right = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
    centre, width = body_frame(pose)
    across = left - right
    head = (pose[NOSE] - centre) / width
    return np.array(
        [
            np.arctan2(across[1], across[0]),  # shoulder line tilt
            across[2] / width,                 # one shoulder forward = torso turned
            head[0],
            head[1],
            head[2],
        ],
        dtype=np.float32,
    )


def drop_depth(block: np.ndarray) -> np.ndarray:
    """Strip every third value — the z channel — from a flattened point list."""
    return np.delete(block.reshape(-1, 3), 2, axis=1).reshape(-1)


def signature(
    sample, frames: int, use_pose: bool, deltas: bool, face_mode: str = "none",
    use_depth: bool = True, face_hold_ms: float = 0.0, use_torso: bool = True,
) -> np.ndarray:
    right, left, pose, face = sample
    length = len(right)
    picks = [0] * frames if length == 1 else [
        int(round(s / (frames - 1) * (length - 1))) for s in range(frames)
    ]

    rows = []
    for index in picks:
        p = pose[index] if use_pose else None
        parts = [hand_block(right[index], "right", p), hand_block(left[index], "left", p)]

        # Separate from use_pose on purpose: dropping that would also drop the
        # pose-relative hand location, which is the one large measured gain (+5.7)
        if use_pose and use_torso:
            parts.append(torso_block(pose[index]))

        if face_mode != "none" and face is not None:
            points = face[stale_face_index(index, face_hold_ms)]
            if face_mode == "expression":
                parts.append(expression(points))
            elif face_mode == "points":
                # Located against the torso like the hands, so face position and hand
                # position live in the same coordinate frame.
                centre, width = body_frame(pose[index])
                parts.append(((points - centre) / width).reshape(-1).astype(np.float32))
            elif face_mode == "both":
                centre, width = body_frame(pose[index])
                parts.append(((points - centre) / width).reshape(-1).astype(np.float32))
                parts.append(expression(points))

        row = np.concatenate(parts)
        rows.append(row)
    stacked = np.stack(rows)

    if not use_depth:
        # MediaPipe's z is inferred from one camera rather than measured, so it is the
        # noisiest channel by far. Whether it earns its place is a question for the test set,
        # not for intuition.
        stacked = np.stack(
            [drop_depth_row(r, use_pose and use_torso, face_mode) for r in stacked]
        )

    if deltas:
        # Frame-to-frame change, so the model is handed motion instead of inferring it.
        motion = np.diff(stacked, axis=0, prepend=stacked[:1])
        stacked = np.concatenate([stacked, motion], axis=1)

    return stacked.reshape(-1).astype(np.float32)


def drop_depth_row(row: np.ndarray, use_pose: bool, face_mode: str) -> np.ndarray:
    """Remove z from the landmark parts of a frame, leaving derived scalars untouched."""
    hand_len = 69 if use_pose else 66
    parts: list[np.ndarray] = []
    offset = 0

    for _ in range(2):
        block = row[offset : offset + hand_len]
        offset += hand_len
        # 21 shape points, then wrist-located (3) and wrist-in-frame (3) when pose is on.
        parts.append(drop_depth(block[:63]))
        parts.append(drop_depth(block[63:]))

    if use_pose:
        parts.append(row[offset : offset + 5])  # torso scalars: already derived
        offset += 5

    if face_mode in ("points", "both"):
        parts.append(drop_depth(row[offset : offset + 63]))
        offset += 63
    if face_mode in ("expression", "both"):
        parts.append(row[offset : offset + 6])
        offset += 6

    return np.concatenate(parts)


def build(
    samples, frames: int, use_pose: bool, deltas: bool, face_mode: str, use_depth: bool,
    face_hold_ms: float = 0.0, use_torso: bool = True,
) -> np.ndarray:
    return np.stack(
        [
            signature(
                s, frames, use_pose, deltas, face_mode, use_depth, face_hold_ms, use_torso
            )
            for s in samples
        ]
    )


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


def run(name: str, frames: int, use_pose: bool, deltas: bool, strength: float,
        face_mode: str, cache: dict, use_depth: bool = True, face_hold_ms: float = 0.0,
        seed: int = SEED, use_torso: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    key = (frames, use_pose, deltas, face_mode, use_depth, face_hold_ms, use_torso)
    if key not in cache:
        cache[key] = {
            split: build(
                cache["samples"][split], frames, use_pose, deltas, face_mode, use_depth,
                face_hold_ms, use_torso,
            )
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

    # Round three. The face expression block costs a whole FaceLandmarker pass per frame for
    # 6 of the 149 floats in a frame, and frame rate is what decides whether the app writes
    # anything at all. So: how stale may that reading be before its +1.2 top-1 is gone?
    #
    # Over several seeds, because the gaps being read are ~1 point on 598 test samples, which
    # is the same size as the effect one seed change can invent.
    # Round four. The torso block's measured +1.0 sits inside the between-seed sd of 0.024
    # that round three found, so it gets the same treatment. Both variants keep the
    # pose-relative hand location: only the five torso scalars move.
    variants = [
        ("with torso (shipped)", True),
        ("without torso", False),
    ]
    seeds = [7, 13, 29, 41]
    results: dict[str, list[float]] = {name: [] for name, _ in variants}

    for seed in seeds:
        for name, use_torso in variants:
            t1, _ = run(f"{name} [seed {seed}]", 16, True, False, 0.0, "none", cache,
                        True, 0.0, seed, use_torso)
            results[name].append(t1)

    print()
    for name, _ in variants:
        got = np.array(results[name])
        print(f"{name:26} mean {got.mean():.3f}  sd {got.std(ddof=1):.3f}  "
              f"min {got.min():.3f}  max {got.max():.3f}  n={len(got)}")


if __name__ == "__main__":
    main()
