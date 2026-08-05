"""Score the model the way the *app* feeds it, not the way training did.

Training resamples every frame of a recording that had a hand in it. The app resamples
whatever `SignSegmenter` hands over: a window that starts when motion begins and ends when
the hand settles. Those are different slices of the same sign, and the gap between them is
invisible from the training numbers — the model would keep reporting 74% while the deployed
app did something worse.

This is a Python port of `src/domain/recognition/services/SignSegmenter.ts`. It exists to
measure the deployed path, so if the TypeScript changes, change this too.
"""

from __future__ import annotations

import numpy as np
import torch

from train import SignHead, load
from vocabulary_features import SIGNATURE_LENGTH, vocabulary_signature

MOTION_THRESHOLD = 0.03
SETTLE_FRAMES = 10
MIN_FRAMES = 4
MAX_FRAMES = 48

WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
TIPS = [4, 8, 12, 16, 20]


def palm_width(points: np.ndarray) -> float:
    width = float(np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP]))
    return width if width > 1e-6 else 1e-6


def motion_between(current: np.ndarray, previous: np.ndarray) -> float:
    """Mean fingertip travel, in palm widths — the same measure the app uses."""
    if not current.any() or not previous.any():
        return 0.0
    total = sum(float(np.linalg.norm(current[t] - previous[t])) for t in TIPS)
    return total / len(TIPS) / palm_width(current)


def dominant(right: np.ndarray, left: np.ndarray, index: int) -> np.ndarray:
    """Whichever hand is actually present, matching the app's `dominantHand`.

    Reading only the right hand silently discards every left-handed sign — it never registers
    motion, so the segmenter never activates and the sign is dropped. That looked like a
    segmenter fault for a while; it was the simulation's.
    """
    r, l = right[index], left[index]
    if not r.any():
        return l
    if not l.any():
        return r
    return r if np.ptp(r[:, 0]) * np.ptp(r[:, 1]) >= np.ptp(l[:, 0]) * np.ptp(l[:, 1]) else l


def segment(right: np.ndarray, left: np.ndarray | None = None) -> list[int] | None:
    """Return the frame indices the app's segmenter would hand to the classifier."""
    if left is None:
        left = np.zeros_like(right)
    window: list[int] = []
    still = 0
    active = False

    for index in range(len(right)):
        motion = (
            motion_between(dominant(right, left, index), dominant(right, left, window[-1]))
            if window
            else 0.0
        )
        window.append(index)

        if motion > MOTION_THRESHOLD:
            active = True
            still = 0
            if len(window) >= MAX_FRAMES:
                return window if len(window) >= MIN_FRAMES else None
            continue

        if not active:
            if len(window) > MIN_FRAMES:
                window.pop(0)
            continue

        if len(window) >= MAX_FRAMES:
            return window if len(window) >= MIN_FRAMES else None

        still += 1
        if still >= SETTLE_FRAMES:
            return window if len(window) >= MIN_FRAMES else None

    # The recording ended mid-sign; the app would emit on the hand leaving frame.
    return window if active and len(window) >= MIN_FRAMES else None


def main() -> None:
    bundle = np.load("data/test_raw.npz", allow_pickle=True)
    count = int(bundle["n"][0])

    x_train, y_train = load("train")
    concepts = sorted(set(np.load("data/train_raw.npz", allow_pickle=True)["y"]))
    index = {c: i for i, c in enumerate(concepts)}

    model = SignHead(len(concepts))
    weights = np.fromfile("../../public/models/lse-vocabulary.bin", dtype=np.float32)
    state = model.state_dict()
    offset = 0
    for key in state:
        size = state[key].numel()
        state[key] = torch.tensor(weights[offset : offset + size]).reshape(state[key].shape)
        offset += size
    model.load_state_dict(state)
    model.eval()

    whole, segmented, dropped = [], [], 0
    truth = []

    for i in range(count):
        right, left = bundle[f"r{i}"], bundle[f"l{i}"]
        pose, face = bundle[f"p{i}"], bundle[f"f{i}"]
        truth.append(index[str(bundle["y"][i])])

        whole.append(vocabulary_signature(right, left, pose, face))

        picks = segment(right, left)
        if picks is None:
            dropped += 1
            segmented.append(np.zeros(SIGNATURE_LENGTH, dtype=np.float32))
        else:
            k = np.array(picks)
            segmented.append(vocabulary_signature(right[k], left[k], pose[k], face[k]))

    y = torch.tensor(truth)
    for name, matrix in (("whole recording (training view)", whole), ("app segmenter", segmented)):
        with torch.no_grad():
            top3 = model(torch.tensor(np.stack(matrix))).topk(3, dim=1).indices
        top1 = (top3[:, 0] == y).float().mean().item()
        hit3 = (top3 == y.unsqueeze(1)).any(dim=1).float().mean().item()
        print(f"{name:34} top1 {top1:.3f}  top3 {hit3:.3f}")

    print(f"\nsigns the segmenter never emitted: {dropped} of {count}")


if __name__ == "__main__":
    main()
