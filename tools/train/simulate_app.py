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

#: SWL-LSE is 20.00 fps in all 300 of its reference videos, measured, not assumed. Every
#: threshold below used to be a frame count, which silently meant "at 20 fps" — and the
#: browser runs at whatever three MediaPipe models allow. This constant is what converts
#: frame indices to the wall-clock the segmenter now reasons in; it belongs to the *dataset*,
#: never to the app.
DATASET_FPS = 20.0
FRAME_MS = 1000.0 / DATASET_FPS

MOTION_RATE = 0.6  # palm widths per second
DECELERATION_DROP = 0.45
DECELERATION_HOLD_MS = 50
#: N frames span N-1 intervals, so the swept counts convert as (n-1)/DATASET_FPS: the 24-frame
#: floor is 1150 ms, not 1200. Kept exact so this reproduces the pre-conversion numbers and any
#: drift is a porting mistake rather than a retune.
MIN_SIGN_MS = 1150
MIN_MS = 150
MIN_FRAMES = 4
MAX_MS = 2350

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


def span_ms(window: list[int], frame_ms: float = FRAME_MS) -> float:
    """Wall-clock span of a window, from frame indices at the *recording's* rate.

    The rate belongs to whatever was recorded, not to this module. SWL-LSE is 20 fps so that
    is the default, but a 25 fps corpus fed through the 20 fps default reads every span 25%
    long, which moves `MIN_SIGN_MS` and `MAX_MS` earlier in real time and inflates the window
    count. Pass the real rate.
    """
    return (window[-1] - window[0]) * frame_ms if len(window) > 1 else 0.0


def windows(
    right: np.ndarray, left: np.ndarray | None = None, fps: float = DATASET_FPS
) -> list[list[int]]:
    """Every window the app would emit over a stream. Port of `SignSegmenter.push`.

    Closes where the signer decelerates off this window's peak speed, not where they hold
    still: stillness is a property of dictionary recordings, not of signing.

    Reasons in milliseconds, like the TypeScript it ports. The frame-counting version agreed
    with the app only at the dataset's own 20 fps, which is why every offline number looked
    healthy while the browser wrote nothing.
    """
    if left is None:
        left = np.zeros_like(right)
    frame_ms = 1000.0 / fps
    out: list[list[int]] = []
    window: list[int] = []
    peak = 0.0
    slow_ms = 0.0
    active = False

    def close() -> None:
        nonlocal window, peak, slow_ms, active
        if span_ms(window, frame_ms) >= MIN_MS and len(window) >= MIN_FRAMES:
            out.append(window)
        window, peak, slow_ms, active = [], 0.0, 0.0, False

    for index in range(len(right)):
        rate = (
            motion_between(dominant(right, left, index), dominant(right, left, window[-1]))
            / (frame_ms / 1000.0)
            if window
            else 0.0
        )
        window.append(index)

        if rate > MOTION_RATE:
            active = True
            peak = max(peak, rate)

        if not active:
            while len(window) > 1 and span_ms(window, frame_ms) > MIN_MS:
                window.pop(0)
            continue

        if span_ms(window, frame_ms) >= MAX_MS:
            close()
            continue

        decelerating = (
            span_ms(window, frame_ms) >= MIN_SIGN_MS
            and peak > 0
            and rate < peak * DECELERATION_DROP
        )
        if not decelerating:
            slow_ms = 0.0
            continue

        slow_ms += frame_ms
        if slow_ms >= DECELERATION_HOLD_MS:
            close()

    # The recording ended mid-sign; the app would emit on the hand leaving frame.
    if active:
        close()
    return out


def segment(
    right: np.ndarray, left: np.ndarray | None = None, fps: float = DATASET_FPS
) -> list[int] | None:
    """The first window only — what a one-sign recording yields."""
    found = windows(right, left, fps)
    return found[0] if found else None


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
