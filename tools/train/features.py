"""Feature extraction for the LSE vocabulary model.

MUST stay byte-for-byte equivalent to `windowSignature` in
`src/domain/recognition/services/windowSignature.ts`. A model trained on one layout and fed
another does not fail — it predicts noise, quietly. If recognition ever degrades for no
visible reason, diff these two files first.
"""

from __future__ import annotations

import numpy as np

SIGNATURE_FRAMES = 8
HAND_LANDMARKS = 21
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17

# Per hand: 21 normalised points (x, y, z) plus the wrist's own position in the frame.
HAND_FLOATS = HAND_LANDMARKS * 3 + 3
# Right hand then left hand, zero-filled when a hand is absent.
FRAME_FLOATS = HAND_FLOATS * 2
SIGNATURE_LENGTH = SIGNATURE_FRAMES * FRAME_FLOATS


def palm_width(points: np.ndarray) -> float:
    """Reference length for every ratio, stable while the fingers move."""
    width = float(np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP]))
    return width if width > 1e-6 else 1e-6


def normalize_hand(points: np.ndarray, handedness: str) -> np.ndarray:
    """Wrist-centred, scaled to unit palm width, left hands mirrored."""
    wrist = points[WRIST]
    scale = palm_width(points)
    mirror = -1.0 if handedness == "left" else 1.0

    out = (points - wrist) / scale
    out[:, 0] *= mirror
    return out


def hand_features(points: np.ndarray, handedness: str) -> np.ndarray:
    """Shape, plus where in the frame the hand is.

    Normalisation deliberately throws away position, which is right for a fingerspelled
    letter and wrong for a sign: in LSE the same handshape at the forehead and at the chest
    are different words. The wrist position is appended to put that back.
    """
    normalised = normalize_hand(points, handedness).reshape(-1)
    return np.concatenate([normalised, points[WRIST]]).astype(np.float32)


def sample_index(slot: int, length: int) -> int:
    """Evenly spaced picks, always including the window's first and last frame."""
    if length == 1:
        return 0
    return int(round((slot / (SIGNATURE_FRAMES - 1)) * (length - 1)))


def window_signature(frames: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Collapse a variable-length sign into one fixed-length vector.

    `frames` is a list of {"right": (21,3) | None, "left": (21,3) | None}.
    """
    signature = np.zeros(SIGNATURE_LENGTH, dtype=np.float32)
    if not frames:
        return signature

    for slot in range(SIGNATURE_FRAMES):
        frame = frames[sample_index(slot, len(frames))]
        base = slot * FRAME_FLOATS
        for offset, side in ((0, "right"), (HAND_FLOATS, "left")):
            points = frame.get(side)
            if points is not None:
                signature[base + offset : base + offset + HAND_FLOATS] = hand_features(
                    points, side
                )

    return signature
