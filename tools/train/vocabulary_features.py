"""Feature layout for the trained LSE vocabulary model.

MUST stay byte-for-byte equivalent to `vocabularySignature` in
`src/domain/recognition/services/vocabularySignature.ts`. A model trained on one layout and
fed another does not fail — it predicts noise, quietly.

Deliberately separate from `features.py`, which describes the *taught-sign* signature. Those
two used to be the same function, and every improvement to this model invalidated every sign
a user had recorded. They ship on different clocks and now have different code.

Every element here earned its place on SWL-LSE's held-out test split:

    hands only, 8 frames                   0.632 / 0.826
    + hand position relative to the torso  0.689 / 0.836
    + 16 frames                            0.719 / 0.849
    + torso and head orientation           0.729 / 0.865
    + facial expression (6 scalars)        0.741 / 0.870   <- shipped

Rejected, with numbers: frame-to-frame motion deltas (0.666), raw face coordinates (0.702),
input augmentation (0.699). Dropping depth costs only 0.003, so `z` stays but carries little.
"""

from __future__ import annotations

import numpy as np

from face import expression

FRAMES = 16

WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12

HAND_FLOATS = 21 * 3 + 3 + 3  # shape, wrist located to the body, wrist in the frame
TORSO_FLOATS = 5
FACE_FLOATS = 6
FRAME_FLOATS = HAND_FLOATS * 2 + TORSO_FLOATS + FACE_FLOATS
SIGNATURE_LENGTH = FRAMES * FRAME_FLOATS


def palm_width(points: np.ndarray) -> float:
    width = float(np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP]))
    return width if width > 1e-6 else 1e-6


def body_frame(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """Origin and scale taken from the torso, not the image.

    This is the whole point of using pose: a wrist at chin height must read the same whether
    the signer is close to the camera or across the room. Shoulder width is the most stable
    body measurement MediaPipe gives, and it does not change as the arms move.
    """
    left, right = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
    centre = (left + right) / 2
    width = float(np.linalg.norm(left - right))
    return centre, (width if width > 1e-6 else 1e-6)


def hand_block(points: np.ndarray, side: str, pose: np.ndarray) -> np.ndarray:
    if not points.any():
        return np.zeros(HAND_FLOATS, dtype=np.float32)

    wrist = points[WRIST]
    shape = (points - wrist) / palm_width(points)
    if side == "left":
        shape[:, 0] *= -1

    centre, width = body_frame(pose)
    located = (wrist - centre) / width
    return np.concatenate([shape.reshape(-1), located, wrist]).astype(np.float32)


def torso_block(pose: np.ndarray) -> np.ndarray:
    """Orientation of the torso, and of the head sitting on it.

    The shoulder line's angle gives the turn and the depth difference says which way the body
    is squared — LSE uses that to point at referents in space. The head offset is what makes
    a negating head-shake visible: small per frame, but it swings across the signature.
    """
    left, right = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
    centre, width = body_frame(pose)
    across = left - right
    head = (pose[NOSE] - centre) / width
    return np.array(
        [np.arctan2(across[1], across[0]), across[2] / width, head[0], head[1], head[2]],
        dtype=np.float32,
    )


def sample_index(slot: int, length: int) -> int:
    if length == 1:
        return 0
    return int(round((slot / (FRAMES - 1)) * (length - 1)))


def frame_floats(include_face: bool = True) -> int:
    return FRAME_FLOATS if include_face else FRAME_FLOATS - FACE_FLOATS


def signature_length(include_face: bool = True) -> int:
    return FRAMES * frame_floats(include_face)


def vocabulary_signature(right, left, pose, face, include_face: bool = True) -> np.ndarray:
    """Collapse one sign into the fixed-length vector the model reads.

    `include_face=False` drops the six expression scalars, and with them the reason to run a
    third MediaPipe model on every frame. The default is what ships, so the layout stays
    byte-for-byte equivalent to the TypeScript unless a caller asks otherwise. Whether the block
    earns its frame rate was only ever measured on SWL-LSE, which is citation form — no
    interrogative, no negation, none of the non-manual marking that makes a face worth reading.
    """
    width = frame_floats(include_face)
    signature = np.zeros(FRAMES * width, dtype=np.float32)
    if len(right) == 0:
        return signature

    for slot in range(FRAMES):
        index = sample_index(slot, len(right))
        blocks = [
            hand_block(right[index], "right", pose[index]),
            hand_block(left[index], "left", pose[index]),
            torso_block(pose[index]),
        ]
        if include_face:
            blocks.append(expression(face[index]))
        signature[slot * width : (slot + 1) * width] = np.concatenate(blocks)

    return signature
