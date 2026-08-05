"""Recover MediaPipe's face mesh from the raw protobuf the pickles carry.

SWL-LSE stores the legacy Holistic face landmarks as a serialised
`NormalizedLandmarkList`, and MediaPipe 1.x no longer ships the generated class to decode
it. The message is trivial, so it is parsed here rather than pinning an ancient MediaPipe.

Non-manual markers are not decoration in LSE: raised eyebrows mark a question, a head shake
negates, and mouth gestures separate minimal pairs. A model that only sees hands is blind to
a whole grammatical channel.
"""

from __future__ import annotations

import struct

import numpy as np

# Face Mesh indices for the parts that actually carry meaning while signing. Using all 468
# points would add 1,404 inputs to a model trained on 6,336 examples — far more capacity to
# memorise signers than to learn signs.
FACE_POINTS = {
    "forehead": 10,
    "nose_tip": 1,
    "chin": 152,
    "brow_left_outer": 70,
    "brow_left_inner": 107,
    "brow_right_outer": 300,
    "brow_right_inner": 336,
    "eye_left_outer": 33,
    "eye_left_inner": 133,
    "eye_left_upper": 159,
    "eye_left_lower": 145,
    "eye_right_outer": 263,
    "eye_right_inner": 362,
    "eye_right_upper": 386,
    "eye_right_lower": 374,
    "mouth_left": 61,
    "mouth_right": 291,
    "lip_upper": 13,
    "lip_lower": 14,
    "cheek_left": 234,
    "cheek_right": 454,
}

FACE_ORDER = list(FACE_POINTS)
FACE_COUNT = len(FACE_ORDER)


def parse_landmark_list(payload: bytes) -> np.ndarray:
    """Decode `repeated NormalizedLandmark landmark = 1` into an (N, 3) array.

    Hand-rolled because the wire format is two cases: field 1 length-delimited (a landmark),
    and inside it fields 1-3 as 32-bit floats. Anything else is skipped by wire type, so an
    unexpected field cannot desynchronise the reader.
    """
    points: list[tuple[float, float, float]] = []
    offset = 0
    end = len(payload)

    while offset < end:
        key, offset = _varint(payload, offset)
        field, wire = key >> 3, key & 7

        if wire == 2:
            length, offset = _varint(payload, offset)
            chunk = payload[offset : offset + length]
            offset += length
            if field == 1:
                points.append(_parse_landmark(chunk))
        elif wire == 5:
            offset += 4
        elif wire == 1:
            offset += 8
        elif wire == 0:
            _, offset = _varint(payload, offset)
        else:
            break

    return np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def _parse_landmark(chunk: bytes) -> tuple[float, float, float]:
    values = [0.0, 0.0, 0.0]
    offset = 0
    while offset < len(chunk):
        key, offset = _varint(chunk, offset)
        field, wire = key >> 3, key & 7
        if wire == 5:
            if 1 <= field <= 3:
                values[field - 1] = struct.unpack_from("<f", chunk, offset)[0]
            offset += 4
        elif wire == 0:
            _, offset = _varint(chunk, offset)
        elif wire == 2:
            length, offset = _varint(chunk, offset)
            offset += length
        else:
            break
    return values[0], values[1], values[2]


def _varint(buffer: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(buffer):
        byte = buffer[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return result, offset


HIGHEST_INDEX = max(FACE_POINTS.values())


def select(mesh: np.ndarray) -> np.ndarray:
    """Reduce a full 468-point mesh to the landmarks that matter, in a fixed order.

    The guard is against the highest index actually read, not the mesh's nominal size: a
    468-point mesh numbers its points 0..467, so demanding 469 rows silently zeroed every
    face in the dataset.
    """
    if len(mesh) <= HIGHEST_INDEX:
        return np.zeros((FACE_COUNT, 3), dtype=np.float32)
    return np.stack([mesh[FACE_POINTS[name]] for name in FACE_ORDER]).astype(np.float32)


def expression(points: np.ndarray) -> np.ndarray:
    """Scalar non-manual features, scaled by face size so distance does not change them.

    Coordinates say where the face is; these say what it is *doing*, which is the part that
    carries grammar. Handing the model six meaningful ratios is a far better use of 6,336
    training examples than sixty raw coordinates it would have to derive them from.
    """
    if not points.any():
        return np.zeros(6, dtype=np.float32)

    index = {name: i for i, name in enumerate(FACE_ORDER)}
    span = float(np.linalg.norm(points[index["cheek_left"]] - points[index["cheek_right"]]))
    scale = span if span > 1e-6 else 1e-6

    def gap(a: str, b: str) -> float:
        return float(np.linalg.norm(points[index[a]] - points[index[b]])) / scale

    return np.array(
        [
            gap("brow_left_inner", "eye_left_upper"),   # eyebrow raise, left
            gap("brow_right_inner", "eye_right_upper"),  # eyebrow raise, right
            gap("eye_left_upper", "eye_left_lower"),     # eye openness, left
            gap("eye_right_upper", "eye_right_lower"),   # eye openness, right
            gap("lip_upper", "lip_lower"),               # mouth openness
            gap("mouth_left", "mouth_right"),            # mouth width
        ],
        dtype=np.float32,
    )
