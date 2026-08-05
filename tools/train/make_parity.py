"""Emit the cross-language parity fixture for the vocabulary signature.

The TypeScript app and this trainer each build the model's input vector independently. If
they ever disagree the model does not fail — it predicts noise. This writes a synthetic
frame sequence and the signature Python computes for it, so a TypeScript test can rebuild
the same frames and demand the same numbers.

Synthetic rather than real landmarks so the fixture is small, readable and committable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from face import select
from vocabulary_features import SIGNATURE_LENGTH, vocabulary_signature

FIXTURES = Path(__file__).parent.parent.parent / "src" / "test" / "fixtures"
FRAMES = 5

CHAINS = [(1, [2, 3, 4]), (5, [6, 7, 8]), (9, [10, 11, 12]),
          (13, [14, 15, 16]), (17, [18, 19, 20])]
KNUCKLE_X = [0.38, 0.44, 0.5, 0.56, 0.6]


def build_hand(curls: list[float], offset: tuple[float, float]) -> np.ndarray:
    """Mirrors `buildHand` in `src/test/handFixtures.ts`, joint for joint."""
    points = np.zeros((21, 3))
    points[0] = (0.5, 0.9, 0.0)
    for finger, (mcp, joints) in enumerate(CHAINS):
        base_x = KNUCKLE_X[finger]
        base_y = 0.82 if finger == 0 else 0.72
        points[mcp] = (base_x, base_y, 0.0)
        pip = (base_x, base_y - 0.06, 0.0)
        points[joints[0]] = pip
        theta = curls[finger] * math.pi
        direction = (math.sin(theta), -math.cos(theta))
        for segment, joint in enumerate(joints[1:]):
            reach = 0.05 * (segment + 1)
            points[joint] = (pip[0] + direction[0] * reach, pip[1] + direction[1] * reach, 0.0)
    points[:, 0] += offset[0]
    points[:, 1] += offset[1]
    return points.astype(np.float32)


def build_pose(step: int) -> np.ndarray:
    return np.array(
        [[0.3 + j * 0.01 + step * 0.005, 0.2 + j * 0.012, j * 0.002 - 0.03] for j in range(33)],
        dtype=np.float32,
    )


def build_mesh(step: int) -> np.ndarray:
    return np.array(
        [
            [0.45 + (j % 37) * 0.002 + step * 0.001, 0.15 + (j % 53) * 0.0015, (j % 11) * 0.001]
            for j in range(478)
        ],
        dtype=np.float32,
    )


def main() -> None:
    curls = [0.9, 0.0, 0.9, 0.9, 0.9]
    right = np.stack([build_hand(curls, (i * 0.02, 0.0)) for i in range(FRAMES)])
    left = np.zeros((FRAMES, 21, 3), dtype=np.float32)
    pose = np.stack([build_pose(i) for i in range(FRAMES)])
    meshes = [build_mesh(i) for i in range(FRAMES)]

    # The cached dataset stores the 21 selected landmarks, not the full mesh. The app indexes
    # the full mesh directly with the same indices, so both reach the same points — but the
    # fixture has to feed each side the shape it actually receives.
    face = np.stack([select(mesh) for mesh in meshes])
    signature = vocabulary_signature(right, left, pose, face)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "signature-parity.json").write_text(
        json.dumps(
            {
                "curls": curls,
                "frames": FRAMES,
                "handOffsetStep": 0.02,
                "signatureLength": SIGNATURE_LENGTH,
                "pose": [p.tolist() for p in pose],
                "faceMesh": [m.tolist() for m in meshes],
                "signature": [round(float(v), 6) for v in signature],
            }
        ),
        encoding="utf-8",
    )
    print(f"signature parity -> src/test/fixtures/signature-parity.json ({SIGNATURE_LENGTH})")


if __name__ == "__main__":
    main()
