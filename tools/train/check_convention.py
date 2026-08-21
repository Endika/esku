"""Do the training corpora agree about which block holds which hand?

They did not, and nobody noticed. SWL-LSE arrives with pre-extracted landmarks and both readers
inverted MediaPipe's handedness label, on the belief that it describes a mirrored selfie view.
LSE-Health, extracted later without that inversion, came out the other way. Concatenating them
taught one model two opposite conventions — and because `vocabulary_signature` reflects the left
block into the right hand's space, the disagreeing half arrives as a mirror image of itself.

Pose settles it without any convention of its own: its wrists are labelled anatomically, so the
hand block that sits nearer Pose's right wrist *is* the right hand. Anything far from 100% or
0% is a corpus that cannot be trained on alongside another.

    python check_convention.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
LEFT_WRIST, RIGHT_WRIST = 15, 16
#: Below this the corpus is not consistently either convention, which is its own problem.
CLEAR = 0.9


def measure(path: Path, limit: int) -> float | None:
    if not path.is_file():
        print(f"   {path.name}: no existe, se omite")
        return None
    bundle = np.load(path, allow_pickle=True)
    count = min(int(bundle["n"][0]), limit)
    anatomical = comparable = 0

    for sample in range(count):
        right, pose = bundle[f"r{sample}"], bundle[f"p{sample}"]
        for frame in range(len(right)):
            if not right[frame].any() or not pose[frame].any():
                continue
            wrist = right[frame][0][:2]
            left_pose, right_pose = pose[frame][LEFT_WRIST][:2], pose[frame][RIGHT_WRIST][:2]
            if not left_pose.any() or not right_pose.any():
                continue
            comparable += 1
            near_right = np.linalg.norm(wrist - right_pose) < np.linalg.norm(wrist - left_pose)
            anatomical += near_right

    if not comparable:
        print(f"   {path.name}: ningun fotograma comparable")
        return None
    share = anatomical / comparable
    verdict = "ANATOMICA" if share > CLEAR else "ESPEJADA" if share < 1 - CLEAR else "MEZCLADA"
    print(f"   {path.name:28} {share:7.1%} anatomica  (n={comparable})  -> {verdict}")
    return share


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=400, help="muestras por corpus (def. 400)")
    arguments = parser.parse_args()

    print("Convencion de lateralidad, medida contra las munecas de Pose")
    print()
    shares = [
        measure(DATA / "train_raw.npz", arguments.limit),
        measure(DATA / "uvigo" / "health_train_raw.npz", arguments.limit),
    ]
    present = [s for s in shares if s is not None]
    print()
    if len(present) < 2:
        print("   hace falta mas de un corpus para comparar")
        return
    if any(s < CLEAR for s in present):
        print("   FALLO: al menos un corpus no es anatomico, o los dos no coinciden.")
        print("   Entrenar sobre esta mezcla ensena dos convenciones opuestas a la vez.")
        sys.exit(1)
    print("   los corpus coinciden y son anatomicos, que es lo que produce la app")


if __name__ == "__main__":
    main()
