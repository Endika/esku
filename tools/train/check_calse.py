"""Check the segmenter against signers it has never seen, from a second LSE source.

Every segmenter constant came from SWL-LSE. That is one corpus, one collection protocol, and
its own distribution of signing tempo — so `minSignFrames` and the deceleration threshold
could easily be fitted to it rather than to signing. CALSE100 is unrelated LSE (curricular
vocabulary, six signers, mixed resolutions, 25 fps) and shares no labels with our model, so
it is useless for accuracy but ideal for this: each file holds exactly one isolated sign, so
the segmenter should emit exactly one window.

Reads only the hands — the segmenter's motion measure needs nothing else, and skipping pose
and face makes this minutes instead of an hour.

The corpus is **not** in this repository and its terms of use are unknown, so the path is an
argument rather than a constant. Nothing here trains on it or redistributes any part of it:
it is a local measurement, and it must stay that way until someone establishes a licence.

Expects one video per sign, named `<signer>_<word>_<n>.mp4`.

    python check_calse.py /path/to/CALSE100/Original [videos-per-signer]
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import simulate_app as sim

MODEL = "../../public/models/hand_landmarker.task"


def landmarks_for(path: str, detector) -> tuple[np.ndarray, np.ndarray]:
    """Right and left hand landmarks per frame, zero-filled when a hand is absent."""
    import mediapipe as mp

    capture = cv2.VideoCapture(path)
    rights, lefts = [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        found = detector.detect(image)
        right = np.zeros((21, 3), dtype=np.float32)
        left = np.zeros((21, 3), dtype=np.float32)
        for i, points in enumerate(found.hand_landmarks):
            array = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
            label = found.handedness[i][0].category_name.lower()
            # MediaPipe labels the mirrored view, so its "left" is the signer's right.
            if label == "left":
                right = array
            else:
                left = array
        rights.append(right)
        lefts.append(left)
    capture.release()
    return np.array(rights), np.array(lefts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="directory of <signer>_<word>_<n>.mp4 files")
    parser.add_argument(
        "per_signer",
        nargs="?",
        type=int,
        default=10,
        help="videos to read per signer (default 10; each takes a few seconds)",
    )
    arguments = parser.parse_args()

    videos = sorted(glob.glob(f"{arguments.source}/*.mp4"))
    if not videos:
        parser.error(f"no .mp4 files under {arguments.source}")

    per_signer = arguments.per_signer
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    by_signer: dict[str, list[str]] = collections.defaultdict(list)
    for path in videos:
        by_signer[os.path.basename(path).split("_")[0]].append(path)

    print(f"{'signante':>10} {'videos':>7} {'ventanas/video':>15} {'1 exacta':>10} {'largo':>8}")
    totals: list[int] = []
    lengths: list[int] = []

    for signer in sorted(by_signer, key=lambda s: int(s[6:])):
        counts, spans = [], []
        for path in by_signer[signer][:per_signer]:
            right, left = landmarks_for(path, detector)
            if len(right) == 0:
                continue
            found = sim.windows(right, left)
            counts.append(len(found))
            spans.extend(len(w) for w in found)
        if not counts:
            continue
        exact = sum(1 for c in counts if c == 1) / len(counts)
        totals.extend(counts)
        lengths.extend(spans)
        median = np.median(spans) if spans else 0
        print(
            f"{signer:>10} {len(counts):>7} {np.mean(counts):15.2f} "
            f"{exact * 100:9.0f}% {median:8.0f}"
        )

    print()
    print(f"un solo signo por video, acertado: {sum(1 for c in totals if c == 1) / len(totals) * 100:.0f}%")
    print(f"ventanas por video (media)       : {np.mean(totals):.2f}   (lo correcto es 1.00)")
    print(f"longitud de ventana (mediana)    : {np.median(lengths):.0f} fotogramas a 25 fps")


if __name__ == "__main__":
    main()
