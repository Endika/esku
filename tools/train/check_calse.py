"""Check the segmenter against signers it has never seen, from a second LSE source.

Every segmenter constant came from SWL-LSE. That is one corpus, one collection protocol, and
its own distribution of signing tempo — so `minSignFrames` and the deceleration threshold
could easily be fitted to it rather than to signing. CALSE100 is unrelated LSE (curricular
vocabulary, six signers, mixed resolutions, 25 fps) and shares no labels with our model, so
it is useless for accuracy but ideal for this: each file holds exactly one isolated sign, so
the segmenter should emit exactly one window.

Reads only the hands — the segmenter's motion measure needs nothing else, and skipping pose
and face makes this minutes instead of an hour.

Two things here were wrong for as long as this file existed, and both inflated the very number
it reports. It fed 25 fps video to a port that converted frame indices at SWL-LSE's 20 fps, so
every window read 25% long and `MIN_SIGN_MS` arrived early; the rate is now read per video. And
it inverted MediaPipe's handedness unconditionally, on the belief that the label names the
mirrored view. Measured, that is backwards for material like this: over these clips the raw
"Right" label sits at image x 0.450 against "Left" at 0.514, and a subject facing the camera
puts their right hand on the image-left — so the raw label is already anatomical. The same
came out of LSE-Health-UVigo, where the raw label agrees with Pose's anatomical wrist in 98.6%
of detections. Inversion is now `--mirror`, for selfie material only. It happens not to move
these numbers, because `dominant()` reads whichever hand is present and swapping both slots is
symmetric, but it moves anything that reads a slot by name.

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
#: The corpus is documented at 25 fps; a container that declares 0, NaN or something
#: faster than any camera here gets this instead of poisoning the wall-clock.
FALLBACK_FPS = 25.0
MAX_PLAUSIBLE_FPS = 120.0


def landmarks_for(
    path: str, detector, mirror: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Right and left hand landmarks per frame, zero-filled when a hand is absent.

    Returns the rate the container declares, unchecked: the segmenter reasons in
    milliseconds, so reading these frame indices at the wrong rate is what made every
    window look 25% long here.
    """
    import mediapipe as mp

    capture = cv2.VideoCapture(path)
    declared = capture.get(cv2.CAP_PROP_FPS)
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
            anatomically_right = (label == "left") if mirror else (label == "right")
            if anatomically_right:
                right = array
            else:
                left = array
        rights.append(right)
        lefts.append(left)
    capture.release()
    return np.array(rights), np.array(lefts), float(declared)


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
    parser.add_argument(
        "--assume-fps",
        type=float,
        default=0.0,
        help="ignorar la tasa del contenedor y usar esta (0 = leerla del video); "
        "sirve para reproducir la medicion historica, que asumia 20 fps",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="invertir la etiqueta de MediaPipe, para material grabado en espejo (selfie)",
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

    header = f"{'signante':>10} {'videos':>7} {'ventanas/video':>15} {'1 exacta':>10}"
    print(f"{header} {'largo':>8} {'ms':>7}")
    convention = "invertida (espejo)" if arguments.mirror else "cruda (tercera persona)"
    clock = f"{arguments.assume_fps:.0f} fps impuestos" if arguments.assume_fps else "tasa real"
    print(f"   lateralidad: {convention}   reloj: {clock}")
    totals: list[int] = []
    lengths: list[int] = []
    spans_ms: list[float] = []
    rates: list[float] = []
    absurd = 0

    for signer in sorted(by_signer, key=lambda s: int(s[6:])):
        counts, spans, millis = [], [], []
        for path in by_signer[signer][:per_signer]:
            right, left, declared = landmarks_for(path, detector, arguments.mirror)
            plausible = 0.0 < declared <= MAX_PLAUSIBLE_FPS
            if not plausible:
                absurd += 1
            fps = arguments.assume_fps or (declared if plausible else FALLBACK_FPS)
            if len(right) == 0:
                continue
            rates.append(declared)
            found = sim.windows(right, left, fps)
            counts.append(len(found))
            spans.extend(len(w) for w in found)
            millis.extend(sim.span_ms(w, 1000.0 / fps) for w in found)
        if not counts:
            continue
        exact = sum(1 for c in counts if c == 1) / len(counts)
        totals.extend(counts)
        lengths.extend(spans)
        spans_ms.extend(millis)
        median = np.median(spans) if spans else 0
        print(
            f"{signer:>10} {len(counts):>7} {np.mean(counts):15.2f} "
            f"{exact * 100:9.0f}% {median:8.0f} {np.median(millis) if millis else 0:7.0f}"
        )

    single = sum(1 for c in totals if c == 1) / len(totals) * 100
    print()
    print(f"un solo signo por video, acertado: {single:.0f}%")
    print(f"ventanas por video (media)       : {np.mean(totals):.2f}   (lo correcto es 1.00)")
    print(
        f"longitud de ventana (mediana)    : {np.median(lengths):.0f} fotogramas, "
        f"{np.median(spans_ms):.0f} ms"
    )
    print(
        f"tasa declarada por el video      : mediana {np.median(rates):.2f} fps   "
        f"(rango {min(rates):.2f}-{max(rates):.2f})"
    )
    print(f"tasas absurdas, leidas a {FALLBACK_FPS:.0f} fps  : {absurd}")


if __name__ == "__main__":
    main()
