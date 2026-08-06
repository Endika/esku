"""Can a video corpus of isolated signs ship as a nearest-prototype vocabulary pack?

**Answered on CALSE100, 2026-08-06: no.** Held-out signer, 20 words, six signers —
top-1 43.3% against a 5% chance baseline, so the signs *are* separable, but the confidence
is useless: the median distance to the correct sign is 0.837 and to the best wrong sign
0.775. The wrong answer is typically *closer*. Usable operating points are 17.5% coverage
at 71% precision, or 38% at 50% — and that is with twenty words, not a hundred.

Three things were tried and none moved the number: lowering the 0.86 floor, recalibrating
`DISTANCE_SCALE` (0.18 saturates every cross-signer score to zero), and dropping the raw
wrist position from the vector on the theory that it encodes framing rather than sign.
Top-1 stayed at 43.3% throughout.

The cause is structural. `windowSignature` is built for one person repeating their own
sign — same body, same camera, same framing — and has no learned invariance across signers.
Making this work needs either many more signers per sign or enough examples to train on;
six of each is neither. Do not re-run this hoping for a different answer: re-run it only
with a corpus that has substantially more signers per sign.


The trained GRU needs thousands of examples; `PrototypeSignClassifier` needs a handful, which
is the whole reason the teach-your-own-sign mode works. A corpus with six recordings per sign
is far too small to train on and comfortably enough to match against — so the question is not
whether it can be *trained*, but whether the signs are separable at all by nearest prototype.

Answered honestly by holding a whole signer out: prototypes come from the other signers only,
so a score here is generalisation to a person the pack has never seen, not memorisation.

Mirrors `windowSignature.ts` and `PrototypeSignClassifier.ts` — 8 resampled frames, both
hands, Gaussian similarity, and the same 0.86 floor the app applies.

    python prototype_pack.py /path/to/videos [--limit N]

Landmarks are cached beside this file so the second run is instant. The corpus itself is
neither committed nor redistributed; see `check_calse.py`.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
from pathlib import Path

import numpy as np

from features import SIGNATURE_LENGTH, window_signature

DISTANCE_SCALE = 0.18
MIN_SIMILARITY = 0.86
MODEL = "../../public/models/hand_landmarker.task"


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Port of `windowSignature.ts`. `b` may be a stack, giving one score per prototype."""
    b = np.atleast_2d(b)
    if a.size == 0:
        return np.zeros(len(b))
    distance = np.sqrt(((b - a) ** 2).mean(axis=1))
    energy_a = (a**2).sum()
    energy_b = (b**2).sum(axis=1)
    scores = np.exp(-((distance / DISTANCE_SCALE) ** 2))
    # "I saw no hand" must not match "I saw no hand" perfectly.
    scores[(energy_b < 1e-9)] = 0.0
    return scores if energy_a >= 1e-9 else np.zeros_like(scores)


def extract(paths: list[str], cache: Path) -> dict[str, np.ndarray]:
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        return {k: stored[k] for k in stored.files}

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            num_hands=2,
            running_mode=vision.RunningMode.IMAGE,
        )
    )

    out: dict[str, np.ndarray] = {}
    for index, path in enumerate(paths, 1):
        capture = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            found = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            )
            right = np.zeros((21, 3), dtype=np.float32)
            left = np.zeros((21, 3), dtype=np.float32)
            for slot, points in enumerate(found.hand_landmarks):
                array = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
                # MediaPipe labels the mirrored view, so its "left" is the signer's right.
                if found.handedness[slot][0].category_name.lower() == "left":
                    right = array
                else:
                    left = array
            if right.any() or left.any():
                frames.append(np.stack([right, left]))
        capture.release()
        out[os.path.basename(path)] = np.array(frames, dtype=np.float32)
        if index % 25 == 0:
            print(f"  extraídos {index}/{len(paths)}", flush=True)

    np.savez_compressed(cache, **out)
    return out


def signature_of(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros(SIGNATURE_LENGTH, dtype=np.float32)
    listed = [
        {"right": f[0] if f[0].any() else None, "left": f[1] if f[1].any() else None}
        for f in frames
    ]
    return window_signature(listed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="words to use (0 = all)")
    arguments = parser.parse_args()

    paths = sorted(glob.glob(f"{arguments.source}/*.mp4"))
    if not paths:
        parser.error(f"no .mp4 files under {arguments.source}")

    def word_of(name: str) -> str:
        return name.split("_", 1)[1].rsplit("_", 1)[0]

    if arguments.limit:
        keep = sorted({word_of(os.path.basename(p)) for p in paths})[: arguments.limit]
        paths = [p for p in paths if word_of(os.path.basename(p)) in keep]

    cache = Path(f".cache_prototypes_{len(paths)}.npz")
    print(f"{len(paths)} vídeos; caché en {cache}")
    landmarks = extract(paths, cache)

    signatures, words, signers = [], [], []
    for name, frames in landmarks.items():
        signatures.append(signature_of(frames))
        words.append(word_of(name))
        signers.append(name.split("_")[0])
    signatures = np.stack(signatures)
    words = np.array(words)
    signers = np.array(signers)

    vocabulary = sorted(set(words))
    print(f"{len(vocabulary)} signos, {len(set(signers))} signantes\n")
    print(f"{'fuera':>10} {'top-1':>8} {'habla':>8} {'acierta al hablar':>19}")

    top1_all, spoke_all, right_all = [], [], []
    for held in sorted(set(signers), key=lambda s: int(s[6:])):
        test = signers == held
        train = ~test
        hits = spoke = correct = 0
        for query, truth in zip(signatures[test], words[test]):
            best_word, best_score = None, 0.0
            for word in vocabulary:
                pool = signatures[train & (words == word)]
                if len(pool) == 0:
                    continue
                # A sign scores as its best example, never its average — as the app does.
                score = float(similarity(query, pool).max())
                if score > best_score:
                    best_word, best_score = word, score
            hits += best_word == truth
            if best_score >= MIN_SIMILARITY:
                spoke += 1
                correct += best_word == truth
        n = int(test.sum())
        top1_all.append(hits / n)
        spoke_all.append(spoke / n)
        right_all.append(correct / max(spoke, 1))
        print(
            f"{held:>10} {hits / n * 100:7.1f}% {spoke / n * 100:7.1f}% "
            f"{correct / max(spoke, 1) * 100:18.1f}%"
        )

    print()
    print(f"top-1 medio (signante nuevo)     : {np.mean(top1_all) * 100:.1f}%")
    print(f"llega al umbral 0.86             : {np.mean(spoke_all) * 100:.1f}%")
    print(f"precisión de lo que sí diría     : {np.mean(right_all) * 100:.1f}%")
    print(f"azar con {len(vocabulary)} signos: {100 / len(vocabulary):.1f}%")


if __name__ == "__main__":
    main()
