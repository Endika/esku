"""LSE-FS-UVigo as per-frame handshape features, one `.npz` per split.

The corpus labels the spelled *word* and gives no per-letter timings, so there is no per-frame
supervision to extract — that is CTC's job in `lsefs_train.py`. What this writes is the input
side: one normalised hand per frame, and the letter sequence the frames have to explain.

Only the dominant hand, chosen by largest span, because that is the rule `dominantHand` applies
in the app and a bench that picks differently measures a different pipeline. Normalisation is
`features.normalize_hand`, the same transform `normalizeHand.ts` performs live.

The wrist's own position is deliberately **not** appended, unlike `hand_features`. Position is
what distinguishes two signs made with the same handshape at forehead and chest; a fingerspelled
letter is the same letter anywhere in frame, and feeding position would invite the model to learn
where this corpus's signers happen to hold their hands.

No handedness inversion: third-person footage, so MediaPipe's label is already anatomical.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from features import normalize_hand

DATA = Path(__file__).parent / "data" / "uvigo"
ARCHIVE = DATA / "RAW_KPS.zip"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÑ"
#: Blank owns index 0, so a letter's class is its position here plus one.
BLANK = 0
#: The corpus spells a handful of words with accents the alphabet has no handshape for.
FOLD = {"É": "E", "Á": "A", "Í": "I", "Ó": "O", "Ú": "U", "Ü": "U"}
#: Digits, spaces and the `@` of spelled email addresses are outside the model's alphabet. They
#: stay in the label the bench scores against — the exam is not softened to flatter the score.
HAND_FLOATS = 63

EXPECTED = {"train": 2158, "validation": 430, "test": 456}


def span(points: np.ndarray) -> float:
    """Bounding-box area, the app's `dominantHand` tie-breaker."""
    return float(
        (points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())
    )


#: Matches `DECAY` in `dominantHandTracker.ts`. The two rules must agree or the model is
#: trained on one hand and shown another.
DECAY = 0.9


def visible(frame: dict) -> dict[str, np.ndarray]:
    """Both hands MediaPipe actually saw, keyed by side.

    An absent hand arrives as 21 points of `null` rather than a missing key. A partly-null hand
    is rejected outright: a shape with invented coordinates is worse than no shape at all.
    """
    out: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        block = frame.get(f"{side}_hand")
        points = block and block.get("keypoints")
        if not points or any(p.get("x") is None for p in points):
            continue
        out[side] = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float32)
    return out


def dominant(frame: dict) -> tuple[np.ndarray, str] | None:
    """Largest span, the stateless rule — kept for the audit that showed it is not good enough."""
    present = visible(frame)
    if not present:
        return None
    side = max(present, key=lambda s: span(present[s]))
    return present[side], side


def target(label: str) -> list[int]:
    folded = "".join(FOLD.get(c, c) for c in label.upper())
    return [ALPHABET.index(c) + 1 for c in folded if c in ALPHABET]


def cut(book: zipfile.ZipFile, name: str) -> tuple[np.ndarray, np.ndarray, str, int] | None:
    document = json.loads(book.read(name))
    label = document["metadata"]["label"]
    letters = target(label)
    if not letters:
        return None

    rows = []
    seen = 0
    motion: dict[str, float] = {}
    previous: dict[str, np.ndarray] = {}
    for frame in document["frames"]:
        present = visible(frame)
        # Accumulated wrist motion, decayed — the signing hand moves and a resting one does not.
        # The stateless largest-span rule takes the wrong hand in 23.7% of two-hand frames, and
        # because the model carries hidden state that error propagates rather than staying put.
        for side, points in present.items():
            before = previous.get(side)
            travelled = float(np.linalg.norm(points[0, :2] - before[0, :2])) if before is not None else 0.0
            motion[side] = motion.get(side, 0.0) * DECAY + travelled
        previous = present

        if not present:
            rows.append(np.zeros(HAND_FLOATS, dtype=np.float32))
            continue
        if len(present) == 1:
            side = next(iter(present))
        else:
            side = max(present, key=lambda s: motion.get(s, 0.0))
            if motion.get(side, 0.0) <= 0.0:
                side = max(present, key=lambda s: span(present[s]))
        rows.append(normalize_hand(present[side], side).reshape(-1).astype(np.float32))
        seen += 1

    # CTC cannot align more labels than timesteps, and a sequence that short is corrupt anyway.
    if len(rows) < len(letters):
        return None
    return np.stack(rows), np.array(letters, dtype=np.int64), label, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="usar solo las primeras N de cada split")
    arguments = parser.parse_args()

    if not ARCHIVE.is_file():
        raise SystemExit(f"falta {ARCHIVE}")

    print("LSE-FS-UVigo -> rasgos de handshape por frame")
    print()
    with zipfile.ZipFile(ARCHIVE) as book:
        for split, expected in EXPECTED.items():
            names = sorted(
                n for n in book.namelist() if f"/{split}/" in n and n.endswith(".json")
            )
            if arguments.limit:
                names = names[: arguments.limit]
            elif len(names) != expected:
                raise SystemExit(
                    f"{len(names)} secuencias en {split}, se esperaban {expected}; el zip no es "
                    "el publicado y las cifras no comparan"
                )

            xs, ys, lengths, tlengths, labels = [], [], [], [], []
            frames = hands = dropped = 0
            for name in names:
                cutting = cut(book, name)
                if cutting is None:
                    dropped += 1
                    continue
                x, y, label, seen = cutting
                xs.append(x)
                ys.append(y)
                lengths.append(len(x))
                tlengths.append(len(y))
                labels.append(label)
                frames += len(x)
                hands += seen

            out = DATA / f"lsefs_{split}.npz"
            np.savez_compressed(
                out,
                x=np.concatenate(xs),
                y=np.concatenate(ys),
                lengths=np.array(lengths, dtype=np.int64),
                tlengths=np.array(tlengths, dtype=np.int64),
                labels=np.array(labels),
            )
            print(
                f"   {split:<11} {len(lengths):>5} secuencias  {frames:>7} frames  "
                f"{hands / frames * 100:>4.1f}% con mano  {sum(tlengths):>6} letras"
                + (f"  ({dropped} descartadas)" if dropped else "")
            )

    print()
    print(f"   alfabeto: {len(ALPHABET)} letras, blank en el indice {BLANK}")


if __name__ == "__main__":
    main()
