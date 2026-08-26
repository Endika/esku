"""Cut LSE-FS-UVigo's held-out split down to what the alphabet engine actually reads.

The corpus ships as two 7.3 GB zips of MediaPipe keypoints — pose, both hands and 468 face
points per frame. `HandshapeAlphabetClassifier` reads one hand and nothing else, so carrying
the rest into the bench would mean parsing gigabytes to throw them away on every run.

Only the `test` split, and deliberately: train (2,158) and validation (430) stay untouched so
they remain usable for training a handshape classifier if the measurement says one is needed.
Measuring on them would burn that.

No handedness inversion. The footage is third person, not a selfie mirror, so MediaPipe's own
label is already anatomical — the same rule as LSE-Health, and the belief that got this wrong
in four places at once is documented in `README.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data" / "uvigo"
ARCHIVE = DATA / "RAW_KPS.zip"
OUT = DATA / "lsefs_test.ndjson"
SPLIT = "RAW_KPS/test/"
EXPECTED_SEQUENCES = 456

#: `donaciones_p31_test_242.json` — the metadata block carries `signer_id: null`, so the
#: filename is the only place the signer survives.
SIGNER = re.compile(r"_(p\d+)_")

#: Enough to separate two knuckles at any sane camera distance, and it roughly halves the file.
PLACES = 4


def hand(points: list[dict] | None) -> list[list[float]] | None:
    """21 points as flat triples, or None when MediaPipe saw no hand in this frame.

    An absent hand arrives as 21 points of `null` rather than as a missing key. A *partly*
    null hand would be a different thing entirely — a shape half-invented — so it is not
    silently coerced here; `main` counts them and says so.
    """
    if not points:
        return None
    if any(p.get("x") is None for p in points):
        return None
    return [[round(p["x"], PLACES), round(p["y"], PLACES), round(p["z"], PLACES)] for p in points]


def partial(points: list[dict] | None) -> bool:
    """A hand with some coordinates present and some missing, which should never happen."""
    if not points:
        return False
    nulls = sum(p.get("x") is None for p in points)
    return 0 < nulls < len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="usar solo las primeras N secuencias")
    arguments = parser.parse_args()

    if not ARCHIVE.is_file():
        raise SystemExit(f"falta {ARCHIVE}")

    with zipfile.ZipFile(ARCHIVE) as book:
        names = sorted(
            n for n in book.namelist() if n.startswith(SPLIT) and n.endswith(".json")
        )
        if arguments.limit:
            names = names[: arguments.limit]
        elif len(names) != EXPECTED_SEQUENCES:
            raise SystemExit(
                f"{len(names)} secuencias en {SPLIT}, se esperaban {EXPECTED_SEQUENCES}; "
                "el zip no es el publicado y las cifras no comparan"
            )

        frames_total = 0
        by_hands = {0: 0, 1: 0, 2: 0}
        partials = 0
        labels: set[str] = set()

        with OUT.open("w", encoding="utf-8") as out:
            for name in names:
                document = json.loads(book.read(name))
                meta = document["metadata"]
                found = SIGNER.search(Path(name).name)

                frames = []
                for raw in document["frames"]:
                    left_points, right_points = raw.get("left_hand"), raw.get("right_hand")
                    partials += partial(left_points and left_points.get("keypoints"))
                    partials += partial(right_points and right_points.get("keypoints"))
                    left = hand(left_points and left_points.get("keypoints"))
                    right = hand(right_points and right_points.get("keypoints"))
                    by_hands[(left is not None) + (right is not None)] += 1
                    frames.append({"left": left, "right": right})

                frames_total += len(frames)
                labels.add(meta["label"])
                out.write(
                    json.dumps(
                        {
                            "segment_id": meta["segment_id"],
                            "label": meta["label"],
                            "signer": found.group(1) if found else None,
                            "frames": frames,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    print("LSE-FS-UVigo, split de test: extracto solo de manos")
    print()
    print(f"   secuencias escritas   : {len(names)}")
    print(f"   etiquetas distintas   : {len(labels)}")
    print(f"   frames totales        : {frames_total}")
    for count in (0, 1, 2):
        share = by_hands[count] / frames_total * 100 if frames_total else 0.0
        print(f"   frames con {count} mano(s)  : {by_hands[count]:>8}  {share:>5.1f}%")
    print(f"   manos a medias        : {partials}")
    print()
    print(f"   escrito en {OUT.relative_to(Path(__file__).parent)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    if partials:
        print()
        print("   AVISO: hay manos con solo parte de los puntos. Se tratan como ausentes, que")
        print("   es lo conservador, pero conviene mirar por que el corpus las trae asi.")


if __name__ == "__main__":
    main()
