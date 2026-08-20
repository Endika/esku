"""Cut LSE-Health into training windows of the kind production actually sees.

The shipped classifier scores 0.8% word recall on real continuous signing because it has only
ever seen dictionary recordings. Its annotations fix that: 15,098 instances with start, end and
label. But cutting only at those gold boundaries would repeat the mistake this project keeps
making — training on a path production does not take. Live, the classifier is handed whatever
`SignSegmenter` closes: a window of roughly constant length, rarely aligned to a sign. So cut
three ways and keep them separable:

- **gold**: the annotated interval exactly. Clean labels, wrong shape.
- **segmenter**: the windows `sim.windows` really emits, labelled by the gloss they contain.
  Right shape, noisier labels. This is the one that matters.
- **reject**: windows in the *gaps between annotated sentence segments*, i.e. moments when
  nobody is signing.

That last source needs care, and the obvious version of it is wrong. "Overlaps no gloss" does
not mean "no sign": only 101 sign types are annotated, about 24 glosses per minute against the
90-150 signs a minute of fluent discourse, so a window overlapping no gloss is usually a real
sign nobody labelled. Training a reject class on those would teach the model to reject signing.
The sentence tier is the honest source — it covers 9.6 hours of translated speech, so the gaps
between its segments are genuine pauses.

Windows whose containment of a gloss falls between the two thresholds are **dropped**, not
forced into a label. An ambiguous window mislabelled is worse than one left out.
"""

from __future__ import annotations

import argparse
import collections
import json
import zipfile

import numpy as np

import health_bench as hb
import simulate_app as sim

SPLIT = hb.CACHE.parent / "split.json"
#: A segmenter window earns a gloss's label when it holds at least this much of the gloss.
#: Containment of the *gloss*, not IoU: a 1,200 ms window over a 400 ms sign holds all of it and
#: is a fine training example, while IoU would score that 0.33 and throw it away.
LABEL_CONTAINMENT = 0.5
#: Below this, the window is treated as not carrying the sign. Between the two it is dropped.
AMBIGUOUS_CONTAINMENT = 0.2
REJECT_LABEL = "__NADA__"


def overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def sentence_spans() -> dict[str, list[tuple[float, float]]]:
    """Translated-sentence intervals per video: the only honest evidence of *not* signing.

    `sheet_rows` keys cells by column letter, so this reads A/B/C and skips the header row.
    """
    with zipfile.ZipFile(hb.WORKBOOK) as book:
        rows = hb.sheet_rows(book, "SegmentsContent")[1:]
    spans: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in rows:
        video, start, end = row.get("A"), row.get("B"), row.get("C")
        if not video or start is None or end is None:
            continue
        spans[video].append((float(start), float(end)))
    for key in spans:
        spans[key].sort()
    return spans


def cut(video: str, glosses, sentences, floor: int, sources: set[str]):
    """(right, left, pose, face, label, source) per window kept for this video."""
    path = hb.CACHE / f"{video}.npz"
    if not path.is_file():
        return None
    with np.load(path) as bundle:
        if "pose" not in bundle or "face" not in bundle:
            return None
        right, left = bundle["right"], bundle["left"]
        pose, face = bundle["pose"], bundle["face"]
        declared = float(bundle["fps"][0])
    if len(right) == 0:
        return None
    fps = declared if 0.0 < declared <= hb.MAX_PLAUSIBLE_FPS else hb.FALLBACK_FPS
    frame_ms = 1000.0 / fps

    out = []

    def take(frames: list[int], label: str, source: str) -> None:
        if len(frames) < sim.MIN_FRAMES:
            return
        k = np.array(frames)
        out.append((right[k], left[k], pose[k], face[k], label, source))

    if "gold" in sources:
        for start, end, label in glosses:
            frames = [i for i in range(len(right)) if start <= i * frame_ms <= end]
            take(frames, label, "gold")

    if {"segmenter", "reject"} & sources:
        sim.MIN_SIGN_MS = floor
        for window in sim.windows(right, left, fps=fps):
            span = (window[0] * frame_ms, window[-1] * frame_ms)
            best_label, best_share = None, 0.0
            for start, end, label in glosses:
                length = max(end - start, 1e-6)
                share = overlap(span, (start, end)) / length
                if share > best_share:
                    best_label, best_share = label, share

            if best_share >= LABEL_CONTAINMENT:
                if "segmenter" in sources:
                    take(window, best_label, "segmenter")
            elif best_share <= AMBIGUOUS_CONTAINMENT and "reject" in sources:
                # Only a gap between *sentence* segments is honest evidence of no signing.
                if not any(overlap(span, s) > 0 for s in sentences):
                    take(window, REJECT_LABEL, "reject")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--floor", type=int, default=sim.MIN_SIGN_MS,
                        help=f"MIN_SIGN_MS del segmentador (def. {sim.MIN_SIGN_MS})")
    parser.add_argument("--sources", default="gold,segmenter,reject",
                        help="fuentes separadas por comas: gold, segmenter, reject")
    parser.add_argument("--videos", type=int, help="limitar a los primeros N de cada conjunto")
    parser.add_argument("--dry-run", action="store_true", help="contar sin escribir el .npz")
    arguments = parser.parse_args()

    if not SPLIT.is_file():
        raise SystemExit(f"falta {SPLIT}: corre primero health_split.py")
    split = json.loads(SPLIT.read_text())
    sources = {s.strip() for s in arguments.sources.split(",") if s.strip()}

    annotations, gloss_rows, segment_rows = hb.load_annotations()
    if gloss_rows != hb.EXPECTED_GLOSSES or segment_rows != hb.EXPECTED_SEGMENTS:
        raise SystemExit("el parseo del xlsx no cuadra con el corpus publicado; no continuo")
    sentences = sentence_spans()

    keep_starred = split["starred_included"]
    print("Recorte de LSE-Health en ventanas de entrenamiento")
    print()
    print(f"   suelo del segmentador : {arguments.floor} ms")
    print(f"   fuentes               : {', '.join(sorted(sources))}")
    print(f"   glosas con *          : {'incluidas' if keep_starred else 'excluidas'}")
    print(f"   contencion para etiquetar / ambigua : {LABEL_CONTAINMENT} / "
          f"{AMBIGUOUS_CONTAINMENT}")
    print()

    original = sim.MIN_SIGN_MS
    try:
        for name in ("train", "test"):
            videos = split[f"{name}_videos"]
            if arguments.videos:
                videos = videos[: arguments.videos]
            per_source = collections.Counter()
            per_label = collections.Counter()
            bundle: dict[str, np.ndarray] = {}
            labels: list[str] = []
            origin: list[str] = []
            provenance: list[str] = []
            missing = 0

            for video in videos:
                glosses = [
                    (start, end, label)
                    for start, end, label in annotations.get(video, [])
                    if keep_starred or not label.strip().startswith("*")
                ]
                cut_rows = cut(video, glosses, sentences.get(video, []), arguments.floor, sources)
                if cut_rows is None:
                    missing += 1
                    continue
                for r, l, p, f, label, source in cut_rows:
                    index = len(labels)
                    if not arguments.dry_run:
                        bundle[f"r{index}"] = r
                        bundle[f"l{index}"] = l
                        bundle[f"p{index}"] = p
                        bundle[f"f{index}"] = f
                    labels.append(label)
                    # Provenance per window, so a validation split can stay signer-disjoint
                    # too: without it the only option is a random tenth, which leaks.
                    origin.append(video)
                    provenance.append(source)
                    per_source[source] += 1
                    per_label[label] += 1

            print(f"   {name}: {len(videos) - missing} videos leidos, {missing} sin cache")
            for source in sorted(per_source):
                print(f"      {source:>10} {per_source[source]:>7}")
            print(f"      {'TOTAL':>10} {len(labels):>7}   clases {len(per_label)}")
            if REJECT_LABEL in per_label:
                share = per_label[REJECT_LABEL] / len(labels)
                print(f"      rechazo: {per_label[REJECT_LABEL]} ({share:.1%} del conjunto)")
            thin = [k for k, v in per_label.items() if v < 5 and k != REJECT_LABEL]
            if thin:
                print(f"      {len(thin)} clases con menos de 5 ejemplos")

            if not arguments.dry_run and labels:
                bundle["y"] = np.array(labels)
                bundle["video"] = np.array(origin)
                bundle["source"] = np.array(provenance)
                bundle["n"] = np.array([len(labels)])
                target = hb.CACHE.parent / f"health_{name}_raw.npz"
                np.savez(target, **bundle)
                print(f"      escrito {target}")
            print()
    finally:
        sim.MIN_SIGN_MS = original


if __name__ == "__main__":
    main()
