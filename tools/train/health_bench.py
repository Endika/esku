"""Score the segmenter against hand-annotated boundaries in real continuous signing.

Every segmenter number this project has ever quoted comes from one of two places, and both
flatter it. SWL-LSE is isolated signs with stillness either side, which is exactly the shape
the segmenter's deceleration rule was tuned on. `continuous.py` splices those recordings back
to back, and its own docstring calls that an upper bound: concatenation never produces the
transitional movement where one sign's hand travels into the next one's start.

LSE-Health-UVigo is the missing third case — 11 h of fluent LSE from 10 signers, with 15,098
gloss instances timed by hand. Nothing here is a candidate boundary invented by us: the
intervals are the annotators'. So this is the first place where "the segmenter recovers 38.4%
of signs" can be either confirmed or exposed as an artefact of spliced audio-free clips.

Three questions, because a segmenter can fail three ways and recall alone hides two of them:

  1. boundaries — does an emitted window land on an annotated gloss at all (temporal IoU)?
  2. rhythm — does it emit roughly one window per sign, or does it over-fire? Rhythm is what
     caught the CALSE100 overfit: 2.56 windows where exactly 1 was correct.
  3. length — is the window length distribution anywhere near the real one (median 440 ms)?
     If windows pile up on the shortest span the rules can produce, the deceleration test is
     not choosing anything: MIN_SIGN_MS is, and window length is a constant dressed up as a
     measurement. That would make every recall number above a statement about where a fixed
     grid happens to fall, not about where signs end. See `floor_ms` — the floor is *not*
     MIN_SIGN_MS, so this has to be checked against the rate-dependent value.

The corpus is not in this repository and is not redistributed by it. This reads a local
landmark cache built elsewhere and reports how much of it it could actually score.

    .venv/bin/python health_bench.py [--iou 0.5] [--videos N] [--only=ID,ID] [--per-signer]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

import simulate_app as sim

DATA = Path("data/uvigo")
WORKBOOK = DATA / "10234465_LSE-Health-UVigo.xlsx"
VIDEOS = DATA / "videos.csv"
CACHE = DATA / "cache"

#: Row counts published with the corpus. Asserted rather than trusted: a silently truncated
#: parse would quietly shrink the denominator of every recall figure below.
EXPECTED_GLOSSES = 15098
EXPECTED_SEGMENTS = 7738

#: 0.5 is the primary threshold, declared here so it cannot be picked after seeing the result.
#: The looser two are reported alongside it, always, for the same reason.
THRESHOLDS = (0.5, 0.3, 0.1)

#: Same guard as `check_calse.py`: the corpus is documented at 25 fps, and a container that
#: declares 0 or something faster than any camera would poison every wall-clock span.
FALLBACK_FPS = 25.0
MAX_PLAUSIBLE_FPS = 120.0

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Temporal intersection over union of two [start, end) spans in milliseconds."""
    overlap = min(first[1], second[1]) - max(first[0], second[0])
    if overlap <= 0:
        return 0.0
    union = (first[1] - first[0]) + (second[1] - second[0]) - overlap
    return overlap / union if union > 0 else 0.0


def match(
    glosses: list[tuple[float, float]],
    windows: list[tuple[float, float]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one pairing of glosses to windows, best IoU first.

    One-to-one is the whole point. Letting one long window claim every gloss it straddles
    would report a segmenter that emits a single window per sentence as perfect recall, and
    letting one gloss absorb a burst of windows would hide over-firing from the precision.
    """
    scored: list[tuple[float, int, int]] = []
    for g, gloss in enumerate(glosses):
        for w, window in enumerate(windows):
            score = iou(gloss, window)
            if score > 0 and score >= threshold:
                scored.append((score, g, w))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))

    used_glosses: set[int] = set()
    used_windows: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, g, w in scored:
        if g in used_glosses or w in used_windows:
            continue
        used_glosses.add(g)
        used_windows.add(w)
        pairs.append((g, w, score))
    return pairs


def sheet_rows(book: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    """Every row of one worksheet as {column letter: value}, shared strings resolved.

    Resolved by sheet *name* through the relationship table, not by file index: the ordering
    of `xl/worksheets/sheetN.xml` is an implementation detail of whoever saved the file.
    """
    shared = [
        "".join(node.text or "" for node in item.iter(f"{NS}t"))
        for item in ET.fromstring(book.read("xl/sharedStrings.xml"))
    ]
    targets = {
        rel.get("Id"): rel.get("Target")
        for rel in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    }
    paths = {
        sheet.get("name"): targets[sheet.get(f"{RELS}id")]
        for sheet in ET.fromstring(book.read("xl/workbook.xml")).iter(f"{NS}sheet")
    }
    if name not in paths:
        raise KeyError(f"no sheet named {name} in {WORKBOOK}")

    rows: list[dict[str, str]] = []
    for row in ET.fromstring(book.read(f"xl/{paths[name]}")).iter(f"{NS}row"):
        cells: dict[str, str] = {}
        for cell in row.iter(f"{NS}c"):
            value = cell.find(f"{NS}v")
            if value is None or value.text is None:
                continue
            column = "".join(c for c in (cell.get("r") or "") if c.isalpha())
            cells[column] = shared[int(value.text)] if cell.get("t") == "s" else value.text
        if cells:
            rows.append(cells)
    return rows


def load_annotations() -> tuple[dict[str, list[tuple[float, float, str]]], int, int]:
    """Gloss intervals per video id, plus the raw gloss and segment row counts."""
    with zipfile.ZipFile(WORKBOOK) as book:
        glosses = sheet_rows(book, "GlossesContent")[1:]
        segments = len(sheet_rows(book, "SegmentsContent")) - 1

    by_video: dict[str, list[tuple[float, float, str]]] = {}
    for row in glosses:
        video, start, end = row.get("A"), row.get("B"), row.get("C")
        if not video or start is None or end is None:
            continue
        by_video.setdefault(video, []).append((float(start), float(end), row.get("D", "")))
    for spans in by_video.values():
        spans.sort()
    return by_video, len(glosses), segments


def load_videos() -> dict[str, dict[str, str]]:
    with VIDEOS.open() as handle:
        return {row["video"]: row for row in csv.DictReader(handle)}


def window_spans(right: np.ndarray, left: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """Emitted windows as absolute [start, end) millisecond spans.

    Frame indices become wall clock through the *file's* own rate. Feeding a 25 fps corpus to
    the 20 fps default reads every window 25% long, which moves MIN_SIGN_MS and MAX_MS earlier
    in real time and invents extra windows — the reason `sim.windows` takes fps at all.
    """
    frame_ms = 1000.0 / fps
    spans = []
    for window in sim.windows(right, left, fps=fps):
        # `sim.span_ms` convention, so block 3 compares like with like against MIN_SIGN_MS.
        spans.append((window[0] * frame_ms, window[-1] * frame_ms))
    return spans


def score_video(path: Path, glosses: list[tuple[float, float, str]]) -> dict | None:
    """Windows, in-cache glosses and duration for one cached video, or None if unreadable."""
    try:
        # Closed eagerly: 273 leaked NpzFile handles would exhaust the descriptor table.
        with np.load(path) as bundle:
            right = bundle["right"]
            left = bundle["left"]
            declared = float(bundle["fps"][0])
            frames = int(bundle["frames"][0]) if "frames" in bundle else len(right)
    except (OSError, KeyError, ValueError, IndexError) as error:
        print(f"   ilegible: {path.name} ({error})")
        return None
    if len(right) == 0:
        return None
    fps = declared if 0.0 < declared <= MAX_PLAUSIBLE_FPS else FALLBACK_FPS
    if fps != declared:
        print(f"   {path.stem}: fps absurdo ({declared}), leido a {fps:g}")

    duration_ms = frames * 1000.0 / fps
    limit = duration_ms + 1000.0 / fps
    inside = [(start, end) for start, end, _ in glosses if end <= limit]

    return {
        "windows": window_spans(right, left, fps),
        "glosses": inside,
        "outside": len(glosses) - len(inside),
        "duration_ms": duration_ms,
        "fps": fps,
    }


def percentiles(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
        float(array.max()),
    )


def rate_per_minute(count: int, duration_ms: float) -> float:
    return count / (duration_ms / 60000.0) if duration_ms > 0 else 0.0


def report_boundaries(
    per_video: dict[str, dict], thresholds: list[float], primary: float
) -> None:
    total_glosses = sum(len(v["glosses"]) for v in per_video.values())
    total_windows = sum(len(v["windows"]) for v in per_video.values())

    print("1. FRONTERAS  (emparejamiento uno a uno, IoU temporal, greedy por IoU descendente)")
    print(f"   glosas puntuadas {total_glosses}   ventanas emitidas {total_windows}")
    print()
    print(f"   {'umbral IoU':>12} {'emparejadas':>12} {'recall':>9} {'precision':>10}")
    for threshold in thresholds:
        paired = sum(
            len(match(v["glosses"], v["windows"], threshold)) for v in per_video.values()
        )
        recall = paired / total_glosses * 100 if total_glosses else 0.0
        precision = paired / total_windows * 100 if total_windows else 0.0
        mark = "   <- principal" if threshold == primary else ""
        print(f"   {threshold:>12.2f} {paired:>12} {recall:>8.1f}% {precision:>9.1f}%{mark}")
    print()


def report_rhythm(
    per_video: dict[str, dict], videos: dict[str, dict[str, str]], per_signer: bool
) -> None:
    print("2. RITMO  (una ventana por signo es lo correcto; mas es sobresegmentar)")
    print()

    def line(label: str, count: int, glosses: int, windows: int, duration: float) -> None:
        gpm = rate_per_minute(glosses, duration)
        wpm = rate_per_minute(windows, duration)
        ratio = wpm / gpm if gpm > 0 else 0.0
        print(
            f"   {label:>10} {count:>7} {duration / 60000.0:>9.1f} "
            f"{gpm:>12.2f} {wpm:>13.2f} {ratio:>7.2f}"
        )

    header = (
        f"   {'signante':>10} {'videos':>7} {'minutos':>9} "
        f"{'glosas/min':>12} {'ventanas/min':>13} {'ratio':>7}"
    )
    print(header)

    if per_signer:
        buckets: dict[str, list[dict]] = {}
        for video, scored in per_video.items():
            signer = videos.get(video, {}).get("signer", "?")
            buckets.setdefault(signer, []).append(scored)
        for signer in sorted(buckets, key=lambda s: (int(s) if s.isdigit() else 1e9, s)):
            group = buckets[signer]
            line(
                signer,
                len(group),
                sum(len(g["glosses"]) for g in group),
                sum(len(g["windows"]) for g in group),
                sum(g["duration_ms"] for g in group),
            )
        print()

    line(
        "TOTAL",
        len(per_video),
        sum(len(v["glosses"]) for v in per_video.values()),
        sum(len(v["windows"]) for v in per_video.values()),
        sum(v["duration_ms"] for v in per_video.values()),
    )
    print()


def floor_ms(fps: float) -> float:
    """Shortest window the segmenter can possibly emit at this frame rate.

    Not MIN_SIGN_MS. A window only starts accumulating `slow_ms` once its span already clears
    MIN_SIGN_MS, and it needs DECELERATION_HOLD_MS of that before closing — both in whole
    frames. At 25 fps that makes the real floor 1200 ms, not 1150, so comparing the observed
    median against MIN_SIGN_MS would miss a distribution sitting exactly on the floor.
    """
    frame_ms = 1000.0 / fps
    first = math.ceil(sim.MIN_SIGN_MS / frame_ms) * frame_ms
    holds = max(1, math.ceil(sim.DECELERATION_HOLD_MS / frame_ms))
    return first + (holds - 1) * frame_ms


def report_lengths(per_video: dict[str, dict]) -> None:
    window_ms = [end - start for v in per_video.values() for start, end in v["windows"]]
    gloss_ms = [end - start for v in per_video.values() for start, end in v["glosses"]]

    print("3. LONGITUD  (ms)")
    print()
    print(f"   {'fuente':>22} {'n':>7} {'mediana':>9} {'p25':>7} {'p75':>7} {'max':>8}")
    for label, values in (("glosa anotada", gloss_ms), ("ventana emitida", window_ms)):
        median, p25, p75, top = percentiles(values)
        print(
            f"   {label:>22} {len(values):>7} {median:>9.0f} {p25:>7.0f} "
            f"{p75:>7.0f} {top:>8.0f}"
        )
    print()

    if not window_ms:
        return
    median = float(np.median(window_ms))
    gloss_median = float(np.median(gloss_ms)) if gloss_ms else 0.0

    rates = [v["fps"] for v in per_video.values()]
    lowest = floor_ms(float(np.median(rates)))
    at_floor = sum(1 for span in window_ms if abs(span - lowest) <= 1.0)
    # Only the end-of-recording flush skips MIN_SIGN_MS, so these are one per video at most.
    below = sum(1 for span in window_ms if span < lowest - 1.0)
    share = at_floor / len(window_ms) * 100
    print(
        f"   suelo real a {np.median(rates):g} fps: {lowest:.0f} ms "
        f"(MIN_SIGN_MS {sim.MIN_SIGN_MS} + espera, en fotogramas enteros)"
    )
    print(
        f"   ventanas cerradas en ese suelo: {at_floor} de {len(window_ms)} ({share:.0f}%)"
    )
    if below:
        print(f"   por debajo del suelo, cierre de fin de grabacion: {below}")
    print()

    if share >= 50.0:
        print("   AVISO: la mayoria de las ventanas cierran en el suelo minimo. El criterio de")
        print("   deceleracion no esta eligiendo el final del signo: lo elige MIN_SIGN_MS, y")
        print("   la longitud de ventana es una constante, no una medida. Cualquier recall de")
        print("   arriba mide 'donde cae una rejilla de 1,2 s', no 'donde acaba un signo'.")
    if gloss_median > 0:
        print(
            f"   La ventana mediana ({median:.0f} ms) es {median / gloss_median:.1f}x la "
            f"glosa mediana ({gloss_median:.0f} ms)."
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="umbral IoU principal (por defecto 0.5; 0.3 y 0.1 se reportan igualmente)",
    )
    parser.add_argument("--videos", type=int, help="puntuar solo los primeros N videos")
    parser.add_argument(
        "--only",
        help="ids de video separados por comas; usa --only=ID porque muchos empiezan por '-'",
    )
    parser.add_argument(
        "--per-signer", action="store_true", help="desglosar el ritmo por signante"
    )
    arguments = parser.parse_args()

    annotations, gloss_rows, segment_rows = load_annotations()
    videos = load_videos()

    print("LSE-Health-UVigo — banco de pruebas del segmentador sobre signado continuo real")
    print()
    print(f"   filas de glosa leidas   : {gloss_rows}  (esperadas {EXPECTED_GLOSSES})")
    print(f"   filas de segmento leidas: {segment_rows}  (esperadas {EXPECTED_SEGMENTS})")
    if gloss_rows != EXPECTED_GLOSSES or segment_rows != EXPECTED_SEGMENTS:
        raise SystemExit("el parseo del xlsx no cuadra con el corpus publicado; no continuo")
    print(f"   videos en videos.csv    : {len(videos)}")
    print(f"   videos con anotacion    : {len(annotations)}")
    print()

    if not CACHE.is_dir():
        raise SystemExit(
            f"no hay cache de landmarks en {CACHE}. El banco esta listo: vuelve a "
            f"ejecutarlo cuando existan los .npz (contrato: right/left/pose/face/fps/"
            f"frames/inverted)."
        )

    index = CACHE / "index.json"
    if index.is_file():
        try:
            entries = json.loads(index.read_text())
            count = len(entries) if isinstance(entries, (list, dict)) else "?"
            print(f"   cache/index.json declara {count} entradas")
        except (OSError, ValueError) as error:
            print(f"   cache/index.json ilegible ({error}); sigo por los .npz del directorio")

    cached = sorted(CACHE.glob("*.npz"))
    available = [path for path in cached if path.stem in annotations]
    unannotated = [path.stem for path in cached if path.stem not in annotations]
    if unannotated:
        print(f"   en cache pero sin anotacion, no puntuables: {', '.join(unannotated)}")
    if not available:
        raise SystemExit(
            f"{CACHE} no contiene ningun .npz con anotacion. El banco esta listo: "
            f"vuelve a ejecutarlo cuando el cache tenga contenido."
        )

    if arguments.only:
        wanted = {name.strip() for name in arguments.only.split(",") if name.strip()}
        available = [path for path in available if path.stem in wanted]
        missing = wanted - {path.stem for path in available}
        if missing:
            print(f"   sin cache o sin anotacion: {', '.join(sorted(missing))}")
    if arguments.videos:
        available = available[: arguments.videos]
    if not available:
        raise SystemExit("la seleccion no deja ningun video que puntuar")

    per_video: dict[str, dict] = {}
    for path in available:
        scored = score_video(path, annotations[path.stem])
        if scored is not None:
            per_video[path.stem] = scored

    if not per_video:
        raise SystemExit("ningun .npz del cache resulto puntuable")

    rates = sorted({round(v["fps"], 2) for v in per_video.values()})
    outside = sum(v["outside"] for v in per_video.values())
    scored_glosses = sum(len(v["glosses"]) for v in per_video.values())
    print(
        f"   COBERTURA: {len(per_video)} de {len(videos)} videos "
        f"({len(per_video) / len(videos) * 100:.1f}%), "
        f"{scored_glosses} de {gloss_rows} instancias de glosa "
        f"({scored_glosses / gloss_rows * 100:.1f}%)"
    )
    print(f"   fps reales en el cache: {', '.join(f'{r:g}' for r in rates)}")
    if outside:
        print(f"   glosas descartadas por caer fuera del tramo cacheado: {outside}")
    print()

    thresholds = sorted({arguments.iou, *THRESHOLDS}, reverse=True)
    report_boundaries(per_video, thresholds, arguments.iou)
    report_rhythm(per_video, videos, arguments.per_signer)
    report_lengths(per_video)

    print("Referencia: 38.4% de recall venia de continuous.py, que concatena signos aislados")
    print("y su propio docstring declara cota superior. Lo de arriba es signado continuo.")


if __name__ == "__main__":
    main()
