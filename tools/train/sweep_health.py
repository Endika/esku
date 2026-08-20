"""Sweep the window floor against real signing, because the floor is what closes the windows.

`health_bench.py` measured the thing this project had been guessing at: on LSE-Health-UVigo,
94% of windows close the instant `MIN_SIGN_MS` permits it, so window length is a constant and
the deceleration rule never gets to choose an ending. That constant came from SWL-LSE, where
the median sign runs ~1.5 s — but a dictionary recording contains the approach and the hold,
and the sign proper in continuous discourse runs ~500 ms. A floor at 1150 ms is more than twice
the sign, so no amount of tuning the criterion can matter until the floor drops below it.

So sweep the floor first, and watch two things that have to move together: recall has to rise,
and the share of windows closing *at* the floor has to fall. If recall rises while that share
stays high, the floor is still doing the choosing and the gain is just a better-sized grid.

Landmarks are read once into memory and only the segmenter re-runs per candidate, so this is
minutes rather than one pass per value.
"""

from __future__ import annotations

import argparse

import numpy as np

import health_bench as hb
import simulate_app as sim

CANDIDATES = [1150, 900, 750, 640, 550, 500, 440, 360, 300, 250, 200]


def load_streams(paths, annotations) -> list[dict]:
    """Hands and rate per video, held in memory so each candidate only re-runs the segmenter."""
    streams = []
    for path in paths:
        with np.load(path) as bundle:
            right, left = bundle["right"], bundle["left"]
            declared = float(bundle["fps"][0])
            frames = int(bundle["frames"][0]) if "frames" in bundle else len(right)
        if len(right) == 0:
            continue
        fps = declared if 0.0 < declared <= hb.MAX_PLAUSIBLE_FPS else hb.FALLBACK_FPS
        duration_ms = frames * 1000.0 / fps
        limit = duration_ms + 1000.0 / fps
        streams.append(
            {
                "name": path.stem,
                "right": right,
                "left": left,
                "fps": fps,
                "duration_ms": duration_ms,
                "glosses": [(s, e) for s, e, _ in annotations[path.stem] if e <= limit],
            }
        )
    return streams


def evaluate(streams: list[dict], thresholds: list[float]) -> dict:
    windows_total = glosses_total = 0
    duration_ms = 0.0
    matched = {t: 0 for t in thresholds}
    lengths: list[float] = []
    at_floor = 0

    for stream in streams:
        spans = hb.window_spans(stream["right"], stream["left"], stream["fps"])
        floor = hb.floor_ms(stream["fps"])
        windows_total += len(spans)
        glosses_total += len(stream["glosses"])
        duration_ms += stream["duration_ms"]
        for start, end in spans:
            length = end - start
            lengths.append(length)
            if abs(length - floor) < 1e-6:
                at_floor += 1
        for threshold in thresholds:
            matched[threshold] += len(hb.match(stream["glosses"], spans, threshold))

    minutes = duration_ms / 60000.0
    return {
        "windows": windows_total,
        "glosses": glosses_total,
        "matched": matched,
        "median_ms": float(np.median(lengths)) if lengths else 0.0,
        "at_floor": at_floor / windows_total if windows_total else 0.0,
        "ratio": (windows_total / minutes) / (glosses_total / minutes) if glosses_total else 0.0,
    }


def by_duration(streams: list[dict], value: int, threshold: float) -> None:
    """Recall split by how long the annotated sign lasts.

    The sweep tops out near 53% no matter how low the floor goes, and the obvious suspect is
    the floor's shape rather than its value: a window that must last `MIN_SIGN_MS` cannot reach
    IoU 0.5 against a sign shorter than that, and half the real signs are shorter than 480 ms.
    If recall collapses below the floor and holds above it, a minimum-duration rule is the
    wrong shape for continuous signing, not merely mistuned.
    """
    edges = [0, 200, 300, 400, 500, 650, 900, 10_000]
    hit = {i: 0 for i in range(len(edges) - 1)}
    total = {i: 0 for i in range(len(edges) - 1)}

    original = sim.MIN_SIGN_MS
    sim.MIN_SIGN_MS = value
    try:
        for stream in streams:
            spans = hb.window_spans(stream["right"], stream["left"], stream["fps"])
            paired = {g for g, _, _ in hb.match(stream["glosses"], spans, threshold)}
            for index, (start, end) in enumerate(stream["glosses"]):
                length = end - start
                bucket = next(
                    i for i in range(len(edges) - 1) if edges[i] <= length < edges[i + 1]
                )
                total[bucket] += 1
                hit[bucket] += index in paired
    finally:
        sim.MIN_SIGN_MS = original

    print()
    print(f"Recall a IoU {threshold} con MIN_SIGN_MS {value}, por duracion de la glosa")
    print()
    print(f"   {'duracion glosa':>16} {'glosas':>8} {'recuperadas':>12} {'recall':>8}")
    for i in range(len(edges) - 1):
        if not total[i]:
            continue
        top = "inf" if edges[i + 1] > 9999 else str(edges[i + 1])
        label = f"{edges[i]}-{top} ms"
        share = hit[i] / total[i] * 100
        print(f"   {label:>16} {total[i]:>8} {hit[i]:>12} {share:>7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=int, help="usar solo los primeros N videos del cache")
    parser.add_argument(
        "--values",
        help="lista de MIN_SIGN_MS separados por comas (por defecto un barrido de 1150 a 200)",
    )
    parser.add_argument(
        "--by-duration",
        type=int,
        help="en vez del barrido, desglosar el recall por duracion de glosa a este MIN_SIGN_MS",
    )
    arguments = parser.parse_args()

    annotations, gloss_rows, segment_rows = hb.load_annotations()
    if gloss_rows != hb.EXPECTED_GLOSSES or segment_rows != hb.EXPECTED_SEGMENTS:
        raise SystemExit("el parseo del xlsx no cuadra con el corpus publicado; no continuo")

    paths = [p for p in sorted(hb.CACHE.glob("*.npz")) if p.stem in annotations]
    if arguments.videos:
        paths = paths[: arguments.videos]
    if not paths:
        raise SystemExit("no hay .npz con anotacion en el cache")

    streams = load_streams(paths, annotations)
    glosses = sum(len(s["glosses"]) for s in streams)
    minutes = sum(s["duration_ms"] for s in streams) / 60000.0
    print("Barrido de MIN_SIGN_MS sobre signado continuo real (LSE-Health-UVigo)")
    print()
    print(f"   videos {len(streams)}   glosas {glosses}   minutos {minutes:.1f}")
    spans = [end - start for stream in streams for start, end in stream["glosses"]]
    print(f"   glosa anotada: mediana {np.median(spans):.0f} ms")
    print(f"   MIN_MS {sim.MIN_MS}   MIN_FRAMES {sim.MIN_FRAMES}   MAX_MS {sim.MAX_MS}")
    print()

    if arguments.by_duration:
        by_duration(streams, arguments.by_duration, 0.5)
        return

    thresholds = [0.5, 0.3]
    header = (
        f"{'MIN_SIGN_MS':>12} {'ventanas':>9} {'vent/glosa':>11} {'mediana ms':>11} "
        f"{'en suelo':>9} {'recall .5':>10} {'prec .5':>8} {'recall .3':>10} {'prec .3':>8}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))

    values = (
        [int(v) for v in arguments.values.split(",")] if arguments.values else CANDIDATES
    )
    original = sim.MIN_SIGN_MS
    try:
        for value in values:
            sim.MIN_SIGN_MS = value
            result = evaluate(streams, thresholds)
            mark = "  <- actual" if value == original else ""
            recall5 = result["matched"][0.5] / result["glosses"]
            prec5 = result["matched"][0.5] / result["windows"] if result["windows"] else 0.0
            recall3 = result["matched"][0.3] / result["glosses"]
            prec3 = result["matched"][0.3] / result["windows"] if result["windows"] else 0.0
            print(
                f"{value:>12} {result['windows']:>9} {result['ratio']:>11.2f} "
                f"{result['median_ms']:>11.0f} {result['at_floor'] * 100:>8.0f}% "
                f"{recall5 * 100:>9.1f}% {prec5 * 100:>7.1f}% "
                f"{recall3 * 100:>9.1f}% {prec3 * 100:>7.1f}%{mark}"
            )
    finally:
        sim.MIN_SIGN_MS = original

    print()
    print("Leer las dos columnas juntas: si el recall sube pero 'en suelo' sigue alto, el suelo")
    print("sigue eligiendo el final de la ventana y lo unico que ha cambiado es el tamano de la")
    print("rejilla. La deceleracion solo empieza a decidir cuando 'en suelo' baja.")


if __name__ == "__main__":
    main()
