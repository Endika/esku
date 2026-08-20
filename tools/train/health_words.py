"""Does Esku write the *right word* on real continuous signing? The only number that decides.

Everything measured before this is a proxy. `health_bench.py` asks whether a window lands where
a sign is; `simulate_app.py` asks whether an isolated dictionary recording classifies. Neither
asks the product question, and the two disagree about the segmenter floor: lowering it triples
boundary recall on real signing and costs accuracy on dictionary signs.

That disagreement is what this file exists to settle, and it sweeps the floor and the confidence
gate *together* on purpose. They are coupled: a lower floor makes shorter windows, a shorter
window holds less of the sign, less sign means less confidence, and less confidence dies at the
gate. Measured apart, each knob looks bad because of the other. Measured together, the cost is
attributable.

51 of the model's 238 concepts appear among LSE-Health's annotated glosses, covering 6,872 of
its 15,098 instances, so this is scored against real hand-annotated occurrences of signs the
model was actually trained to know.

Windows are classified once per floor; the grace period and the gate are then swept in memory,
because they are post-processing over the same scores.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

import health_bench as hb
import simulate_app as sim
from train import SignHead
from vocabulary_features import vocabulary_signature

MANIFEST = Path("../../public/models/lse-vocabulary.json")
WEIGHTS = Path("../../public/models/lse-vocabulary.bin")
SHARED = hb.CACHE.parent / "shared_classes.json"

FLOORS = (1150, 750, 500)
#: A window closes roughly one window-length after the previous one, so a grace shorter than
#: the window can never group anything — 600 ms grouped exactly nothing at a 1200 ms window,
#: which is how this range was chosen rather than guessed.
GRACES = (0, 600, 1500, 2500)
GATES = (0.30, 0.40, 0.45, 0.50, 0.60)


def normalize(label: str) -> str:
    """Fold a gloss or concept to a comparable key.

    LSE-Health writes variants the model does not: a leading `*`, a parenthesised suffix like
    `AFECTAR(s)`, and accents the concept list spells inconsistently. Folding all three is what
    turns 101 gloss labels and 238 concepts into the 51 that genuinely name the same sign.
    """
    text = re.sub(r"\(.*?\)", "", label.strip().lstrip("*"))
    stripped = unicodedata.normalize("NFD", text)
    return "".join(c for c in stripped if unicodedata.category(c) != "Mn").upper().strip()


def load_model() -> tuple[SignHead, list[str]]:
    concepts = json.loads(MANIFEST.read_text())["concepts"]
    model = SignHead(len(concepts))
    weights = np.fromfile(WEIGHTS, dtype=np.float32)
    state = model.state_dict()
    offset = 0
    for key in state:
        size = state[key].numel()
        state[key] = torch.tensor(weights[offset : offset + size]).reshape(state[key].shape)
        offset += size
    model.load_state_dict(state)
    model.eval()
    return model, concepts


def shared_vocabulary(concepts: list[str], annotations: dict) -> dict[str, int]:
    """Frozen gloss-key -> concept index map, written once so it cannot drift between runs."""
    by_key = {normalize(c): i for i, c in enumerate(concepts)}
    gloss_keys = {normalize(label) for spans in annotations.values() for *_, label in spans}
    shared = {key: by_key[key] for key in sorted(gloss_keys & set(by_key))}
    if SHARED.is_file():
        stored = json.loads(SHARED.read_text())
        if stored != shared:
            raise SystemExit(f"{SHARED} no coincide con el mapeo recalculado; revisalo a mano")
    else:
        SHARED.write_text(json.dumps(shared, indent=1, ensure_ascii=False, sort_keys=True))
    return shared


def score_windows(model, path: Path, floor: int) -> tuple[list[tuple], float, float]:
    """(close_ms, concept, confidence, start_ms, end_ms) per window, plus duration and rate."""
    with np.load(path) as bundle:
        if "pose" not in bundle or "face" not in bundle:
            return [], 0.0, 0.0
        right, left = bundle["right"], bundle["left"]
        pose, face = bundle["pose"], bundle["face"]
        declared = float(bundle["fps"][0])
        frames = int(bundle["frames"][0]) if "frames" in bundle else len(right)
    if len(right) == 0:
        return [], 0.0, 0.0
    fps = declared if 0.0 < declared <= hb.MAX_PLAUSIBLE_FPS else hb.FALLBACK_FPS

    sim.MIN_SIGN_MS = floor
    spans = sim.windows(right, left, fps=fps)
    if not spans:
        return [], frames * 1000.0 / fps, fps

    matrix = [
        vocabulary_signature(right[np.array(s)], left[np.array(s)], pose[np.array(s)],
                             face[np.array(s)])
        for s in spans
    ]
    with torch.no_grad():
        scores = torch.softmax(model(torch.tensor(np.stack(matrix))), dim=1)

    frame_ms = 1000.0 / fps
    scored = []
    for i, span in enumerate(spans):
        confidence = float(scores[i].max())
        scored.append(
            (span[-1] * frame_ms, int(scores[i].argmax()), confidence,
             span[0] * frame_ms, span[-1] * frame_ms)
        )
    return scored, frames * frame_ms, fps


def arbitrate(scored: list[tuple], grace_ms: float, gate: float) -> list[tuple]:
    """Hold the first candidate, let windows closing within `grace_ms` compete, emit the best.

    The deadline comes from the first candidate and is never extended, so the delay a viewer
    sees is bounded by `grace_ms` however many windows a sign is shredded into.
    """
    emitted = []
    i = 0
    while i < len(scored):
        deadline = scored[i][0] + grace_ms
        group = [scored[i]]
        j = i + 1
        while j < len(scored) and scored[j][0] <= deadline:
            group.append(scored[j])
            j += 1
        best = max(group, key=lambda row: row[2])
        if best[2] >= gate:
            emitted.append(best)
        i = j
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=int, help="limitar a los primeros N del cache")
    arguments = parser.parse_args()

    annotations, gloss_rows, segment_rows = hb.load_annotations()
    if gloss_rows != hb.EXPECTED_GLOSSES or segment_rows != hb.EXPECTED_SEGMENTS:
        raise SystemExit("el parseo del xlsx no cuadra con el corpus publicado; no continuo")

    model, concepts = load_model()
    shared = shared_vocabulary(concepts, annotations)

    paths = [p for p in sorted(hb.CACHE.glob("*.npz")) if p.stem in annotations]
    if arguments.videos:
        paths = paths[: arguments.videos]
    if not paths:
        raise SystemExit("no hay .npz con anotacion en el cache")

    print("Esku sobre signado continuo real: la palabra escrita, no la frontera")
    print()
    print(f"   clases compartidas modelo/corpus: {len(shared)} de {len(concepts)}")
    print(f"   videos en cache con anotacion   : {len(paths)}")

    truth: dict[str, list[tuple[float, float, int]]] = {}
    for path in paths:
        rows = []
        for start, end, label in annotations[path.stem]:
            key = normalize(label)
            if key in shared:
                rows.append((start, end, shared[key]))
        truth[path.stem] = rows
    instances = sum(len(v) for v in truth.values())
    print(f"   instancias puntuables           : {instances}")
    print()

    header = (
        f"{'suelo':>6} {'espera':>7} {'puerta':>7} {'recall palabra':>15} "
        f"{'pal/min':>8} {'glosa/min':>10} {'muertas en puerta':>18}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))

    original = sim.MIN_SIGN_MS
    try:
        for floor in FLOORS:
            cached = {}
            minutes = 0.0
            for path in paths:
                scored, duration_ms, _ = score_windows(model, path, floor)
                cached[path.stem] = scored
                minutes += duration_ms / 60000.0
            total_windows = sum(len(v) for v in cached.values())

            for grace in GRACES:
                for gate in GATES:
                    recovered = words = 0
                    for name, scored in cached.items():
                        emitted = arbitrate(scored, grace, gate)
                        words += len(emitted)
                        claimed = set()
                        for _, concept, _, start, end in emitted:
                            for index, (gstart, gend, gconcept) in enumerate(truth[name]):
                                if index in claimed or gconcept != concept:
                                    continue
                                if start < gend and end > gstart:
                                    claimed.add(index)
                                    recovered += 1
                                    break
                    dead = total_windows - words
                    dead_text = f"{dead}/{total_windows}"
                    print(
                        f"{floor:>6} {grace:>7} {gate:>7.2f} "
                        f"{recovered / instances * 100:>14.1f}% {words / minutes:>8.1f} "
                        f"{instances / minutes:>10.1f} {dead_text:>18}"
                    )
            print()
    finally:
        sim.MIN_SIGN_MS = original

    print("recall palabra = instancias de las 51 clases con una palabra emitida del concepto")
    print("correcto que solapa temporalmente la glosa anotada, contada una vez por instancia.")
    print("La glosa/min es el techo teorico solo de estas 51 clases, no de todo lo que se signa.")


if __name__ == "__main__":
    main()
