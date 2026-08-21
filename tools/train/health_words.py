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
#: 0.70 and 0.80 are here because a model that has seen coarticulation should finally make
#: confidence mean something, and the shipped one never reached them.
GATES = (0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80)


def normalize(label: str) -> str:
    """Fold a gloss or concept to a comparable key.

    LSE-Health writes variants the model does not: a leading `*`, a parenthesised suffix like
    `AFECTAR(s)`, and accents the concept list spells inconsistently. Folding all three is what
    turns 101 gloss labels and 238 concepts into the 51 that genuinely name the same sign.
    """
    text = re.sub(r"\(.*?\)", "", label.strip().lstrip("*"))
    stripped = unicodedata.normalize("NFD", text)
    return "".join(c for c in stripped if unicodedata.category(c) != "Mn").upper().strip()


def load_model(
    state_path: Path | None = None, concepts_path: Path | None = None
) -> tuple[SignHead, list[str]]:
    """The shipped model by default, or a retrained one for comparison.

    The shipped weights are the flat float32 blob the browser reads; a retrained model is a
    torch state_dict with its own class list. Both become the same `SignHead`, so every number
    this file prints stays comparable across them.
    """
    if state_path is not None:
        loaded = json.loads((concepts_path or MANIFEST).read_text())
        concepts = loaded["concepts"] if isinstance(loaded, dict) else loaded
        model = SignHead(len(concepts))
        model.load_state_dict(torch.load(state_path, weights_only=True))
        model.eval()
        return model, concepts

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


def scoring_vocabulary(annotations: dict) -> set[str]:
    """The frozen set of gloss keys this bench scores, whatever model is being measured.

    It comes from the *shipped* 238 concepts on purpose. A retrained model may know every gloss
    in the corpus, and letting it widen the denominator from 6,872 instances to 12,871 would
    make its recall incomparable with the 0.8% baseline it has to beat. Same instances, same
    criterion, different model.
    """
    shipped = json.loads(MANIFEST.read_text())["concepts"]
    by_key = {normalize(c) for c in shipped}
    gloss_keys = {normalize(label) for spans in annotations.values() for *_, label in spans}
    keys = sorted(gloss_keys & by_key)
    if not SHARED.is_file():
        SHARED.write_text(json.dumps({k: i for i, k in enumerate(keys)}, indent=1,
                                     ensure_ascii=False, sort_keys=True))
        return set(keys)

    # The frozen file wins, and that is the point of freezing it. Shipping a model with more
    # concepts widens this intersection, and letting it widen would quietly change the
    # denominator every recall figure was measured against — a better-looking number for a
    # different exam. Say so loudly and keep scoring the original one.
    stored = sorted(json.loads(SHARED.read_text()))
    if stored != keys:
        print(
            f"   AVISO: el manifest actual daria {len(keys)} clases puntuables; se mantienen "
            f"las {len(stored)} congeladas en {SHARED.name} para que las cifras comparen"
        )
    return set(stored)


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
    parser.add_argument("--weights", type=Path, help="state_dict de un modelo reentrenado")
    parser.add_argument("--concepts", type=Path, help="lista de clases de ese modelo")
    parser.add_argument(
        "--only-test-signers",
        action="store_true",
        help="puntuar solo los videos de los signantes apartados en split.json",
    )
    arguments = parser.parse_args()

    annotations, gloss_rows, segment_rows = hb.load_annotations()
    if gloss_rows != hb.EXPECTED_GLOSSES or segment_rows != hb.EXPECTED_SEGMENTS:
        raise SystemExit("el parseo del xlsx no cuadra con el corpus publicado; no continuo")

    model, concepts = load_model(arguments.weights, arguments.concepts)
    # Predictions are compared as normalised *labels*, not class indices, so the same bench
    # measures the shipped 238-class model and a retrained 286-class one without translation.
    predicted_key = [normalize(c) for c in concepts]
    scoring = scoring_vocabulary(annotations)

    paths = [p for p in sorted(hb.CACHE.glob("*.npz")) if p.stem in annotations]
    if arguments.only_test_signers:
        split = json.loads((hb.CACHE.parent / "split.json").read_text())
        wanted = set(split["test_videos"])
        paths = [p for p in paths if p.stem in wanted]
    if arguments.videos:
        paths = paths[: arguments.videos]
    if not paths:
        raise SystemExit("no hay .npz con anotacion en el cache")

    print("Esku sobre signado continuo real: la palabra escrita, no la frontera")
    print()
    print(f"   vocabulario puntuable (congelado): {len(scoring)} clases")
    print(f"   clases del modelo evaluado       : {len(concepts)}")
    print(f"   videos en cache con anotacion   : {len(paths)}")

    truth: dict[str, list[tuple[float, float, int]]] = {}
    for path in paths:
        rows = []
        for start, end, label in annotations[path.stem]:
            key = normalize(label)
            if key in scoring:
                rows.append((start, end, key))
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
                            said = predicted_key[concept]
                            for index, (gstart, gend, gkey) in enumerate(truth[name]):
                                if index in claimed or gkey != said:
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
