"""Score the segmenter on continuous signing, which nothing else here measures.

Every number this project trusts — 0.741 top-1, 0.739 through the segmenter — comes from
SWL-LSE, where each recording holds one sign with stillness either side. The app is pointed
at people signing fluently, and there a window is not a truncated sign: it is the tail of
one glued to the head of the next. Measured in the browser, the model scores ~16% on that
input while measuring 74% here, and no ablation over isolated signs reproduces the gap.

So build the missing benchmark: concatenate test recordings back to back with no pause, keep
the true boundaries, and ask how many signs the segmenter actually recovers.

This is optimistic — splicing recordings does not reproduce real co-articulation, where the
hand travels from one sign's end to the next one's start. Treat the numbers as an upper
bound on continuous performance, not an estimate of it.
"""

from __future__ import annotations

import numpy as np
import torch

import simulate_app as sim
from train import SignHead
from vocabulary_features import vocabulary_signature

SIGNS_PER_STREAM = 4
MIN_CONFIDENCE = 0.45


windows_over = sim.windows


def load_model(concepts: list[str]) -> SignHead:
    model = SignHead(len(concepts))
    weights = np.fromfile("../../public/models/lse-vocabulary.bin", dtype=np.float32)
    state = model.state_dict()
    offset = 0
    for key in state:
        size = state[key].numel()
        state[key] = torch.tensor(weights[offset : offset + size]).reshape(state[key].shape)
        offset += size
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    bundle = np.load("data/test_raw.npz", allow_pickle=True)
    count = int(bundle["n"][0])
    concepts = sorted(set(np.load("data/train_raw.npz", allow_pickle=True)["y"]))
    index = {c: i for i, c in enumerate(concepts)}
    model = load_model(concepts)

    rng = np.random.default_rng(0)
    order = rng.permutation(count)

    recovered = truth_total = spoken = correct_spoken = 0
    confidences: list[float] = []
    coverage: list[float] = []

    for start in range(0, count - SIGNS_PER_STREAM, SIGNS_PER_STREAM):
        picks = order[start : start + SIGNS_PER_STREAM]

        parts, spans, labels = [], [], []
        offset = 0
        for i in picks:
            r, l, p, f = bundle[f"r{i}"], bundle[f"l{i}"], bundle[f"p{i}"], bundle[f"f{i}"]
            parts.append((r, l, p, f))
            spans.append((offset, offset + len(r)))
            labels.append(str(bundle["y"][i]))
            offset += len(r)

        right = np.concatenate([p[0] for p in parts])
        left = np.concatenate([p[1] for p in parts])
        pose = np.concatenate([p[2] for p in parts])
        face = np.concatenate([p[3] for p in parts])

        found = windows_over(right, left)
        if not found:
            truth_total += len(labels)
            continue

        matrices = [
            vocabulary_signature(right[np.array(w)], left[np.array(w)], pose[np.array(w)], face[np.array(w)])
            for w in found
        ]
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.tensor(np.stack(matrices))), 1)
        best = probabilities.max(1)

        # A window belongs to whichever true sign it overlaps most.
        said: dict[int, list[tuple[float, int]]] = {}
        for slot, w in enumerate(found):
            lo, hi = w[0], w[-1] + 1
            overlaps = [max(0, min(hi, b) - max(lo, a)) for a, b in spans]
            owner = int(np.argmax(overlaps))
            coverage.append(overlaps[owner] / (spans[owner][1] - spans[owner][0]))
            confidence = float(best.values[slot])
            confidences.append(confidence)
            said.setdefault(owner, []).append((confidence, int(best.indices[slot])))
            if confidence >= MIN_CONFIDENCE:
                spoken += 1
                if concepts[int(best.indices[slot])] == labels[owner]:
                    correct_spoken += 1

        for owner, label in enumerate(labels):
            truth_total += 1
            for confidence, guess in said.get(owner, []):
                if confidence >= MIN_CONFIDENCE and concepts[guess] == label:
                    recovered += 1
                    break

    print(f"signos encadenados por secuencia : {SIGNS_PER_STREAM}")
    print(f"signos verdaderos totales        : {truth_total}")
    print(f"ventanas emitidas                : {len(confidences)}")
    print(f"confianza media de una ventana   : {np.mean(confidences) * 100:.1f}%")
    print(f"solape ventana/signo (media)     : {np.mean(coverage) * 100:.1f}%")
    print()
    print(f"signos recuperados               : {recovered / truth_total * 100:.1f}%")
    if spoken:
        print(f"precision de lo que llega a decir: {correct_spoken / spoken * 100:.1f}%")
    print()
    print("mismo segmentador sobre signos aislados (simulate_app.py): 0.696 top-1")
    print("regla anterior, esperar quietud                        : 14.6% recuperados")


if __name__ == "__main__":
    main()
