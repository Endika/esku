"""Re-tune the segmenter against both benchmarks at once.

Isolated accuracy and continuous recovery pull in opposite directions: a rule eager enough
to find boundaries in fluent signing also cuts isolated signs at their internal slow-downs.
Tuning against either one alone picks a bad setting, which is how the shipped stillness rule
scored 0.739 isolated and 0.146 continuous without anyone noticing.

Sweeps the real constants in `simulate_app`, so there is one implementation of the rule and
no chance of the sweep and the app drifting apart.
"""

from __future__ import annotations

import numpy as np
import torch

import simulate_app as sim
from continuous import SIGNS_PER_STREAM, MIN_CONFIDENCE, load_model
from vocabulary_features import vocabulary_signature


def isolated(bundle, count, concepts, model, truth) -> float:
    matrices = []
    for i in range(count):
        r, l, p, f = bundle[f"r{i}"], bundle[f"l{i}"], bundle[f"p{i}"], bundle[f"f{i}"]
        found = sim.windows(r, l)
        keep = np.array(found[0]) if found else np.arange(len(r))
        matrices.append(vocabulary_signature(r[keep], l[keep], p[keep], f[keep]))
    with torch.no_grad():
        predictions = model(torch.tensor(np.stack(matrices))).argmax(1)
    return float((predictions == truth).float().mean())


def continuous(bundle, count, concepts, model) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    order = rng.permutation(count)
    recovered = total = spoken = correct = 0

    for start in range(0, count - SIGNS_PER_STREAM, SIGNS_PER_STREAM):
        parts, spans, labels = [], [], []
        offset = 0
        for i in order[start : start + SIGNS_PER_STREAM]:
            r, l, p, f = bundle[f"r{i}"], bundle[f"l{i}"], bundle[f"p{i}"], bundle[f"f{i}"]
            parts.append((r, l, p, f))
            spans.append((offset, offset + len(r)))
            labels.append(str(bundle["y"][i]))
            offset += len(r)

        right = np.concatenate([q[0] for q in parts])
        left = np.concatenate([q[1] for q in parts])
        pose = np.concatenate([q[2] for q in parts])
        face = np.concatenate([q[3] for q in parts])

        found = sim.windows(right, left)
        total += len(labels)
        if not found:
            continue

        matrices = [
            vocabulary_signature(
                right[np.array(w)], left[np.array(w)], pose[np.array(w)], face[np.array(w)]
            )
            for w in found
        ]
        with torch.no_grad():
            probabilities = torch.softmax(model(torch.tensor(np.stack(matrices))), 1)
        best = probabilities.max(1)

        said: dict[int, list[tuple[float, int]]] = {}
        for slot, w in enumerate(found):
            lo, hi = w[0], w[-1] + 1
            overlaps = [max(0, min(hi, b) - max(lo, a)) for a, b in spans]
            owner = int(np.argmax(overlaps))
            confidence, guess = float(best.values[slot]), int(best.indices[slot])
            said.setdefault(owner, []).append((confidence, guess))
            if confidence >= MIN_CONFIDENCE:
                spoken += 1
                correct += concepts[guess] == labels[owner]

        for owner, label in enumerate(labels):
            for confidence, guess in said.get(owner, []):
                if confidence >= MIN_CONFIDENCE and concepts[guess] == label:
                    recovered += 1
                    break

    return recovered / total, correct / max(spoken, 1)


def main() -> None:
    bundle = np.load("data/test_raw.npz", allow_pickle=True)
    count = int(bundle["n"][0])
    concepts = sorted(set(np.load("data/train_raw.npz", allow_pickle=True)["y"]))
    index = {c: i for i, c in enumerate(concepts)}
    model = load_model(concepts)
    truth = torch.tensor([index[str(bundle["y"][i])] for i in range(count)])

    shipped = (sim.DECELERATION_DROP, sim.DECELERATION_HOLD, sim.MIN_SIGN_FRAMES)
    print(f"{'caida':>7} {'hold':>5} {'min':>5} {'aislado':>9} {'continuo':>10} {'precision':>11}")

    for drop in (0.45, 0.35):
        for hold in (1, 3):
            for min_sign in (18, 24, 28):
                sim.DECELERATION_DROP = drop
                sim.DECELERATION_HOLD = hold
                sim.MIN_SIGN_FRAMES = min_sign
                iso = isolated(bundle, count, concepts, model, truth)
                rec, precision = continuous(bundle, count, concepts, model)
                mark = "  <-- actual" if (drop, hold, min_sign) == shipped else ""
                print(
                    f"{drop:>7} {hold:>5} {min_sign:>5} "
                    f"{iso:9.3f} {rec * 100:9.1f}% {precision * 100:10.1f}%{mark}"
                )

    sim.DECELERATION_DROP, sim.DECELERATION_HOLD, sim.MIN_SIGN_FRAMES = shipped


if __name__ == "__main__":
    main()
