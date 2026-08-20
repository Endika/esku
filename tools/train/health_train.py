"""Retrain the classifier on coarticulated signing, the regime it has never seen.

The shipped model writes the right word for 0.8% of signs on real continuous signing, and
88-96% of the windows it is handed never reach 0.30 confidence. It was trained on dictionary
recordings; LSE-Health's annotations are the same kind of labelled data for the missing regime,
so this trains on those and reports on **held-out signers** — the split is frozen in
`data/uvigo/split.json` precisely so retraining cannot eat the only honest bench in LSE.

Two variants, both on the shipped architecture and the shipped feature layout, because the
question here is what the data changes, not what an architecture change would:

- **A** fine-tunes the released 238-class weights on the coarticulated instances of the 51
  concepts it already knows. Cheapest possible intervention.
- **B** trains from scratch on SWL-LSE union LSE-Health, over the union of classes, which adds
  the ~50 everyday gloss labels the model has never had (`PERSONA`, `NO`, `OTRO`, `DENTRO`).

Early stopping needs a validation set and it is signer-disjoint too: one *training* signer is
held back, chosen as the one closest to 10% of training instances, and never the sole left-handed
signer left in training. A random validation split would leak the test signers' neighbours into
the stopping decision and inflate exactly the number this file exists to trust.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import health_bench as hb
from health_words import normalize
from train import BATCH, EPOCHS, PATIENCE, SignHead, accuracy
from vocabulary_features import vocabulary_signature

DATA = Path(__file__).parent / "data"
UVIGO = DATA / "uvigo"
MANIFEST = Path("../../public/models/lse-vocabulary.json")
WEIGHTS = Path("../../public/models/lse-vocabulary.bin")
SEEDS = (7, 13, 29, 41)
VAL_SHARE = 0.10


def signatures(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix, labels, source video and cut source from a `_raw.npz`."""
    bundle = np.load(path, allow_pickle=True)
    count = int(bundle["n"][0])
    x = np.stack(
        [
            vocabulary_signature(
                bundle[f"r{i}"], bundle[f"l{i}"], bundle[f"p{i}"], bundle[f"f{i}"]
            )
            for i in range(count)
        ]
    )
    labels = np.array([str(v) for v in bundle["y"]])

    def column(name: str) -> np.ndarray:
        if name in bundle:
            return np.array([str(v) for v in bundle[name]])
        return np.array([""] * count)

    return x, labels, column("video"), column("source")


def validation_signer(
    videos: np.ndarray, meta: dict[str, dict[str, str]]
) -> tuple[str, np.ndarray]:
    """Hold back one *training* signer for early stopping, never the last left-hander.

    A random tenth would put windows from a signer's own other videos on both sides of the
    stopping decision, which is the mild version of the leak this whole split exists to avoid.
    Chosen as the signer closest to `VAL_SHARE` of the windows.
    """
    per_signer: collections.Counter = collections.Counter()
    for video in videos:
        per_signer[meta[video]["signer"]] += 1
    profile: dict[str, tuple[str, str]] = {}
    for row in meta.values():
        profile.setdefault(row["signer"], (row["handedness"], row["deaf"]))
    left = [s for s in per_signer if profile.get(s, ("", ""))[0] == "Left"]
    eligible = [s for s in per_signer if not (len(left) == 1 and s in left)]
    total = sum(per_signer.values())
    chosen = min(eligible, key=lambda s: abs(per_signer[s] / total - VAL_SHARE))
    mask = np.array([meta[v]["signer"] == chosen for v in videos])
    return chosen, mask


def by_source(
    model: nn.Module, x: np.ndarray, y: np.ndarray, source: np.ndarray
) -> str:
    """Top-1 split by how the window was cut.

    A gold cut is the annotated boundary exactly; the live app never gets one. If the gain
    lives only in that column it is not a gain, so both are always printed together.
    """
    parts = []
    for name in ("gold", "segmenter", "reject"):
        mask = source == name
        if not mask.any():
            continue
        top1, _ = accuracy(
            model, torch.tensor(x[mask]), torch.tensor(y[mask], dtype=torch.long)
        )
        parts.append(f"{name} {top1:.3f} (n={int(mask.sum())})")
    return "  ".join(parts)


def fit(
    model: nn.Module,
    xt: torch.Tensor,
    yt: torch.Tensor,
    xv: torch.Tensor,
    yv: torch.Tensor,
    lr: float,
    quiet: bool,
) -> nn.Module:
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, stale = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            optimiser.zero_grad()
            criterion(model(xt[batch]), yt[batch]).backward()
            optimiser.step()
        schedule.step()

        top1, _ = accuracy(model, xv, yv)
        if top1 > best:
            best, stale = top1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if not quiet and epoch % 20 == 0:
            print(f"      epoch {epoch:3}  val top1 {top1:.3f}")
        if stale >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def shipped_model(concepts: list[str]) -> SignHead:
    model = SignHead(len(concepts))
    blob = np.fromfile(WEIGHTS, dtype=np.float32)
    state = model.state_dict()
    offset = 0
    for key in state:
        size = state[key].numel()
        state[key] = torch.tensor(blob[offset : offset + size]).reshape(state[key].shape)
        offset += size
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="A,B", help="A y/o B, separadas por comas")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--export",
        action="store_true",
        help="guardar pesos y lista de clases por variante y semilla, para health_words.py",
    )
    arguments = parser.parse_args()

    split = json.loads((UVIGO / "split.json").read_text())
    with (UVIGO / "videos.csv").open() as handle:
        meta = {row["video"]: row for row in csv.DictReader(handle)}

    concepts = json.loads(MANIFEST.read_text())["concepts"]
    shared = {normalize(c): i for i, c in enumerate(concepts)}

    print("Reentreno sobre signado coarticulado — signantes de test apartados")
    print()
    print(f"   signantes de test : {', '.join(split['test_signers'])}")
    print(f"   glosas con *      : {'incluidas' if split['starred_included'] else 'excluidas'}")

    xh, yh, vh, _ = signatures(UVIGO / "health_train_raw.npz")
    xs_test, ys_test, _, sh_test = signatures(UVIGO / "health_test_raw.npz")
    print(f"   ventanas          : {len(xh)} entrenamiento / {len(xs_test)} test")
    print("   AVISO: la extraccion no ha terminado, asi que esto es una senal temprana")
    print()

    val_signer, val_mask = validation_signer(vh, meta)
    print(f"   validacion        : signante {val_signer} apartado del entrenamiento "
          f"({int(val_mask.sum())} ventanas)")
    print()

    # The number that gives every other number meaning: the released weights, untouched, on the
    # very same held-out windows. Fine-tuning that lands below this line has cost accuracy.
    keep_base = np.array([i for i, label in enumerate(ys_test) if normalize(label) in shared])
    base_y = np.array([shared[normalize(ys_test[i])] for i in keep_base])
    base_top1, base_top3 = accuracy(
        shipped_model(concepts),
        torch.tensor(xs_test[keep_base]),
        torch.tensor(base_y, dtype=torch.long),
    )
    print(f"   LINEA BASE, pesos publicados sin tocar, mismas {len(keep_base)} ventanas de "
          f"test: top1 {base_top1:.3f}  top3 {base_top3:.3f}")
    print(f"      por fuente del recorte: "
          f"{by_source(shipped_model(concepts), xs_test[keep_base], base_y, sh_test[keep_base])}")
    print()

    variants = [v.strip().upper() for v in arguments.variants.split(",") if v.strip()]
    seeds = [int(s) for s in arguments.seeds.split(",") if s.strip()]
    results: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)

    for variant in variants:
        if variant == "A":
            keep = np.array([i for i, label in enumerate(yh) if normalize(label) in shared])
            keep_test = np.array(
                [i for i, label in enumerate(ys_test) if normalize(label) in shared]
            )
            classes = concepts
            index = shared
            xt_all, yt_all, mask = xh[keep], yh[keep], val_mask[keep]
            xe, ye, se = xs_test[keep_test], ys_test[keep_test], sh_test[keep_test]
            lr = 3e-4
            print(f"   A: {len(xt_all)} ventanas de las {len(shared)} clases compartidas, "
                  f"test {len(xe)}")
        else:
            xl, yl, _, _ = signatures(DATA / "train_raw.npz")
            xv2, yv2, _, _ = signatures(DATA / "val_raw.npz")
            xt_all = np.concatenate([xh, xl, xv2])
            yt_all = np.concatenate([yh, yl, yv2])
            # SWL-LSE windows never validate here: the stopping decision has to be made on
            # coarticulated material, which is the regime this retrain is about.
            mask = np.concatenate([val_mask, np.zeros(len(xl) + len(xv2), dtype=bool)])
            classes = sorted({normalize(c) for c in concepts} | {normalize(c) for c in yh})
            index = {c: i for i, c in enumerate(classes)}
            xe, ye, se = xs_test, ys_test, sh_test
            lr = 2e-3
            print(f"   B: {len(xt_all)} ventanas ({len(xh)} coarticuladas + {len(xl) + len(xv2)}"
                  f" aisladas), {len(classes)} clases, test {len(xe)}")

        def to_index(labels: np.ndarray) -> np.ndarray:
            return np.array([index.get(normalize(v), -1) for v in labels])

        yt_index = to_index(yt_all)
        ye_index = to_index(ye)
        good = yt_index >= 0
        xt_all, yt_index = xt_all[good], yt_index[good]
        good_test = ye_index >= 0
        xe, ye_index, se = xe[good_test], ye_index[good_test], se[good_test]

        mask = mask[good]
        ti = np.flatnonzero(~mask)
        vi = np.flatnonzero(mask)
        if len(vi) == 0:
            raise SystemExit("la validacion quedo vacia; revisa el signante elegido")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = shipped_model(concepts) if variant == "A" else SignHead(len(classes))
            model = fit(
                model,
                torch.tensor(xt_all[ti]),
                torch.tensor(yt_index[ti], dtype=torch.long),
                torch.tensor(xt_all[vi]),
                torch.tensor(yt_index[vi], dtype=torch.long),
                lr,
                arguments.quiet,
            )
            top1, top3 = accuracy(
                model,
                torch.tensor(xe),
                torch.tensor(ye_index, dtype=torch.long),
            )
            if arguments.export:
                target = UVIGO / f"model_{variant}_s{seed}.pt"
                torch.save(model.state_dict(), target)
                (UVIGO / f"concepts_{variant}.json").write_text(
                    json.dumps(list(classes), ensure_ascii=False, indent=1)
                )
                print(f"         exportado {target.name}")
            results[variant].append((top1, top3))
            print(f"      {variant} semilla {seed:>2}  test top1 {top1:.3f}  top3 {top3:.3f}")
            print(f"         por fuente: {by_source(model, xe, ye_index, se)}")

    print()
    print("Sobre signantes apartados, media y sd entre semillas:")
    for variant, rows in results.items():
        t1 = np.array([r[0] for r in rows])
        t3 = np.array([r[1] for r in rows])
        print(f"   {variant}  top1 {t1.mean():.3f} (sd {t1.std():.3f})  "
              f"top3 {t3.mean():.3f} (sd {t3.std():.3f})  n={len(rows)}")
    print()
    print("Recuerda: exactitud de clase sobre ventanas ya recortadas NO es recall de palabra.")
    print("La cifra que decide sigue siendo health_words.py sobre los signantes apartados.")


if __name__ == "__main__":
    main()
