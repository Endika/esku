"""Train the fingerspelling head with CTC, and export the blob the browser reads.

LSE-FS labels the spelled word and never says when each letter happened, so there is no
per-frame target to fit. CTC marginalises over every alignment that could produce the word,
which is exactly the supervision the corpus can give.

**The blank is the point, not a side effect.** A handshape table can only answer "which letter
is this most like", never "this is not a letter", and that is why the shipped engine writes into
transitions or says nothing at all. CTC's blank is a trained class for "no letter here", learned
from word-level labels alone — the negatives `health_dataset.py` could not build for the
vocabulary head come free here.

**Causal on purpose.** A bidirectional GRU is the natural choice for CTC offline and is
unusable: the app classifies frame by frame on a live camera, so the model may never see the
future. Measured with a unidirectional GRU throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lsefs_dataset import ALPHABET, DATA, HAND_FLOATS

CLASSES = len(ALPHABET) + 1
HIDDEN = 128
LAYERS = 2
#: How many batches a shuffling window spans. Wider trades padding back for diversity.
WINDOW_BATCHES = 8
OUT = Path(__file__).parent.parent.parent / "public" / "models"


class Speller(nn.Module):
    """Two causal GRU layers and a linear head, ~177k parameters, 0.71 MB as float32.

    Deliberately small. The training set is 18,887 letters, and the SPIKE that sized this saw
    validation CER bottom out around epoch 80 and then worsen — the ceiling is data, not
    capacity, so spending the download budget on a wider model would buy overfitting.
    """

    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.gru = nn.GRU(HAND_FLOATS, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0])


def load(split: str) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    bundle = np.load(DATA / f"lsefs_{split}.npz", allow_pickle=True)
    xs, ys = [], []
    at = bt = 0
    for n, m in zip(bundle["lengths"], bundle["tlengths"], strict=True):
        xs.append(torch.tensor(bundle["x"][at : at + n]))
        at += int(n)
        ys.append(torch.tensor(bundle["y"][bt : bt + m]))
        bt += int(m)
    return xs, ys, list(bundle["labels"])


def augment(x: torch.Tensor, generator: np.random.Generator, flags: set[str]) -> torch.Tensor:
    """Widen 18,887 letters, without inventing poses the app cannot produce.

    Mirroring is absent and that is deliberate: `normalize_hand` already folds left hands into
    right-hand space, so negating x again would manufacture shapes no live frame ever carries.
    Scale jitter is absent for the same reason — every coordinate is already divided by palm
    width, so there is no scale left to vary.
    """
    points = x.reshape(len(x), -1, 3).clone()

    if "rotate" in flags:
        # Camera tilt and wrist roll, in the image plane. Small: a large rotation turns one
        # letter into another rather than into a variant of itself.
        angle = float(generator.uniform(-0.20, 0.20))
        cos, sin = math.cos(angle), math.sin(angle)
        px, py = points[..., 0].clone(), points[..., 1].clone()
        points[..., 0] = px * cos - py * sin
        points[..., 1] = px * sin + py * cos

    if "noise" in flags:
        # MediaPipe's own estimation jitter, in palm widths.
        points += torch.tensor(
            generator.normal(0.0, 0.01, points.shape).astype(np.float32)
        )

    out = points.reshape(len(x), -1)

    if "drop" in flags:
        # Detection dropouts. The corpus already carries 21% empty frames; this teaches the
        # model that a hole is a hole and not a new handshape.
        mask = torch.tensor(generator.random(len(out)) < 0.05)
        out[mask] = 0.0

    if "speed" in flags:
        # Signers spell at different speeds, and train does not span what test contains.
        rate = float(generator.uniform(0.8, 1.25))
        target = max(1, int(round(len(out) * rate)))
        index = torch.linspace(0, len(out) - 1, target).round().long()
        out = out[index]

    return out


def batches(xs, ys, size, generator=None, flags=frozenset()):
    """Batches of similar length, in random order — but not *only* by length.

    Padding to the longest member of each batch is not free: sequences here run from 8 frames
    to 422 with a median of 150, so purely random batches push 2.4x the real frames through the
    GRU and 57% of the arithmetic is spent on zeros.

    Sorting strictly by length fixes that and costs accuracy, which is worth stating because it
    is not obvious: length correlates with how many letters the word has, so length-sorted
    batches are also *content*-sorted, and the gradients get less diverse. Measured on seed 7,
    strict sorting scored CER 0.349 against 0.313 for random batching.

    So the sort is coarse. Sequences are sorted, cut into windows several batches wide, and
    shuffled **within** each window before the batches are formed. Neighbouring lengths still
    travel together — most of the padding is still gone — while each batch mixes words.
    """
    lengths = np.array([len(x) for x in xs])
    order = np.argsort(lengths, kind="stable")
    if generator is not None:
        window = size * WINDOW_BATCHES
        for start in range(0, len(order), window):
            block = order[start : start + window]
            generator.shuffle(block)
            order[start : start + window] = block
    groups = [order[i : i + size] for i in range(0, len(order), size)]
    if generator is not None:
        generator.shuffle(groups)
    for chunk in groups:
        rows = [augment(xs[j], generator, flags) if flags else xs[j] for j in chunk]
        lengths = torch.tensor([len(r) for r in rows])
        padded = torch.zeros(len(rows), int(lengths.max()), HAND_FLOATS)
        for k, row in enumerate(rows):
            padded[k, : len(row)] = row
        targets = torch.cat([ys[j] for j in chunk])
        tlengths = torch.tensor([len(ys[j]) for j in chunk])
        yield padded, lengths, targets, tlengths


def collapse(ids: list[int]) -> str:
    """CTC greedy decode: drop repeats, then drop blanks. The app's stabiliser does the same."""
    out: list[str] = []
    previous = 0
    for i in ids:
        if i != previous and i != 0:
            out.append(ALPHABET[i - 1])
        previous = i
    return "".join(out)


def edit(want: str, got: str) -> int:
    previous = list(range(len(got) + 1))
    for i, a in enumerate(want, 1):
        current = [i]
        for j, b in enumerate(got, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def evaluate(model: Speller, xs, ys) -> tuple[float, int]:
    model.eval()
    distance = characters = exact = 0
    with torch.no_grad():
        for padded, lengths, targets, tlengths in batches(xs, ys, 64):
            logits = model(padded)
            at = 0
            for k, n in enumerate(lengths.tolist()):
                want = "".join(
                    ALPHABET[i - 1] for i in targets[at : at + tlengths[k]].tolist()
                )
                at += int(tlengths[k])
                got = collapse(logits[k, :n].argmax(-1).tolist())
                distance += edit(want, got)
                characters += len(want)
                exact += want == got
    return distance / characters, exact


def train_one(seed: int, flags: set[str], epochs: int, patience: int, quiet: bool):
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    xtr, ytr, _ = load("train")
    xva, yva, _ = load("validation")

    model = Speller()
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_cer, best_state, best_epoch, stale = math.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for padded, lengths, targets, tlengths in batches(
            xtr, ytr, 32, generator, flags
        ):
            logp = model(padded).log_softmax(-1).transpose(0, 1)
            loss = loss_fn(logp, targets, lengths, tlengths)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        schedule.step()

        if epoch % 5:
            continue
        cer, exact = evaluate(model, xva, yva)
        if cer < best_cer - 1e-4:
            best_cer, best_epoch, stale = cer, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if not quiet:
            print(f"      epoca {epoch:>3}  CER {cer:.3f}  exactas {exact}/{len(xva)}")
        # Early stopping, because the SPIKE watched CER bottom out at 80 and worsen by 100.
        if stale >= patience:
            break

    return best_cer, best_epoch, best_state


def export(state: dict, path: Path) -> None:
    """Flat float32 in `gru.ts`'s tensor order, plus the manifest that names it."""
    order = [k for k in state if k.startswith("gru.")] + ["head.weight", "head.bias"]
    blob = np.concatenate([state[k].numpy().astype(np.float32).reshape(-1) for k in order])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lse-alphabet.bin").write_bytes(blob.tobytes())
    (OUT / "lse-alphabet.json").write_text(
        json.dumps(
            {
                "hidden": HIDDEN,
                "layers": LAYERS,
                "inputs": HAND_FLOATS,
                "classes": CLASSES,
                "blank": 0,
                "letters": list(ALPHABET),
                # `order` and `shapes`, the same shape of manifest `VocabularySignClassifier`
                # already reads, so the loader is the one that exists rather than a second one.
                "order": order,
                "shapes": {k: list(state[k].shape) for k in order},
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"   pesos {blob.nbytes / 1e6:.2f} MB -> public/models/lse-alphabet.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="7", help="lista separada por comas")
    parser.add_argument("--augment", default="", help="rotate,noise,drop,speed o 'all'")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=4, help="evaluaciones sin mejorar")
    parser.add_argument("--export", action="store_true", help="exportar la mejor semilla")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    every = {"rotate", "noise", "drop", "speed"}
    flags = every if arguments.augment == "all" else {
        f.strip() for f in arguments.augment.split(",") if f.strip()
    }
    if flags - every:
        raise SystemExit(f"aumentaciones desconocidas: {sorted(flags - every)}")

    seeds = [int(s) for s in arguments.seeds.split(",")]
    print(f"CTC sobre LSE-FS  |  aumentacion: {','.join(sorted(flags)) or 'ninguna'}")
    print()

    results, best = [], (math.inf, None)
    for seed in seeds:
        print(f"   semilla {seed}")
        cer, epoch, state = train_one(
            seed, flags, arguments.epochs, arguments.patience, arguments.quiet
        )
        results.append(cer)
        print(f"      mejor CER {cer:.3f} en la epoca {epoch}")
        if cer < best[0]:
            best = (cer, state)

    print()
    mean = statistics.mean(results)
    spread = statistics.stdev(results) if len(results) > 1 else 0.0
    print(f"   CER validation  media {mean:.3f}  sd {spread:.3f}  n={len(results)}")
    print("   Una semilla no es una medida: nada se decide por debajo de ~1 sd.")

    if arguments.export and best[1] is not None:
        export(best[1], OUT)


if __name__ == "__main__":
    main()
