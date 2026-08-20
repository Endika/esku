"""Train the LSE vocabulary head and export it to ONNX.

Reports accuracy on SWL-LSE's own held-out test split, never on data the model trained on.
The number this prints is the number the README is allowed to claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vocabulary_features import FRAMES as SIGNATURE_FRAMES
from vocabulary_features import SIGNATURE_LENGTH, vocabulary_signature

DATA = Path(__file__).parent / "data"
ARTIFACTS = Path(__file__).parent / "artifacts"
FIXTURES = Path(__file__).parent.parent.parent / "src" / "test" / "fixtures"
OUT = Path(__file__).parent.parent.parent / "public" / "models"
FRAME_FLOATS = SIGNATURE_LENGTH // SIGNATURE_FRAMES

EPOCHS = 120
BATCH = 64
PATIENCE = 15
SEED = 7


class SignHead(nn.Module):
    """A GRU over the eight resampled frames, then a classifier.

    Recurrent rather than a flat MLP because a sign is a trajectory: the same handshapes in
    the reverse order can be a different word, and an MLP over the concatenated frames has
    to learn that from scratch per class.

    `hidden` is small on purpose. At 256 the model carried ~2M parameters against 6,336
    training examples and shipped as 7.6 MB of weights — over-parameterised for the data and
    heavy for a phone to download. 128 keeps the accuracy and costs a quarter of that.
    """

    def __init__(self, classes: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(FRAME_FLOATS)
        self.gru = nn.GRU(FRAME_FLOATS, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.norm(x.view(-1, SIGNATURE_FRAMES, FRAME_FLOATS))
        output, _ = self.gru(frames)
        # Mean over time rather than the last state: signs end on a hold, so the final frame
        # is often the least informative one.
        return self.head(output.mean(dim=1))


# The order the browser reads the weight blob in. Changing it silently breaks inference, so
# it is declared once here and mirrored in `GruSignClassifier`'s loader.
WEIGHT_ORDER = [
    "norm.weight",
    "norm.bias",
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "gru.weight_ih_l0_reverse",
    "gru.weight_hh_l0_reverse",
    "gru.bias_ih_l0_reverse",
    "gru.bias_hh_l0_reverse",
    "gru.weight_ih_l1",
    "gru.weight_hh_l1",
    "gru.bias_ih_l1",
    "gru.bias_hh_l1",
    "gru.weight_ih_l1_reverse",
    "gru.weight_hh_l1_reverse",
    "gru.bias_ih_l1_reverse",
    "gru.bias_hh_l1_reverse",
    "head.0.weight",
    "head.0.bias",
    "head.3.weight",
    "head.3.bias",
]


DEFAULT_SOURCE = "SWL-LSE (CC-BY-4.0), doi:10.5281/zenodo.13691887"


def export_weights(
    model: nn.Module,
    concepts: list[str],
    top1: float,
    top3: float,
    source: str = DEFAULT_SOURCE,
) -> None:
    """Dump the weights as one flat float32 blob plus a manifest.

    Not ONNX, deliberately. onnxruntime-web needs 13 MB of its own WASM before it can run a
    2.4 MB model, and on GitHub Pages it cannot even use threads because Pages sends no
    COOP/COEP headers. This network is a fixed stack of matrix multiplies, so the browser
    side is a few hundred lines of arithmetic and no runtime at all — and `check_parity.py`
    proves the two agree.
    """
    state = model.state_dict()
    unlisted = [key for key in state if key not in WEIGHT_ORDER]
    if unlisted:
        raise SystemExit(f"WEIGHT_ORDER is out of date, missing: {unlisted}")

    blob = np.concatenate(
        [state[key].detach().numpy().astype(np.float32).reshape(-1) for key in WEIGHT_ORDER]
    )
    (OUT / "lse-vocabulary.bin").write_bytes(blob.tobytes())
    (OUT / "lse-vocabulary.json").write_text(
        json.dumps(
            {
                "concepts": concepts,
                "signatureLength": SIGNATURE_LENGTH,
                "frames": SIGNATURE_FRAMES,
                "hidden": model.gru.hidden_size,
                "layers": model.gru.num_layers,
                "order": WEIGHT_ORDER,
                "shapes": {key: list(state[key].shape) for key in WEIGHT_ORDER},
                "testTop1": round(top1, 4),
                "testTop3": round(top3, 4),
                "source": source,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"weights {blob.nbytes / 1024 / 1024:.1f} MB -> public/models/lse-vocabulary.bin")


def export_reference_vector(model: nn.Module) -> None:
    """A fixed input and this model's logits for it, for the TypeScript parity test.

    Feature parity was already checked once and the layouts matched; this checks the *model*
    the same way, so a reimplementation that is subtly wrong fails a test instead of quietly
    predicting nonsense.
    """
    torch.manual_seed(11)
    sample = torch.rand(1, SIGNATURE_LENGTH) * 2 - 1
    with torch.no_grad():
        logits = model(sample)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "model-parity.json").write_text(
        json.dumps(
            {
                "input": [round(v, 6) for v in sample[0].tolist()],
                "logits": [round(v, 5) for v in logits[0].tolist()],
            }
        ),
        encoding="utf-8",
    )
    print("parity vector -> src/test/fixtures/model-parity.json")


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Build signatures from the cached raw landmarks, so features can change freely."""
    bundle = np.load(DATA / f"{split}_raw.npz", allow_pickle=True)
    count = int(bundle["n"][0])
    x = np.stack(
        [
            vocabulary_signature(
                bundle[f"r{i}"], bundle[f"l{i}"], bundle[f"p{i}"], bundle[f"f{i}"]
            )
            for i in range(count)
        ]
    )
    return x, bundle["y"]


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Top-1 and top-3. Top-3 matters because the UI can offer alternatives."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        top3 = logits.topk(3, dim=1).indices
    top1 = (top3[:, 0] == y).float().mean().item()
    hit3 = (top3 == y.unsqueeze(1)).any(dim=1).float().mean().item()
    return top1, hit3


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train, y_train_raw = load("train")
    x_val, y_val_raw = load("val")
    x_test, y_test_raw = load("test")

    concepts = sorted(set(y_train_raw) | set(y_val_raw) | set(y_test_raw))
    index = {concept: i for i, concept in enumerate(concepts)}
    print(f"{len(concepts)} concepts, {len(x_train)} train / {len(x_val)} val / {len(x_test)} test")

    to_y = lambda raw: torch.tensor([index[c] for c in raw], dtype=torch.long)
    xt, yt = torch.tensor(x_train), to_y(y_train_raw)
    xv, yv = torch.tensor(x_val), to_y(y_val_raw)
    xs, ys = torch.tensor(x_test), to_y(y_test_raw)

    model = SignHead(len(concepts))
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    # Label smoothing: several concepts are near-synonyms in the dataset, so demanding
    # absolute confidence on the exact one mostly teaches overfitting.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 0.0
    best_state: dict | None = None
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            optimiser.zero_grad()
            loss = criterion(model(xt[batch]), yt[batch])
            loss.backward()
            optimiser.step()
        schedule.step()

        val_top1, val_top3 = accuracy(model, xv, yv)
        if val_top1 > best_val:
            best_val, stale = val_top1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % 10 == 0 or stale == 0:
            print(f"  epoch {epoch:3}  val top1 {val_top1:.3f}  top3 {val_top3:.3f}")
        if stale >= PATIENCE:
            print(f"  stopping early at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    test_top1, test_top3 = accuracy(model, xs, ys)
    print(f"\nHELD-OUT TEST  top1 {test_top1:.3f}  top3 {test_top3:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.eval()
    export_weights(model, concepts, test_top1, test_top3)
    export_reference_vector(model)


if __name__ == "__main__":
    main()
