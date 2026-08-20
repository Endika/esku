"""Check the shipped model artefacts still agree with each other and with the code.

This exists because of a real failure. When the vocabulary model went from 238 concepts to 287,
`simulate_app.py` kept deriving its class list from SWL-LSE's labels, so it built a 238-class
network and loaded 287 classes of weights into it. Nothing threw: it scored 0.000 and looked like
a broken model rather than a broken assumption. CI stayed green throughout, because nothing under
`tools/train` runs there.

So this is the cheap half of that gap: no torch, no corpus, no video — just the invariants that
tie the three shipped artefacts together, which is where that class of bug shows up.

    python check_manifest.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[2] / "public" / "models"
FIXTURE = Path(__file__).resolve().parents[2] / "src" / "test" / "fixtures" / "model-parity.json"
COPY = Path(__file__).resolve().parents[2] / "src" / "presentation" / "App.ts"
MANIFEST = MODELS / "lse-vocabulary.json"
WEIGHTS = MODELS / "lse-vocabulary.bin"
FLOAT_BYTES = 4


def failures() -> list[str]:
    problems: list[str] = []
    manifest = json.loads(MANIFEST.read_text())

    frames = manifest["frames"]
    length = manifest["signatureLength"]
    if length % frames:
        problems.append(f"signatureLength {length} no es multiplo de frames {frames}")

    # Declared shapes must account for the blob exactly. A mismatch means the weights and the
    # manifest were written by different runs, which is the state that scores 0.000 in silence.
    declared = sum(math.prod(shape) for shape in manifest["shapes"].values())
    actual = WEIGHTS.stat().st_size
    if declared * FLOAT_BYTES != actual:
        problems.append(
            f"el .bin mide {actual} bytes y las formas declaradas suman "
            f"{declared * FLOAT_BYTES}; manifest y pesos no vienen de la misma tirada"
        )

    if set(manifest["order"]) != set(manifest["shapes"]):
        problems.append("`order` y `shapes` no listan los mismos tensores")

    # The classifier's last layer decides how many concepts the network can name. If that
    # disagrees with the concept list, every prediction is offset or out of range.
    final = [key for key in manifest["order"] if key.endswith("weight")][-1]
    outputs = manifest["shapes"][final][0]
    if outputs != len(manifest["concepts"]):
        problems.append(
            f"el modelo produce {outputs} salidas y el manifest lista "
            f"{len(manifest['concepts'])} conceptos"
        )

    if len(set(manifest["concepts"])) != len(manifest["concepts"]):
        problems.append("hay conceptos repetidos en el manifest")

    # The browser feeds the parity fixture straight into the network, so its input has to be
    # exactly one signature long or the TypeScript port is being checked against nothing.
    if FIXTURE.is_file():
        fixture = json.loads(FIXTURE.read_text())
        if len(fixture["input"]) != length:
            problems.append(
                f"el fixture de paridad tiene {len(fixture['input'])} entradas y la firma "
                f"mide {length}"
            )
        if len(fixture.get("logits", [])) != len(manifest["concepts"]):
            problems.append(
                f"el fixture de paridad tiene {len(fixture.get('logits', []))} logits y el "
                f"manifest lista {len(manifest['concepts'])} conceptos"
            )
    else:
        problems.append(f"falta {FIXTURE}")

    # The About panel states the vocabulary size in prose, and it is rendered before the manifest
    # loads, so it cannot read the real number. Shipping 287 concepts left it saying 238 — wrong
    # on the screen people actually read. Cheaper to fail here than to notice months later.
    copy = COPY.read_text() if COPY.is_file() else ""
    claimed = {int(n) for n in re.findall(r"\b(\d{2,4}) signos\b", copy)}
    expected = len(manifest["concepts"])
    wrong = sorted(n for n in claimed if n != expected)
    if wrong:
        problems.append(
            f"{COPY.name} dice {wrong} signos y el modelo tiene {expected}"
        )
    elif not claimed:
        problems.append(f"no encuentro la cifra de signos en {COPY.name}; revisa el patron")

    return problems


def main() -> None:
    problems = failures()
    manifest = json.loads(MANIFEST.read_text())
    print(f"conceptos {len(manifest['concepts'])}   firma {manifest['signatureLength']} "
          f"({manifest['frames']} fotogramas)   pesos {WEIGHTS.stat().st_size} bytes")
    if problems:
        for problem in problems:
            print(f"  FALLO: {problem}")
        sys.exit(1)
    print("  los artefactos publicados cuadran entre si")


if __name__ == "__main__":
    main()
