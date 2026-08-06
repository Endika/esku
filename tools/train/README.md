# Training the LSE vocabulary head

Run offline, once, on a machine with the dataset. CI never runs this; the app only ever
loads the exported `.onnx` from `public/models/`.

## Dataset

[SWL-LSE](https://zenodo.org/records/13691887) — CC-BY-4.0, so the derived weights are
redistributable with attribution (see `../../NOTICE.md`).

```bash
# 3.5 GB. MediaPipe landmarks are already extracted, so no video decoding and no
# GPU-hours of pose estimation — this is the reason SWL-LSE was chosen over the
# alternatives.
curl -L -o data/MEDIAPIPE.zip \
  "https://zenodo.org/api/records/13691887/files/MEDIAPIPE.zip/content"
curl -L -o data/ANNOTATIONS.zip \
  "https://zenodo.org/api/records/13691887/files/ANNOTATIONS.zip/content"
```

`data/` is gitignored. Nothing from the dataset is committed — only weights derived from it.

## Two rules that will silently ruin the model if broken

1. **Normalisation must match `src/domain/landmarks/services/normalizeHand.ts` exactly** —
   wrist-centred, scaled by palm width, left hands mirrored. A mismatch does not throw; it
   just predicts noise.
2. **Feature order must match `toFeatureVector`** — flattened `[x, y, z]` per landmark, in
   MediaPipe's own point order.

## Class merging

The 300 labels contain variants of the same concept (`AZUCAR`, `AZUCAR2`, `AZUCAR(2M)`,
`AZUCAR(M-ES)`, `AZUCAR(M-ES)(2M)`) and one row labelled `#N/A`.

Drop `#N/A`, then merge on the concept id produced by `conceptIdOf` in
`src/domain/recognition/value-objects/Gloss.ts` — strip `(…)` markers and trailing digits.
This lands around 180–200 classes instead of 300, which both reads better in a transcript
and trains better: fewer classes, more examples each.

Keep the concept mapping in the exported label file so the app can name what it predicts.

## Measuring

Three benchmarks, and the third exists because the first two agreed with each other and
with nothing else.

| script | what it answers |
| --- | --- |
| `simulate_app.py` | accuracy on isolated signs, fed the way the app segments them |
| `continuous.py` | how many signs survive **fluent** signing, by splicing test recordings into unbroken streams with known boundaries |
| `sweep_continuous.py` | re-tunes the segmenter against both at once — they pull in opposite directions |

`continuous.py` is what found the bug that made the app write nothing: the old segmenter
scored 0.739 isolated and **0.146** continuous. Everything the project measured came from
isolated signs, and the app is used on fluent signing. Note it splices isolated recordings
and so cannot reproduce real co-articulation — read it as an upper bound.

### Checking against a second corpus

`check_calse.py <videos> [per-signer]` runs the segmenter over an unrelated LSE corpus of
isolated signs and counts how many windows each single-sign video produces. It should be
one. Against CALSE100 it is 2.56, with the median window landing exactly on
`minSignFrames` — the segmenter closes at the first permitted moment rather than at a
boundary, which means the tuning above is fitted to SWL-LSE rather than to signing.

That corpus is not in this repository and its terms of use are unknown, hence the path
argument. Do not train on it or redistribute any part of it without establishing a licence.
