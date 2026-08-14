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

### One seed is not a measurement

`experiment.py` compares feature variants on the same split, same budget, and — until
2026-08-14 — the same single seed. Sweeping seeds 7/13/29/41 over the same variants shows the
between-seed spread is as large as the gains the project had been crediting:

| variant | mean top-1 | sd | min | max |
| --- | --- | --- | --- | --- |
| pose + 16 frames, no face | 0.722 | 0.011 | 0.706 | 0.731 |
| + face expression, every frame | 0.716 | 0.024 | 0.694 | 0.741 |
| + face expression, held ≤300 ms | 0.714 | 0.008 | 0.704 | 0.724 |
| no torso scalars | 0.713 | 0.011 | 0.701 | 0.726 |

**The face block's `+1.2` was the max of four seeds.** Its mean is *below* no-face at all, and
which side wins flips with the seed. The torso block behaves differently: +0.9 on average and
the same sign in three seeds of four, which reproduces the `+1.0` originally credited to it.
Neither reaches significance at n=4 — the standard error of a difference here is about 0.008 —
but small-and-consistent and zero-and-erratic are not the same finding.

What decides those two is not the significance, it is the cost. The torso scalars are derived
from pose landmarks the pipeline must have anyway, since hand coordinates are torso-relative;
they cost five subtractions. The face block costs a whole `FaceLandmarker` pass per frame, and
frame rate is what decides whether the app writes anything at all.

**This is not a reason to delete the face block.** SWL-LSE is a dictionary: isolated signs in
citation form, so there is no negation, interrogative or topicalisation for non-manual features
to mark, and the mouthings that distinguish manually identical signs never occur. What is
measured is that these six scalars buy nothing *on this corpus for this task*. The richer
`face points (21 located)` variant also measured worse, which reads as too little data to learn
the face at all rather than as a verdict on the face.

Two flags exist for this. `face_hold_ms` simulates running the face model less often and
holding the last reading — asked for in **milliseconds**, not frames, because the corpus is
20.00 fps and the app is nowhere near it, so "every third frame" means 150 ms here and ~600 ms
on a phone. And `use_torso` is separate from `use_pose` on purpose: turning off `use_pose`
would also drop the pose-relative hand location, which is the one large gain (+5.7), so
measuring the torso through it would answer a different question.

Rule of thumb from this: on this corpus, do not accept a gain under two points from one seed.

### Checking against a second corpus

`check_calse.py <videos> [per-signer]` runs the segmenter over an unrelated LSE corpus of
isolated signs and counts how many windows each single-sign video produces. It should be
one. Against CALSE100 it is 2.56, with the median window landing exactly on
`minSignFrames` — the segmenter closes at the first permitted moment rather than at a
boundary, which means the tuning above is fitted to SWL-LSE rather than to signing.

That corpus is not in this repository and its terms of use are unknown, hence the path
argument. Do not train on it or redistribute any part of it without establishing a licence.

### Shipping a corpus as a taught-sign pack — measured, and it does not work

`prototype_pack.py <videos> [--limit N]` asks whether a small corpus can ship as a
nearest-prototype pack, the way user-taught signs already do. Six examples per sign is far
too few to train on and, in principle, plenty to match against.

Against CALSE100 it scores 43.3% top-1 on a held-out signer over twenty words — well above
the 5% chance baseline, so the signs are separable. But the confidence is unusable: the
median distance to the correct sign is 0.837 and to the best *wrong* sign 0.775. Lowering
the threshold, recalibrating `DISTANCE_SCALE` and dropping the raw wrist position all left
top-1 untouched.

`windowSignature` is built for one person repeating their own sign and carries no invariance
across bodies, cameras and styles. The negative result is recorded here so nobody spends
another afternoon on it; revisit only with a corpus that has many more signers per sign.
