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
### Real co-articulation: LSE-Health-UVigo

`continuous.py` had to splice recordings because nothing here held real co-articulated signing.
LSE-Health-UVigo does: 273 videos, 10.8 hours of continuous discourse from ten signers, and
15,098 hand-annotated gloss occurrences with start and end times. That is both the missing
benchmark and, once cut into windows, the missing training set — so the split below exists to
keep those two uses apart.

The corpus is CC BY-NC on Zenodo (DOI `10.5281/zenodo.10234465`) while the reply that pointed us
at it said CC BY; until that is settled, measure locally and do not publish weights derived from
it. Nothing from it is committed: `data/` is gitignored.

| script | what it answers |
| --- | --- |
| `health_extract.py` | caches hand, pose and face landmarks per video, once. `--mirror` is opt-in: see the handedness note below |
| `health_bench.py` | does a window land where a sign is? IoU-matched, one-to-one, against the 15,098 boundaries |
| `health_words.py` | does the app write the **right word**? The only number about the product. Scores by label, so a retrained model with different classes stays comparable |
| `sweep_health.py` | sweeps `MIN_SIGN_MS`, plus `--by-duration` for recall split by how long the sign lasts |
| `health_split.py` | freezes a signer-disjoint train/test split into `data/uvigo/split.json` |
| `health_dataset.py` | cuts the corpus into training windows: gold boundaries, real segmenter windows, and rejects |
| `health_train.py` | retrains on those windows and reports on held-out signers |

**Handedness — the same wrong belief was written in four places.** «MediaPipe labels the mirrored
selfie view» appeared in `extract.py`, `extract_raw.py`, `check_calse.py` and the app's
`handednessFor`, and it is false: `getUserMedia` hands over the sensor's own frames and studio
footage is third person, so the label is already anatomical everywhere. Inverting it in the two
readers of SWL-LSE, but not in LSE-Health's extraction, taught one model **two opposite
conventions at once** — 2.4% anatomical against 98.4% — and since the signature reflects the left
block into the right hand's space, the disagreeing half arrived mirrored. It survived because no
offline bench goes through the app's mapping, and it was found by a user saying the app used to
understand more.

`check_convention.py` is what stops it recurring: it measures each corpus's right block against
Pose's own wrists, which carry no convention of their own, and fails when two corpora disagree.
Run it before training on any new corpus.

The footage is third person, not a selfie mirror, so MediaPipe's raw label is
already anatomical and must not be inverted. Verified three ways — it agrees with Pose's
anatomical wrist in 98.6% of detections, and inverting it sends both left-handed signers (6 and 7)
to the wrong slot. `check_calse.py` had been inverting unconditionally and no longer does.

Two findings from it worth not rediscovering:

- **A minimum-duration floor makes short signs unreachable, as arithmetic.** A window forced to
  last at least F, matched against a sign of length L, has IoU ≤ L/F, so IoU ≥ 0.5 needs
  L ≥ F/2. With `MIN_SIGN_MS` at 1150 and a median real sign of 480 ms, **828 of 1,576 glosses
  could never be recovered**, and recall by duration is flatly 0% below F/2 in both columns of
  `sweep_health.py --by-duration`.
- **Boundary recall is not a product metric.** Lowering the floor tripled it while word recall
  stayed near zero, because 88-96% of windows never reached 0.30 confidence: the classifier had
  only ever seen dictionary recordings. Fixing the classifier first is what makes the boundary
  work pay — the two levers multiply.

### The floor, settled: 1150 → 850

With the classifier fixed, the second finding was re-tested and the trade reversed. Both columns
below are the same knob at `gate 0.50`, `grace 0` — the app's own `CandidateStabilizer(1, 0.5)` —
scored on the four held-out signers, 1,060 annotated instances of the 51 frozen classes, against
the shipped model. Isolated is the `app segmenter` row of `simulate_app.py`, 598 SWL-LSE samples.

| `MIN_SIGN_MS` | continuous word recall | isolated top-1 |
| ---: | ---: | ---: |
| 1150 | 33.2% | 65.3% |
| 1000 | 36.4% | 62.4% |
| 900 | 38.7% | 60.9% |
| **850** | **41.8%** | **59.9%** |
| 800 | 43.1% | 57.1% |
| 750 | 46.2% | 55.6% |
| 500 | 53.0% | 44.4% |

The trade stays favourable down to 750 and inverts by 500. 850 has the best ratio, +8.6 for −5.4,
and it is the value the pre-retrain comment in `SignSegmenter.ts` had explicitly rejected at −8
for +3 — the same decision, measured against a classifier that has seen co-articulation, comes
out the other way. Re-run over all four seeds of the shipped variant, the two ends read **34.9%
(sd 1.2)** and **41.9% (sd 0.9)**: +6.9 points, same direction in every seed.

Two knobs that did *not* move:

- **The grace period buys nothing.** 0 and 600 ms are bit-identical at every floor, because a
  window is longer than the grace and so nothing ever groups; 1500 and 2500 lose 8-10 points.
- **The gate stays at 0.50.** Lower reads better here — 55.1% at 0.30 with the 750 floor — but
  `health_words.py` counts recall per instance and never counts a wrong word, so it cannot price
  a looser gate. Moving it needs a precision measurement this bench does not make.

`health_dataset.py` deliberately cannot build a useful reject class from this corpus. "Overlaps no
gloss" is not evidence of silence — only 101 sign types are annotated, about 24 glosses a minute
against the 90-150 signs a minute of fluent discourse, so such a window is usually a real sign
nobody labelled. The honest source is the gap between annotated *sentence* segments, and those
cover 9.6 of the 10.8 hours, so there is almost nothing there. Teaching the model to stay quiet
needs negatives from somewhere else.

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
