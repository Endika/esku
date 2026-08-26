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

### The floor and the gate, settled: 1150/0.50 → 850/0.60

With the classifier fixed, the second finding was re-tested and the trade reversed. Three
columns, because two of them alone would pick the wrong config. Continuous recall and pause
babble come from `health_words.py` on the four held-out signers — 1,060 annotated instances of
the 51 frozen classes, and 211 windows landing in gaps between annotated sentences. Isolated is
the `app segmenter` row of `simulate_app.py`, 598 SWL-LSE samples, shipped model.

| `MIN_SIGN_MS` | gate | continuous recall | words written into pauses | isolated top-1 |
| ---: | ---: | ---: | ---: | ---: |
| 1150 | 0.50 | 33.2% | 27.3% | 65.3% |
| 1000 | 0.50 | 36.4% | — | 62.4% |
| 900 | 0.50 | 38.7% | — | 60.9% |
| **850** | **0.60** | **38.1%** | **27.5%** | **59.9%** |
| 850 | 0.50 | 41.8% | 33.6% | 59.9% |
| 800 | 0.50 | 43.1% | — | 57.1% |
| 750 | 0.60 | 42.2% | 26.7% | 55.6% |
| 750 | 0.50 | 46.2% | — | 55.6% |
| 500 | 0.70 | 42.4% | 23.2% | 44.4% |
| 500 | 0.50 | 53.0% | — | 44.4% |

Three findings, and only the first was expected:

- **Lowering the floor is not paid for in false positives.** Hold the babble rate at ~27% and
  recall still climbs as the floor drops: 33.2% at 1150, 38.1% at 850, 42.2% at 750. The high
  floor was not buying silence, it was losing signs. 850 is where the ratio is best, +8.6 for
  −5.4 of isolated, and it is the value the pre-retrain comment in `SignSegmenter.ts` had
  explicitly rejected at −8 for +3 — the same decision against a classifier that has seen
  co-articulation comes out the other way.
- **The gate is a poor discriminator, which is the real ceiling.** From 0.30 to 0.80 recall
  halves (49.6% → 24.3%) while babble falls only from 48% to 21%. Even at the strictest gate
  **one pause in five gets a word written**. Confidence does not separate signing from not
  signing, because the model has never been shown what not-signing looks like — the negatives
  `health_dataset.py` cannot build from this corpus are the same ones missing here.
- **Read the seed, not the run.** Scored on the shipped weights alone the change looks like
  +4.9; over all four seeds of that variant it is +2.3, because seed 29 is the worst at 1150
  and among the best at 850. The four-seed figures for the two shipped pairings:

  | pairing | continuous recall | words written into pauses |
  | --- | ---: | ---: |
  | 1150 / 0.50 (before) | 35.0% (sd 1.2) | 30.6% (sd 2.5) |
  | 850 / 0.60 (now) | 37.3% (sd 0.9) | 30.0% (sd 2.4) |

  +2.3 points of recall at a flat babble rate, about three standard errors. Modest, and the
  only pairing on the grid that improves one axis without giving back the other.

The grace period buys nothing and is not shipped: 0 and 600 ms are bit-identical at every
floor, because a window is longer than the grace and nothing ever groups; 1500 and 2500 lose
8-10 points.

**What the false-positive column can and cannot see.** Only a gap *between* annotated sentences
counts as silence. Inside a sentence, a window matching no gloss is usually a real sign nobody
labelled — 101 annotated sign types against the 90-150 signs a minute of fluent discourse — so
it cannot be called an error. A window straddling a boundary counts as speech. Both choices
undercount, deliberately: the figure is a floor on the babble, never a flattering ceiling. And
n is 211, so each rate carries about ±6 points; the trend across gates is monotone and solid,
neighbouring pairs are not separable.

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

## Fingerspelling — the alphabet engine, measured at last

`HandshapeAlphabetClassifier` shipped for its whole life as the one engine with no number
against it: "geometric handshape rules, no training data needed", and nothing else. LSE-FS-UVigo
(Zenodo `10.5281/zenodo.15797079`, CC BY 4.0) is 3,044 sequences of real continuous
fingerspelling with the corpus's own signer-disjoint split. Its held-out 456 are the exam.

| script | what it answers |
| --- | --- |
| `lsefs_extract.py` | caches the held-out split's hands from the 7.3 GB `RAW_KPS.zip`, once |
| `tools/bench/alphabet.bench.ts` | what the **shipped TypeScript** spells, run unmodified |

The bench is TypeScript on purpose. Every other bench here reimplements the app in Python and
then has to prove the port faithful; that proof has failed twice, once on handedness and once on
frame-counted thresholds. Running the real table, the real geometry and the real stabiliser
removes the question. `npm run bench:alphabet`, after `python lsefs_extract.py` once.

### The number

**0 of 456 words spelled exactly. 0.919 edit operations per annotated character.**

That total hides the shape of the failure, which is the whole point of the breakdown:

| | count | share |
| --- | ---: | ---: |
| letters annotated | 4,321 | |
| letters the app wrote | 742 | 17% of annotated |
| of those, correct | 354 | **48% of what it wrote** |
| letters never written | 3,581 | **83% of annotated** |
| substitutions | 386 | 9% |
| insertions | 2 | 0% |

**The engine is not wrong so much as silent.** When it commits to a letter it is right about half
the time, which for a table written by hand against no data at all is not a bad table. It just
almost never commits.

### Why it stays silent, and what that costs to fix

- **85.7% of frames with a hand in them produce no candidate at all.** The best-scoring template
  on the median such frame reaches **0.406** against a 0.72 floor; p75 is 0.648 and even p90 only
  0.758. This is not a threshold set slightly too high. Fluent fingerspelling is mostly
  *transitions* — the engine's own docstring says "letters are held, not travelled", and at
  conversational speed they are barely held.
- **Three consecutive agreeing frames is the second gate.** With candidates arriving on one frame
  in seven, three in a row of the same letter is rare, so even the frames that do fire mostly
  never reach the transcript.
- **The table covers 14 letters of 27** — a b c d e f i l o s u v w y. Only **24 of the 456 test
  words (5.3%)** contain no letter outside it, so exact-word accuracy was capped at 5.3% before
  a single frame was read, the same arithmetic ceiling `MIN_SIGN_MS` imposed on short signs. The
  missing letters are the common ones: R touches 53% of the test words, N 44%, T 35%, M 29%.
- Also worth fixing while there: **`k` is in neither list** — not in `LSE_ALPHABET` and not in
  `UNSUPPORTED_LETTERS`, so the UI does not warn about it.

The most-confused pairs are O→F (34), I→Y (18), A→D and O→D (12 each). D, F, C and Y absorb most
of the wrong answers: the permissive templates win whenever the real shape is mid-transition.

### What this says about retraining

Per the decision rule set before the measurement: the failure is **deletions with low top
scores**, not substitutions and not a mistuned threshold. Loosening the floor would buy noise —
p50 is 0.406, so reaching most frames means accepting near-random shapes — and editing the table
cannot help a frame that is genuinely between two letters.

So the geometric rules do not describe these hands, and a learned handshape classifier is
justified. Two things it must do that the table does not:

1. **Cover the alphabet.** Accuracy on 14 letters cannot exceed 5.3% word accuracy here.
2. **Say something about transition frames** — either recognise mid-motion shapes or abstain in
   a way the stabiliser can use, which is the same missing-negatives problem the vocabulary
   engine has.

LSE-FS labels the spelled word, not the letters, so supervision has to come from alignment or
CTC. That is a separate project with its own spec, and 2,158 train plus 430 validation sequences
are untouched and waiting for it.

### Reading the number fairly

The labels are health-domain: 456 words, some of them multi-token — 87 contain a space, 12 an
`@` and 11 a `.`, because the corpus includes spelled emails and addresses. Those are scored as
written, which is honest but means the figure is not "accuracy on Spanish words". Nothing here
extrapolates to fingerspelling in general, and the corpus is not redistributed: `data/` is
gitignored, and only derived numbers appear in this file.
