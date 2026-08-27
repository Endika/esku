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

### Re-downloading, so deleting is reversible

`data/uvigo/` holds ~18.6 GB of corpus archives. They are safe to delete when disk is tight,
but only because this table exists: without it, "safe to delete" means "safe to lose", and
finding the right Zenodo record again is the part nobody remembers.

| file | GB | record | needed for |
| --- | ---: | --- | --- |
| `RAW_KPS.zip` | 6.9 | [`10.5281/zenodo.15797079`][fs] | `lsefs_dataset.py` — the fingerspelling model |
| `PROC_KPS.zip` | 6.9 | [`10.5281/zenodo.15797079`][fs] | nothing in the pipeline; see below |
| `Videos-LSE-Health-UVigo.zip` | 4.8 | [`10.5281/zenodo.10234465`][h] | `health_extract.py` only — the landmark cache it builds already exists |
| `10234465_ELAN-…zip`, `.xlsx` | 0.002 | [`10.5281/zenodo.10234465`][h] | the annotations every LSE-Health bench reads |

[fs]: https://doi.org/10.5281/zenodo.15797079
[h]: https://doi.org/10.5281/zenodo.10234465

Everything under `data/uvigo/*.npz` and `*.ndjson` is derived and regenerates in minutes; delete
those freely.

**`PROC_KPS.zip` is not what it looks like.** It holds the same keypoints "preprocessed" —
wrist-centred and scaled — and the obvious question is whether their preprocessing beats ours.
Checked, and it does not: on the same frame their vectors correlate **1.0000** with
`normalize_hand`'s, differing only by a constant scale factor, which the first layer of any
network absorbs. There is nothing to gain and no experiment to run.

What it does carry that `RAW_KPS.zip` does not is the corpus's own **`handness`** label, and
that is worth keeping it for. Our dominant-hand rule picks the larger bounding box; audited
against those labels over 200 train sequences and 31,959 frames it agrees **96.8%** of the time
and takes the wrong hand in **3.2%**. Whether that matters is not yet measured, but it is the
only ground truth available for a rule the app has to apply live.

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

### Starred glosses: excluded from training, scored anyway — on purpose

2,178 of LSE-Health's 15,098 gloss occurrences carry a `*` prefix, and the two halves of this
pipeline treat them differently. That looks like an inconsistency and is not, so it is written
down here before someone "fixes" it.

The corpus authors explain the prefix as "slight drift from normal realization, or an OOV sign
very similar visually", and the *reason* for each one lives in a `VAR` tier inside the ELAN
bundle — one code per starred gloss, all 2,178 of them:

| code | n | what it is | usable? |
| --- | ---: | --- | --- |
| `SIM` | 472 | a different gloss that merely looks like this one | **no — the label is wrong** |
| `OCC` | 68 | the sign is occluded by another body part | no |
| `OUT` | 18 | the sign goes out of frame | no |
| `SHO` | 485 | very short sign, from speed and heavy co-articulation | yes, and it is the hard case |
| `MPH` `LAX` `MAN` `LOC` | 1,129 | morphology, relaxed execution, odd non-dominant hand, shifted location | yes — the real sign, executed differently |

**Training excludes them** (`health_split.py`, `starred_included: false`), on the corpus authors'
own advice: they are articulations that differ from the nominal one and would smear the classes.

**Scoring includes them.** `health_words.py:normalize` strips the `*` and treats the gloss like
any other. That is deliberate, and it makes the published figure the *strictest* of the
available policies rather than a flattering one — the 472 `SIM` occurrences ask the model for a
label belonging to a sign it is not being shown. Measured on the held-out signers at the shipped
floor and gate, everything else equal:

| policy | scorable instances | word recall |
| --- | ---: | ---: |
| **all starred scored (shipped)** | 1,060 | **38.1%** |
| drop only `SIM`/`OCC`/`OUT` | 996 | 39.7% |
| drop all starred | 849 | 43.1% |

So the published number is conservative by about 1.6 points against a defensible fair exam, and
by 5 against the flattering one. **Do not "fix" the asymmetry by excluding them from scoring
too** — that would raise the figure without the engine improving, which is the definition of
moving the goalposts. If anything is ever changed here, drop `SIM`/`OCC`/`OUT` only, and say so
next to the number.

The `SHO` group is worth reporting separately when the segmenter is under discussion: those are
exactly the short, heavily co-articulated signs that a minimum-duration floor makes
*arithmetically* unreachable, per the IoU ≤ L/F bound below.

### The floor and the gate: 1150/0.50 → 850/0.60, and later 0.45

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
  signing — though the reason written here for months was wrong, and the section below has the
  correction: the model *had* been shown what not-signing looks like, and this pipeline was
  counting that answer as a word.
- **Read the seed, not the run.** Scored on the shipped weights alone the change looks like
  +4.9; over all four seeds of that variant it is +2.3, because seed 29 is the worst at 1150
  and among the best at 850. The four-seed figures for the two shipped pairings:

  | pairing | continuous recall | words written into pauses |
  | --- | ---: | ---: |
  | 1150 / 0.50 (before) | 35.0% (sd 1.2) | 30.6% (sd 2.5) |
  | 850 / 0.60 (then) | 37.3% (sd 0.9) | 30.0% (sd 2.4) |
  | **850 / 0.45 (now)** | **44.2% (sd 0.7)** | **19.9% (sd 1.9)** |

  +2.3 points of recall at a flat babble rate, about three standard errors. Modest, and the
  only pairing on the grid that improved one axis without giving back the other — until the
  abstention below turned out to be inflating the babble column of every row, which is what
  made the third row affordable. Both babble figures in the first two rows count the model's
  `__NADA__` class as a written word; the third does not.

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

### The silence class was already trained, and nobody was reading it

`health_dataset.py` has cut a third source since the co-articulated retrain: **reject** windows,
taken from the gaps between LSE-Health's translated-sentence tier, labelled `__NADA__`. They went
into training, the class shipped inside the concept list — index 286 of 287 — and then **nothing
consumed it**. `VocabularySignClassifier` treated it as an ordinary concept and `createGloss`
humanised it to the word `__nada__`; `health_words.py` counted any emission as a written word. So
the app wrote **131 of 1,477 words as `__nada__`** on held-out signers, and the pause-babble
figure this file published included the model's own abstention as babble.

Fixing the reading, with nothing retrained and the weights byte-identical:

| policy at 850 / 0.60 | continuous recall | words written into pauses |
| --- | ---: | ---: |
| as published — abstention counted as a word | 38.1% | 58/211 (27.5%) |
| **A — abstention wins, nothing is written** | **38.1%** | **23/211 (10.9%)** |
| B — abstention dropped, runner-up written | 38.1% | 23/211 (10.9%) |

**A and B were identical to the word**, because under an abstention no real concept ever cleared
the gate. A is what ships: if the model answers "nobody is signing" at 0.95, writing the 0.46
runner-up contradicts the answer it just gave, and A stays correct if the gate moves. The class
is declared in the manifest as `abstentionConcept` rather than hardcoded in TypeScript, and
`check_manifest.py` fails on any `__`-prefixed concept that goes undeclared — this bug was two
places having to agree with nothing checking.

**The freed room bought recall.** With babble no longer counting the abstention, the gate could
come down. Four seeds, floor 850, grace 0, held-out signers:

| gate | continuous recall | babble | babble counting `__NADA__` | words/min |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | 50.9% (sd 1.1) | 30.7% (sd 3.7) | 51.5% | 27.6 |
| 0.40 | 46.4% (sd 1.1) | 23.0% (sd 1.8) | 43.0% | 21.9 |
| **0.45 (now)** | **44.2% (sd 0.7)** | **19.9% (sd 1.9)** | 39.1% | 19.5 |
| 0.50 | 41.8% (sd 0.9) | 17.4% (sd 1.5) | 35.9% | 17.4 |
| 0.60 (before) | 37.3% (sd 0.9) | 13.3% (sd 2.1) | 30.0% | 13.9 |

0.45 gains **+6.9 points of recall in every one of the four seeds** and still writes into fewer
pauses than the 30% that shipped for months. Lower gates keep paying and the babble goes with
them; 0.30 is back to the old published rate, which is the trade this corpus cannot defend.

**And the abstention does not cover the babble that remains — measured, not assumed.** The class
is precise: over windows holding a real sign it is argmax in 0.9% (segmenter cuts) and 0.1% (gold
cuts), and **never** reaches 0.60, which is why honouring it costs no recall at all. But over the
pause windows where the app was writing a *real* word, `P(__NADA__) > 0.30` in **0.0%** of them.
The abstention and the remaining babble are **disjoint**: it catches the easy pauses and stays
quiet on exactly the ones that fool the classifier.

That is the correction to the sentence this file used to carry. The negatives
`health_dataset.py` *can* build — genuine gaps between sentences — are the easy ones, and they
are already in. The ones that would fix the rest are not labellable here: a window inside a
sentence matching no gloss is usually a real sign nobody annotated, 101 annotated types against
90-150 signs a minute. **So no learned abstention is being built.** With 23 of 211 windows left
and ±6 points of sampling error on that column, a further 3 or 4 points could not be demonstrated
with this bench even if it worked.

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

### Replaced, and by how much

The table is gone. `CtcAlphabetClassifier` reads the same frames with a causal GRU trained on
LSE-FS's 2,158 held-in sequences, and the bench above — unchanged, scoring by output string,
which is why it could measure both — puts the two side by side on the same 456 test sequences.
Each engine is driven the way its own app version drove it: the table needed three agreeing
frames to damp its flicker, the CTC head needs one. Scoring both under one rule would flatter
whichever the rule was picked for.

| | words exact | edit distance / char | letters written | correct among them |
| --- | ---: | ---: | ---: | ---: |
| handshape table | 0/456 | 0.919 | 742 of 4,321 | 47.7% |
| **GRU + CTC** | **126/456** | **0.187** | **4,054 of 4,321** | **90.2%** |

It writes 5.5x as many letters and is right nine times in ten instead of fewer than five. And
**it beats the table on every one of the 14 letters the table even attempted** — five of which
the table scored a flat 0%. There is no hybrid worth building; the replacement is clean.

### How it works, and the one thing that had to change

`lsefs_dataset.py` cuts the corpus to one normalised dominant hand per frame — 63 floats from
`normalize_hand`, the app's own transform, and deliberately without the wrist position that
`hand_features` appends, because a letter is the same letter anywhere in frame.
`lsefs_train.py` fits a causal 2x128 GRU with CTC, 176,796 parameters, **0.71 MB**.

**Causal is not a preference.** The app classifies frame by frame on a live camera, so the
model may never see the future; the bidirectional GRU that CTC papers reach for is unusable
here.

**The blank is the point.** LSE-FS labels the spelled word and never says when each letter
happened, so CTC is what makes the corpus usable at all — but its blank class is also a
*trained* "no letter here", which a similarity table cannot express at any threshold. Those are
the negatives `health_dataset.py` could not build for the vocabulary head, and here they fall
out of the training method.

**And blank forced the one app change.** CTC posteriors are peaky by construction — the loss is
invariant to how long a letter is held, so the model spikes on a frame or two and blanks the
rest. Measured: non-blank runs are 26% one frame long and only 23% reach three. Under the old
three-frame agreement rule this model writes 550 letters of 3,758 and *not one whole word*;
at one frame it writes 2,981 and 53 words. So `RecognizeSignsUseCase` gives the alphabet engine
`CandidateStabilizer(1, 0.5)` and releases the latch whenever the engine abstains. With that
release, one-frame agreement plus the existing no-repeat latch **is** CTC's greedy collapse —
the rule was already there, with the wrong constant.

### The hand it reads, which turned out to matter more than the model

`dominantHand` picks the hand with the larger bounding box, and that is the best a single frame
allows. It is also close to a coin toss when it matters: PROC_KPS.zip carries the corpus's own
`handness` label, and audited against it over 200 sequences and 31,959 frames the rule takes the
wrong hand in **23.7% of the frames where both hands are visible**.

Those are only 11.8% of frames, so the arithmetic says 2.8% of frames overall — and the
arithmetic is badly wrong, because this engine carries hidden state. A wrong hand does not lose
one frame; it feeds a wrong shape into the GRU and corrupts everything after it. Measured end to
end on the same weights, only swapping the selection rule at inference:

| hand rule | CER | words exact |
| --- | ---: | ---: |
| largest span | 0.307 | 115/456 |
| accumulated motion | **0.166** | **139/456** |

`DominantHandTracker` is the fix: accumulated wrist motion with exponential decay, because the
spelling hand moves and a resting hand does not, and because decay follows a signer who changes
hands instead of remembering the first one. Retrained on hands chosen that way, validation CER
went from **0.255 (sd 0.005)** to **0.201 (sd 0.016)** over the same four seeds, and the test
figures in the table above are that model.

What it is *not* is "prefer the right hand". That scores 99% on this corpus, which is 199 of 200
right-dominant, and fails every left-handed signer — the same shape of mistake as the handedness
inversion recorded above. Signals ranked on two-hand frames: accumulated motion 92.1%, higher
hand 89.8%, instantaneous motion 84.1%, largest span 76.3%, widest finger spread 69.9%.

**`SignSegmenter` deliberately still uses the stateless rule.** It reads `dominantHand` to
measure motion for window boundaries, and changing that would move the segmentation the whole
vocabulary engine depends on. This change is scoped to the alphabet, where it risks nothing.

### What augmentation bought, and what it did not

The ceiling here is data, not capacity: 18,887 training letters over 27 classes, and validation
CER bottoms out and then worsens. A wider model buys overfitting, so the budget went to widening
the corpus instead. One seed each:

| | validation CER |
| --- | ---: |
| none | 0.349 |
| temporal resampling | 0.309 |
| rotation + noise + frame dropout | 0.272 |
| **all four** | **0.262** |

Two that are *not* there, and the reason matters. **Mirroring** looked free and is harmful:
`normalize_hand` already folds left hands into right-hand space, so negating x again
manufactures shapes no live frame carries. **Scale jitter** is pointless for the same reason —
every coordinate is already divided by palm width.

Batching is also worth a warning. Sorting sequences by length to kill padding waste sped
training up 2.4x and cost 0.036 of CER, because length correlates with word length and the
batches stopped being diverse. Shuffling within windows eight batches wide keeps the speed and
gives the accuracy back.

Final: **validation CER 0.255, sd 0.005 over seeds 7/13/29/41.** Tight enough that one run means
something here, unlike the 0.024 spread the vocabulary head showed.

### The letters it does not know

Recall per letter on the held-out signers is the number a single average would hide, and it
splits in two. **Measured weak:** Y 17%, J 32%, V 43%, Z 44%. **Too rare to judge:** K appears
3 times in the test split, Ñ 8, W 10 — those percentages move by one word. Both come from the
same place, the training set, where K appears 16 times in 18,887 characters, W 23, Ñ 28, Y 45.
Everything else runs 74-98%.

They are surfaced to the user as `WEAK_LETTERS`, and that list is a *measurement of the shipped
weights*: re-derive it from block 7 rather than copying it forward. The previous weights had Q
and X in it, and better hand selection alone took them to 55% and 79%.

### Reading the number fairly

The labels are health-domain: 456 words, some of them multi-token — 87 contain a space, 12 an
`@` and 11 a `.`, because the corpus includes spelled emails and addresses. Those are scored as
written, which is honest but means the figure is not "accuracy on Spanish words". Nothing here
extrapolates to fingerspelling in general, and the corpus is not redistributed: `data/` is
gitignored, and only derived numbers appear in this file.
