# Esku ✋

Offline PWA that reads sign language from your phone camera and turns it into text.

**Live:** <https://endika.github.io/esku/>

Everything runs on the device. No frame, landmark or transcript ever leaves the browser —
there is no backend to send them to. See [`.github/SECURITY.md`](.github/SECURITY.md).

## What it recognises

Esku recognises **signs and letters, one at a time**, and appends them to a running text.
Three engines answer through one port, so the app does not care which one produced a hit:

| Engine | What it reads | Where it comes from |
| --- | --- | --- |
| **Alphabet** | Fingerspelled letters (dactilológico). Spell anything, letter by letter. | Geometric handshape rules — no training data needed. |
| **Vocabulary** | Whole LSE signs, one word each. | GRU over 287 concepts, trained on SWL-LSE and LSE-Health-UVigo. **73% top-1** sign by sign; **33%** written correctly signing continuously. |
| **Taught** | Any sign you record yourself, in any sign language. | Nearest-prototype match over 3+ recordings, stored in IndexedDB on your device. Working now. |

### What it does not do

It does **not translate grammar**. LSE has its own syntax — topic-comment order,
classifiers, meaningful use of space — and Esku reads a *sequence of signs*, not a sentence.
Sign `[YO] [CABEZA] [DOLOR]` and you get "yo cabeza dolor", not "me duele la cabeza".

Continuous sentence-level sign language translation is an open research problem. Treating
word-level output honestly is a design decision, not a missing feature.

### How good is the vocabulary model, really

Two numbers, because one of them alone would be misleading.

**Sign by sign, at a deliberate pace: 73% top-1, 84% top-3** on 598 held-out samples of isolated
signs — against a 0.4% random baseline over 287 classes.

**Signing continuously: 33% of signs get the right word written** — but read the scope before
quoting that. Measured end to end, camera to transcript with the confidence gate included, over
1,060 hand-annotated occurrences from four held-out signers of
[LSE-Health-UVigo](https://zenodo.org/records/10234465). Mean of four seeds, sd 2.1.

The scope is narrow on purpose: **51 health-domain signs scored inside a 287-sign model**. It does
not extrapolate. Recognising isolated signs over 300 classes, over the ~8,000 of a full LSE
dictionary, and recognising continuous signing over a large vocabulary are three different
problems of increasing difficulty, and this figure belongs to the easiest of the three. Nobody has
solved blind segmentation of continuous signing.

What the number does say is that the same app scored **0.6%** on the same test before it was
trained on co-articulated signs. The gap between the two rows is what a sign looks like when it is
glued to the sign before it.

Useful, not authoritative. The UI says so, and the transcript is editable.

Every input was measured rather than assumed. Starting from hands alone:

| Input | top-1 | top-3 |
| --- | --- | --- |
| hands only, 8 frames | 0.632 | 0.826 |
| + hand position relative to the **torso** | 0.689 | 0.836 |
| + 16 frames instead of 8 | 0.719 | 0.849 |
| + **torso and head** orientation | 0.729 | 0.865 |
| + **facial expression**, 6 scalars | **0.741** | **0.870** |

Rejected, with numbers rather than opinions: frame-to-frame motion deltas (0.666), raw face
coordinates instead of derived ratios (0.702), input augmentation (0.699). Dropping depth
entirely costs 0.003 — MediaPipe's `z` is inferred from one camera rather than measured, so
it carries far less than it looks like it should.

Why the body helps so much: "hand at chin height" is a fixed number in body coordinates and
a moving one in image coordinates. Normalising against shoulder width makes it invariant to
how far the signer stands from the camera, and location is phonemic in LSE.

Why six face scalars beat sixty face coordinates: with ~27 examples per class, coordinates
the model would have to derive eyebrow-raise and mouth-openness from are capacity spent
memorising faces.

Those numbers come from SWL-LSE's own train/val/test split, never from data the model saw.
`tools/train/train.py` prints them on every run and writes them into the shipped manifest, so
the figure in this README cannot drift from the model that is actually deployed.

**There is no inference runtime.** onnxruntime-web needs 13 MB of WASM to run a 2.4 MB model,
and on GitHub Pages it cannot even use threads — Pages sends no COOP/COEP headers. The
network is a fixed stack (LayerNorm → 2-layer bidirectional GRU → mean-pool → 2 linear
layers), so `src/infrastructure/recognition/gru.ts` computes it directly and the weights ship
as one flat float32 blob. `VocabularySignClassifier.test.ts` checks the whole stack against
logits PyTorch produced for a fixed input, because a hand-written GRU that is subtly wrong
still runs and still returns plausible numbers.

## Architecture

Hexagonal, with constructor injection and no DI framework — a plain `Container` wires
concrete adapters into use cases at startup.

```
src/
  domain/          pure model and ports; no browser, no I/O, no dependencies
    landmarks/     geometry: handshape description, normalisation
    recognition/   segmentation, stabilisation, ISignClassifier port
    transcript/    the running text
  application/     use cases, orchestrating ports
  infrastructure/  adapters: MediaPipe camera, ONNX runtime, IndexedDB
  presentation/    UI
  bootstrap/       Container — the only place adapters meet use cases
```

The domain is where the interesting logic lives and it is fully testable without a camera:
`SignSegmenter` decides where one sign ends and the next begins, `CandidateStabilizer` stops
a jittering classifier from spelling `AAAABAAAA`, `handShape` reduces 21 landmarks to scale-
and handedness-invariant ratios, and `windowSignature` collapses a whole sign into one
fixed-length vector so two performances of it can be compared.

**Signature similarity is Euclidean, not cosine, and that is load-bearing.** Every hand
shares the same gross structure, so cosine scored a fist against an open hand at 0.965 and an
index point against a Y at 0.962 — no threshold separates those. Distance over already
normalised coordinates gives 0.10 and 0.25 for the same pairs. If taught-sign recognition
ever starts matching everything, check that this has not been "simplified" back to cosine.

**Two signatures, on purpose.** `vocabularySignature` feeds the trained model and changes
whenever the model does. `windowSignature` describes a *taught* sign and is frozen, because
the user's own recordings are stored against it — they used to share one function, and every
model improvement invalidated everything the user had taught.

**Normalisation must match training.** `vocabularySignature` and
`tools/train/vocabulary_features.py` must produce byte-identical layouts, and
`vocabularySignature.test.ts` checks that element by element against a fixture the trainer
writes. A model trained on one and fed the other predicts noise
silently rather than failing — the loader throws on a length mismatch, but a same-length
reordering would slip through the loader, which is exactly what the parity test is for. If
recognition degrades for no visible reason, diff those two files first.

**Known limitation in taught signs.** The signature carries the wrist position because LSE
gives location meaning, but it is 3 floats out of 66 per hand, so plain distance matching
barely weights it. The trained model learns its own weighting and copes; taught-sign matching
does not, so two taught signs differing only in height will be confused. Covered by a test
that asserts the real figure rather than a hoped-for one.

## Development

```bash
npm install
npm run dev        # vite dev server
npm test           # vitest
npm run typecheck  # tsc --noEmit
npm run lint       # biome
npm run build      # tsc && vite build
npm run icons      # regenerate PWA icons from public/favicon.svg
```

**The camera cannot be tested under WSL2** — no device access. Use a browser on the host OS
or a real phone against the dev server over the network.

### Engine assets

The recognition engine is served same-origin from `public/`, never a CDN — the page holds
camera permission, so a third party must not be able to serve executable code into it.

- **WASM** is staged from `node_modules` by `scripts/copy-wasm.mjs`, run automatically before
  `dev` and `build`. Not committed: 22 MB in every clone, and a committed copy can drift from
  the `@mediapipe/tasks-vision` version that loads it. `public/wasm/` is gitignored.
- **`hand_landmarker.task`** (7.5 MB) is committed, since it is not published on npm.

Neither is precached by the service worker. Together they are ~29 MB, and precaching would
put that download in front of the first paint; they are cached on first use instead, after
which the app is fully offline.

## Privacy

Everything runs on the device. The camera feed is never recorded, uploaded, or sent anywhere —
there is no server to send it to. No cookies, no analytics, no identifiers, and so no consent
banner. Taught signs live in the browser's own storage and are deletable from the app.

Full notice, in Spanish, as the app's users read it: [`PRIVACY.md`](PRIVACY.md).

## Licence

**MIT for the code. The vocabulary model is not** — it is trained partly on LSE-Health-UVigo,
which Zenodo declares CC BY-NC 4.0, so the weights ship under
[`public/models/LICENSE.md`](public/models/LICENSE.md) instead. Using Esku is unaffected; a
commercial fork would have to retrain on SWL-LSE alone.

See [`NOTICE.md`](NOTICE.md) for attribution, and for why sign.mt and LSA64 are deliberately not
used.
