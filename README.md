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
| **Vocabulary** | Whole LSE signs, one word each. | ONNX model trained on [SWL-LSE](https://zenodo.org/records/13691887): 300 classes, ~180 concepts after merging variants. Health domain. **Not built yet.** |
| **Taught** | Any sign you record yourself, in any sign language. | Nearest-prototype match over 3+ recordings, stored in IndexedDB on your device. Working now. |

### What it does not do

It does **not translate grammar**. LSE has its own syntax — topic-comment order,
classifiers, meaningful use of space — and Esku reads a *sequence of signs*, not a sentence.
Sign `[YO] [CABEZA] [DOLOR]` and you get "yo cabeza dolor", not "me duele la cabeza".

Continuous sentence-level sign language translation is an open research problem. Treating
word-level output honestly is a design decision, not a missing feature.

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

**Normalisation must match training.** `normalizeHand` mirrors what `tools/train` applies to
the dataset. A model trained on normalised coordinates and fed raw ones predicts noise
silently rather than failing — if recognition degrades for no visible reason, check this
first.

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

## Licence

MIT for the code. The trained weights derive from a CC-BY-4.0 dataset — see
[`NOTICE.md`](NOTICE.md) for attribution and for why sign.mt and LSA64 are deliberately
not used.
