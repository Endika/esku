# Measuring the app in a real browser

Run locally, by hand. CI never runs this: it needs a Chrome binary, a corpus and a machine
fast enough to matter, and none of those belong in a pull request check.

## Why it exists

Every other number this project has comes from replaying landmarks offline — `tools/train`.
Offline replay carries the corpus's own frame rate baked in, and SWL-LSE is 20.00 fps in all
300 of its reference videos. That blind spot hid the bug that made the app write nothing:
every `SignSegmenter` threshold was a frame count tuned at 20 fps, so on a phone reaching
fewer frames per second the same sign never survived. `simulate_app.py` reported 0.696 while a
browser wrote nothing at all, and no offline test could tell the difference.

The first browser run showed it immediately: 23 frames in 16 s, six windows discarded as too
short, the vocabulary engine asked **zero** times.

## Setup

Three things, none of them committed.

**1. Chrome.** `@puppeteer/browsers install` shells out to `unzip`, which is not always
present — it leaves an empty version folder and fails with "the executable is missing". Pull
the archive directly instead, and restore the exec bit that Python's `zipfile` drops:

```bash
cd tools/browser
curl -sLO "https://storage.googleapis.com/chrome-for-testing-public/151.0.7922.76/linux64/chrome-linux64.zip"
python3 -c "
import zipfile, os
z = zipfile.ZipFile('chrome-linux64.zip'); z.extractall('.')
for i in z.infolist():
    mode = i.external_attr >> 16
    if mode: os.chmod(i.filename, mode)
"
./chrome-linux64/chrome --version   # must print a version
```

**2. Clips.** SWL-LSE's reference videos, one per sign — 24 MB, not the 3.5 GB landmark
archive. CC-BY-4.0, and nothing from it is committed (see `../train/README.md`).

```bash
curl -L -o videos.zip "https://zenodo.org/api/records/13691887/files/VIDEOS_REF.zip/content"
python3 -c "import zipfile; zipfile.ZipFile('videos.zip').extractall('videos')"
```

`../train/data/videos_ref_annotations.csv` maps each filename to its label — the files
themselves are named `recXXXXXXXX.mp4`, so you need it to find a given sign.

**3. Dependencies.** `npm install` here, not in the app. The app must never ship a browser
automation dependency.

## Running

```bash
cd ../..                                      # repo root
npm run build
npx vite preview --port 4199 --base /esku/    # see the trap below
```

Then, in another shell:

```bash
cd tools/browser
node harness.mjs videos/VIDEOS_REF/reck9BdXzyIJjiWws.mp4    # DOLOR
```

**The trap:** `vite preview` does *not* apply the production `base`. `vite.config.ts` sets
`base` only when `command === 'build'`, so preview serves at `/` while the built `index.html`
asks for `/esku/assets/…`. Worse, the SPA fallback answers **200 with index.html** for every
missing path, so checking status codes suggests everything is fine while the app never boots.
Pass `--base /esku/` and verify content, not status.

| variable | default | what it is for |
| --- | --- | --- |
| `BASE` | `http://localhost:4199/esku/` | where the built app is served |
| `CHROME` | `./chrome-linux64/chrome` | browser binary |
| `SECONDS` | `20` | how long to sign at it (the clip loops) |
| `PLAYBACK` | `1` | clip speed. Below 1 the signer is genuinely slower — it no longer compensates for a slow pipeline, because the thresholds are in time now |
| `GL` | swiftshader | `auto` lets Chrome pick. Measured: no faster here, WebGL falls back to software regardless |

## The other script: `layout.mjs`

Same browser, no corpus and no camera — it measures the built page at 390×844 and 320×640 and
prints page height, horizontal overflow, the height of the action bar and the buttons actually
visible. It serves `dist/` itself, so the `vite preview` trap above does not apply:

```bash
cd ../..
npm run build
node tools/browser/layout.mjs
```

It exists because the page had grown to 3.1 screens with ten buttons on it and every review
read fine — jsdom computes no heights, so nothing in `npm test` could see it. Two readings it
takes are forced rather than reached (`is-running` and a filled transcript): both need a camera
and a recognised sign, and what is being measured is the layout they produce.

## What it asserts, and what it only measures

It **asserts** what must hold on any device, and exits non-zero:

- the clip decoded at all (a missing codec must not read as "recognised nothing")
- the vocabulary weights loaded — this is what catches a base-path regression 404ing them
- at least one frame contained a hand
- and, *only when the measured frame rate clears the shipped floor*, that the engine was asked

It **measures** and prints, without asserting: frame rate, windows closed against windows
discarded as short, raw unfiltered scores, the per-body-part feature profile against its
training reference, and the transcript.

Recognition is deliberately not asserted. It needs a frame rate a software-WebGL headless box
does not reach — about 1.3 fps here against a floor of 3.5 — and a permanently red check
invites someone to lower a shipped threshold to make it green. The harness reads
`minSignMs` and `minFrames` straight out of `SignSegmenter.ts` rather than copying them, so it
cannot drift from the app it is measuring; if they are renamed it throws instead of guessing.

## Known open finding

The signing hand lands in the **left** slot where the corpus's own reference video used the
right — right-hand block 100% empty against the 24% the training split shows. MediaPipe Tasks
and the legacy Holistic the corpus was extracted with appear to disagree on handedness. The
model tolerates it because SWL-LSE is itself mixed (DOLOR has one test sample in each slot),
but it is costing accuracy.
