/**
 * Runs the real Esku build in a real browser, feeding it a known recording instead of a
 * camera, and reads the diagnostics panel the way a person would.
 *
 * Why this exists: every other measurement in this project replays landmarks offline, and
 * offline replay carries the dataset's own frame rate baked in. That blind spot cost a long
 * investigation — the app scored 0.696 in `tools/train/simulate_app.py` while writing nothing
 * in a browser, because every segmenter threshold was a frame count tuned at SWL-LSE's 20 fps.
 * Nothing offline could see it. This can.
 *
 * Deliberately *not* a vitest test: CI has no Chrome, no camera and no corpus, and the numbers
 * that matter here depend on how fast the machine is. It asserts only what must hold on any
 * device, and reports the rest for a human to read.
 *
 * Usage and setup: see README.md in this directory.
 */
import { readFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import process from 'node:process';
import puppeteer from 'puppeteer-core';

const HERE = import.meta.dirname;
const ROOT = join(HERE, '..', '..');

const BASE = process.env.BASE ?? 'http://localhost:4199/esku/';
const CHROME = process.env.CHROME ?? join(HERE, 'chrome-linux64', 'chrome');
const SECONDS = Number(process.env.SECONDS ?? 20);
const PLAYBACK = Number(process.env.PLAYBACK ?? 1);

/** MediaPipe logs this at info level on every start; it is not an error. */
const BENIGN = /XNNPACK delegate|Created TensorFlow Lite/;

/**
 * The shipped thresholds, read from the source rather than copied.
 *
 * Duplicating them here would let this tool quietly disagree with the app it is measuring,
 * which is the exact class of bug it was built to catch. A miss throws instead of guessing.
 */
function shippedFloor() {
  const source = readFileSync(
    join(ROOT, 'src', 'domain', 'recognition', 'services', 'SignSegmenter.ts'),
    'utf-8',
  );
  const read = (key) => {
    const found = source.match(new RegExp(`^\\s+${key}:\\s*([0-9.]+),`, 'm'));
    if (!found) throw new Error(`cannot read ${key} from SignSegmenter.ts — did it get renamed?`);
    return Number(found[1]);
  };
  const minSignMs = read('minSignMs');
  const minFrames = read('minFrames');
  // A window may only close once it has run minSignMs, and is then thrown away unless it
  // holds minFrames samples. Below this rate the two rules cannot both be satisfied.
  return { minSignMs, minFrames, requiredFps: minFrames / (minSignMs / 1000) };
}

/**
 * Replaces the camera with a video, before any app code runs.
 *
 * `captureStream()` on a playing <video> yields a MediaStream indistinguishable from a camera
 * to getUserMedia's callers — no fake-device flags, no y4m conversion, no file serving. The
 * clip travels as a data URL so it is same-origin by construction. Decode state is reported
 * back, because a missing codec must not be able to masquerade as "recognised nothing".
 */
function fakeCamera({ dataUrl, rate }) {
  const shim = async () => {
    const video = document.createElement('video');
    video.src = dataUrl;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.playbackRate = rate;
    await new Promise((resolve, reject) => {
      video.addEventListener('loadeddata', resolve, { once: true });
      video.addEventListener('error', () => reject(new Error('clip failed to decode')), {
        once: true,
      });
      setTimeout(() => reject(new Error('clip timed out loading')), 20000);
    });
    await video.play();
    window.__feed = { width: video.videoWidth, height: video.videoHeight };
    return video.captureStream();
  };
  navigator.mediaDevices.getUserMedia = shim;
  navigator.getUserMedia = (_constraints, ok, fail) => shim().then(ok, fail);
}

/** Reads the panel as label/value pairs, which survives wording changes better than text. */
function readPanel() {
  const rows = {};
  for (const row of document.querySelectorAll('#diag-body .diagnostics__row')) {
    const label = row.querySelector('dt')?.textContent?.trim();
    const value = row.querySelector('dd')?.textContent?.trim();
    if (label) rows[label] = value ?? '';
  }
  return {
    rows,
    transcript: document.querySelector('#transcript')?.textContent?.trim() ?? '',
    status: document.querySelector('#status')?.textContent?.trim() ?? '',
    feed: window.__feed ?? null,
  };
}

const firstNumber = (text) => Number(text?.match(/-?\d+(\.\d+)?/)?.[0] ?? Number.NaN);
const nthNumber = (text, n) => Number(text?.match(/-?\d+(\.\d+)?/g)?.[n] ?? Number.NaN);

async function measure(browser, clip, floor) {
  const page = await browser.newPage();
  const problems = [];
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error' && !BENIGN.test(message.text())) {
      problems.push(`console: ${message.text().slice(0, 200)}`);
    }
  });
  page.on('requestfailed', (request) =>
    problems.push(`request failed: ${request.url()} (${request.failure()?.errorText})`),
  );

  const bytes = readFileSync(clip).toString('base64');
  await page.evaluateOnNewDocument(fakeCamera, {
    dataUrl: `data:video/mp4;base64,${bytes}`,
    rate: PLAYBACK,
  });
  await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 180000 });

  // The app not booting at all is the loudest regression this can catch — a wrong base path
  // 404s the bundle, and the SPA fallback answers 200 for it, so the page looks served and is
  // empty. Waiting for the control gives that a clear verdict instead of a puppeteer stack.
  const booted = await page
    .waitForSelector('#toggle', { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  if (!booted) {
    problems.push('the app never mounted: #toggle absent (wrong BASE? bundle 404?)');
    await page.close();
    return { word: basename(clip, '.mp4'), problems, booted: false, framesWithHands: 0 };
  }

  await page.click('#toggle');

  const startedAt = Date.now();
  await new Promise((resolve) => setTimeout(resolve, SECONDS * 1000));
  const elapsedS = (Date.now() - startedAt) / 1000;

  await page.click('#diag-toggle');
  const report = await page.evaluate(readPanel);
  await page.close();

  const framesSeen = firstNumber(report.rows.Fotogramas);
  return {
    word: basename(clip, '.mp4'),
    problems,
    feed: report.feed,
    status: report.status,
    transcript: report.transcript,
    framesSeen,
    framesWithHands: nthNumber(report.rows.Fotogramas, 1),
    fps: framesSeen / elapsedS,
    windowsClosed: firstNumber(report.rows['Ventanas cerradas']),
    windowsShort: nthNumber(report.rows['Ventanas cerradas'], 1),
    engineLoaded: report.rows['Motor cargado'] === 'sí',
    invocations: firstNumber(report.rows['Veces consultado']),
    words: firstNumber(report.rows['Palabras del vocabulario']),
    vetoedBy: report.rows['Bloqueado por'],
    rawTop: report.rows['Mejores opciones, sin filtrar'],
    signature: report.rows['Lo que recibió el modelo (esperado entre paréntesis)'],
    fastEnough: framesSeen / elapsedS >= floor.requiredFps,
  };
}

const clips = process.argv.slice(2);
if (clips.length === 0) {
  console.error('usage: node harness.mjs <clip.mp4> [more.mp4 ...]   (see README.md)');
  process.exit(2);
}

const floor = shippedFloor();
console.log(
  `shipped floor: minSignMs ${floor.minSignMs} over minFrames ${floor.minFrames} ` +
    `=> a device must sustain ${floor.requiredFps.toFixed(1)} fps for any sign to survive\n`,
);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--no-sandbox',
    '--use-fake-ui-for-media-stream',
    '--autoplay-policy=no-user-gesture-required',
    // MediaPipe wants a GPU delegate. Headless here gets software WebGL — about 1.3 fps, well
    // under the floor above, so recognition cannot be asserted from this machine. Measured:
    // WSL's /dev/dxg does not help, WebGL falls back to software regardless.
    ...(process.env.GL === 'auto'
      ? []
      : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
  ],
});

const results = [];
try {
  for (const clip of clips) results.push(await measure(browser, clip, floor));
} finally {
  await browser.close();
}

for (const r of results) {
  console.log(`${'='.repeat(70)}\n${r.word}\n${'='.repeat(70)}`);
  if (r.booted === false) {
    console.log(`problems      : ${r.problems.join('\n                ')}\n`);
    continue;
  }
  console.log(`feed          : ${r.feed ? `${r.feed.width}x${r.feed.height}` : 'NO DECODE'}`);
  console.log(`frames        : ${r.framesSeen} (${r.framesWithHands} with a hand)`);
  console.log(`frame rate    : ${r.fps.toFixed(1)} fps  ${r.fastEnough ? '' : '<-- below floor'}`);
  console.log(`engine loaded : ${r.engineLoaded ? 'yes' : 'NO'}`);
  console.log(`windows       : ${r.windowsClosed} closed, ${r.windowsShort} discarded as short`);
  console.log(`engine asked  : ${r.invocations}`);
  console.log(`words written : ${r.words}   vetoed by: ${r.vetoedBy}`);
  console.log(`raw scores    : ${r.rawTop}`);
  console.log(`fed the model : ${r.signature}`);
  console.log(`transcript    : ${JSON.stringify(r.transcript)}`);
  if (r.problems.length) console.log(`problems      : ${r.problems.join('\n                ')}`);
  console.log();
}

/**
 * Only what holds on any device is a failure.
 *
 * Recognition itself is not assertable here: it needs a frame rate this machine cannot reach,
 * and pretending otherwise would either produce a permanently red check or invite someone to
 * lower a shipped threshold to make it green. What *is* assertable catches real regressions —
 * a base-path break that 404s the weights, a landmark pipeline that stops producing hands, a
 * clip that silently fails to decode.
 */
const failures = [];
for (const r of results) {
  if (r.problems.length) failures.push(`${r.word}: ${r.problems.join('; ')}`);
  if (r.booted === false) continue;
  if (!r.feed?.width) failures.push(`${r.word}: clip never decoded`);
  if (!r.engineLoaded) failures.push(`${r.word}: vocabulary weights never loaded`);
  if (!(r.framesWithHands > 0)) failures.push(`${r.word}: no frame ever had a hand in it`);
  // Only meaningful once the device is fast enough for a window to be able to survive.
  if (r.fastEnough && r.invocations === 0) {
    failures.push(
      `${r.word}: ${r.fps.toFixed(1)} fps is above the floor yet the engine was never asked`,
    );
  }
}

if (failures.length) {
  console.log('FAILED');
  for (const failure of failures) console.log(`  - ${failure}`);
  process.exit(1);
}

const slow = results.filter((r) => !r.fastEnough);
console.log('PASSED — invariants hold');
if (slow.length) {
  console.log(
    `  note: ${slow.length}/${results.length} run(s) below ${floor.requiredFps.toFixed(1)} fps, ` +
      'so recognition was measured, not asserted. Read the numbers above.',
  );
}
