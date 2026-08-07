/**
 * Measures the built app's layout at real phone widths.
 *
 * The page being three screens tall, and the action bar wrapping to three rows at 320 px,
 * were both invisible in review and obvious the moment anything measured them. jsdom cannot
 * do it — it computes no heights — so this runs the production build in the same Chrome the
 * recognition harness uses.
 *
 *   npm run build && node tools/browser/layout.mjs
 */
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const DIST = path.resolve(import.meta.dirname, '../../dist');
const CHROME = path.resolve(import.meta.dirname, 'chrome-linux64/chrome');
const PORT = 8931;
const WIDTHS = [
  [390, 844],
  [320, 640],
];

const TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.json': 'application/json',
  '.webmanifest': 'application/manifest+json',
};

/** Serves dist under /esku/, because `vite preview` ignores the production base. */
const server = createServer(async (request, response) => {
  const asked =
    new URL(request.url, 'http://localhost').pathname.replace(/^\/esku\/?/, '') || 'index.html';
  const file = path.join(DIST, asked);
  try {
    if (!file.startsWith(DIST)) throw new Error('outside dist');
    const body = await readFile(file);
    const type = TYPES[path.extname(file)] ?? 'application/octet-stream';
    response.writeHead(200, { 'content-type': type });
    response.end(body);
  } catch {
    response.writeHead(404).end('not found');
  }
});
await new Promise((resolve) => server.listen(PORT, resolve));

/** Runs in the page. Must stay self-contained: puppeteer ships it across as source. */
const measure = () => {
  const root = document.documentElement;
  const app = document.querySelector('#app');
  const height = (el) => Math.round(el.getBoundingClientRect().height);
  // Not offsetParent: Chrome renders a shut <details> body through ::details-content with
  // content-visibility:hidden, so its buttons still have an offsetParent and count as shown.
  const visible = (el) => height(el) > 0 && !el.closest('details:not([open])');

  return {
    pageHeight: root.scrollHeight,
    viewport: window.innerHeight,
    overflowX: Math.max(0, root.scrollWidth - root.clientWidth),
    barHeight: height(document.querySelector('#actions')),
    buttons: [...app.querySelectorAll('button')]
      .filter(visible)
      .map((button) => (button.getAttribute('aria-label') ?? button.textContent).trim()),
    blocks: [...app.children]
      .flatMap((el) =>
        el.id === 'shell'
          ? [...el.children].map((child) => [child.className.split(' ')[0] || child.tagName, child])
          : [[el.querySelector('.card__title')?.textContent.trim() ?? el.tagName, el]],
      )
      .map(([label, el]) => [label, height(el)]),
  };
};

/**
 * Camera mode with a filled transcript, forced. Reaching it for real needs a camera and a
 * recognised sign; what is measured here is the layout those produce, not how they arrive.
 */
const forceRunning = () => {
  document.querySelector('#app').classList.add('is-running');
  document.querySelector('#transcript').textContent = 'dolor cabeza fiebre';
  document.querySelector('#edit').hidden = false;
  // The label the button really carries while running, and the widest thing in the row.
  document.querySelector('#toggle').textContent = 'Parar';
  return new Promise((resolve) => requestAnimationFrame(resolve));
};

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox'],
});

for (const [width, height] of WIDTHS) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 2, isMobile: true });
  await page.goto(`http://localhost:${PORT}/esku/`, { waitUntil: 'networkidle2' });

  const off = await page.evaluate(measure);
  console.log(`\n=== ${width}x${height} · camera off ===`);
  console.log(`page ${off.pageHeight}px = ${(off.pageHeight / off.viewport).toFixed(1)} screens`);
  console.log(`overflow-x ${off.overflowX}px · action bar ${off.barHeight}px`);
  console.log(`buttons: ${off.buttons.length} -> ${off.buttons.join(' | ')}`);
  for (const [label, px] of off.blocks) console.log(`  ${String(px).padStart(5)}px  ${label}`);

  await page.evaluate(forceRunning);
  const on = await page.evaluate(measure);
  console.log(`--- camera on, transcript filled`);
  console.log(`overflow-x ${on.overflowX}px · action bar ${on.barHeight}px`);
  console.log(`buttons: ${on.buttons.length} -> ${on.buttons.join(' | ')}`);

  await page.close();
}

await browser.close();
server.close();
