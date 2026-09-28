import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';

// jsdom has none of these; the footnote needs none of them, but the page wires them at render time.
vi.stubGlobal('matchMedia', () => ({
  matches: false,
  addEventListener: () => {},
  removeEventListener: () => {},
}));
vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    disconnect() {}
  },
);

HTMLCanvasElement.prototype.getContext = (() =>
  new Proxy({}, { get: () => () => {} })) as unknown as HTMLCanvasElement['getContext'];

const { renderApp } = await import('../App');

/** Each corpus the shipped weights were trained on, as public/models/LICENSE.md records it. */
function trainingCorpora(): { name: string; record: string }[] {
  const licence = readFileSync(
    path.resolve(import.meta.dirname, '../../../public/models/LICENSE.md'),
    'utf-8',
  );
  const rows = [...licence.matchAll(/^\| \*\*(.+?)\*\* \|.*?zenodo\.(\d+)/gm)];
  return rows.map(([, name, record]) => ({ name: name!, record: record! }));
}

describe('the footnote credits every corpus behind the shipped models', () => {
  it('links each one by its Zenodo record', () => {
    const root = document.createElement('div');
    renderApp(root);
    const links = [...root.querySelectorAll<HTMLAnchorElement>('.footnote a')].map((a) => ({
      text: a.textContent?.trim(),
      href: a.href,
    }));

    const corpora = trainingCorpora();
    expect(corpora.length).toBeGreaterThanOrEqual(3);
    for (const { name, record } of corpora) {
      expect(links).toContainEqual({ text: name, href: `https://zenodo.org/records/${record}` });
    }
  });
});
