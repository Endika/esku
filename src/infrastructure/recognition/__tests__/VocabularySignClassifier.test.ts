import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { VocabularySignClassifier } from '../VocabularySignClassifier';

const ROOT = join(import.meta.dirname, '..', '..', '..', '..');
const MODELS = join(ROOT, 'public', 'models');

/** PyTorch's own input and output for this exact model, written by `tools/train/train.py`. */
const parity = JSON.parse(
  readFileSync(join(ROOT, 'src', 'test', 'fixtures', 'model-parity.json'), 'utf-8'),
) as { input: number[]; logits: number[] };

const originalFetch = globalThis.fetch;

/** `forward` is private; the parity check has to reach it to compare raw logits. */
type Internals = {
  forward(signature: Float32Array, manifest: unknown, tensors: unknown): Float32Array;
  manifest: unknown;
  tensors: unknown;
};

function logitsFor(engine: VocabularySignClassifier, input: number[]): Float32Array {
  const internals = engine as unknown as Internals;
  return internals.forward(Float32Array.from(input), internals.manifest, internals.tensors);
}

/** Serves the real shipped model files off disk — no stand-in weights, no stubbed maths. */
function serveModelFromDisk(): void {
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const name = String(input).split('/').pop()!;
    const bytes = readFileSync(join(MODELS, name));
    return {
      json: async () => JSON.parse(bytes.toString('utf-8')),
      arrayBuffer: async () =>
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    } as Response;
  }) as typeof fetch;
}

function classifier(): VocabularySignClassifier {
  return new VocabularySignClassifier('/models/lse-vocabulary.json', '/models/lse-vocabulary.bin');
}

describe('VocabularySignClassifier', () => {
  beforeEach(serveModelFromDisk);
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('is not ready before the weights are loaded', () => {
    expect(classifier().isReady()).toBe(false);
  });

  it('stays silent before loading rather than throwing', async () => {
    const engine = classifier();
    const frames = [buildFrame(0, buildHand({ curls: [0.9, 0, 0.9, 0.9, 0.9] }))];
    expect(await engine.classify(frames)).toEqual([]);
  });

  it('loads the shipped weights', async () => {
    const engine = classifier();
    await engine.load();
    expect(engine.isReady()).toBe(true);
    expect(engine.accuracy?.top1).toBeGreaterThan(0.5);
  });

  it('reproduces PyTorch logits for the same input', async () => {
    // The load-bearing test. A hand-written GRU that is subtly wrong — reset gate applied to
    // the wrong term, gates read in the wrong order, a transposed weight — still runs and
    // still returns plausible numbers. Only this catches that.
    const engine = classifier();
    await engine.load();

    const logits = logitsFor(engine, parity.input);

    expect(logits).toHaveLength(parity.logits.length);
    for (let i = 0; i < parity.logits.length; i += 1) {
      // float32 accumulation ordering differs between numpy and JS; 1e-3 is well inside the
      // margin that would hide a structural mistake.
      expect(logits[i]).toBeCloseTo(parity.logits[i]!, 3);
    }
  });

  it('agrees with PyTorch on which concept wins', async () => {
    const engine = classifier();
    await engine.load();
    const expected = parity.logits.indexOf(Math.max(...parity.logits));

    const logits = logitsFor(engine, parity.input);
    const actual = logits.indexOf(Math.max(...logits));

    expect(actual).toBe(expected);
  });

  it('reads windows, so the segmenter decides when to ask it', () => {
    expect(classifier().granularity).toBe('window');
  });

  it('ignores an empty window', async () => {
    const engine = classifier();
    await engine.load();
    expect(await engine.classify([])).toEqual([]);
  });

  it('tags what it recognises as vocabulary, so it reads as a word not a letter', async () => {
    const engine = classifier();
    await engine.load();
    const frames = Array.from({ length: 12 }, (_, i) =>
      buildFrame(
        i * 33,
        buildHand({ curls: [0.5, 0.2, 0.4, 0.6, 0.3], offset: { x: i * 0.01, y: 0 } }),
      ),
    );
    for (const candidate of await engine.classify(frames)) {
      expect(candidate.source).toBe('vocabulary');
    }
  });
});
