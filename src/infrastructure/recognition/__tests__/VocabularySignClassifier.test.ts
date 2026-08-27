import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { VOCABULARY_SIGNATURE_LENGTH } from '@domain/recognition/services/vocabularySignature';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { AbstentionUndeclaredError, VocabularySignClassifier } from '../VocabularySignClassifier';

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
  describe('the abstention class', () => {
    const CONCEPTS = ['DOLOR', 'CABEZA', '__NADA__'];

    /**
     * A three-concept model whose posteriors are dictated exactly.
     *
     * Every tensor is zero except `head.3.bias`, and a GRU with zero weights pools to zeros, so
     * the logits *are* that bias: pass `Math.log(p)` and the softmax comes back as `p`. Real
     * arithmetic end to end with a chosen output — nothing about `forward` is stubbed out.
     */
    function serveModel(probabilities: readonly number[], abstentionConcept: string | null): void {
      const hidden = 1;
      const frames = 16;
      const gates = 3 * hidden;
      const width = VOCABULARY_SIGNATURE_LENGTH / frames;
      const shapes: Record<string, number[]> = {
        'norm.weight': [width],
        'norm.bias': [width],
      };
      for (const layer of [0, 1]) {
        const inputs = layer === 0 ? width : 2 * hidden;
        for (const suffix of ['', '_reverse']) {
          shapes[`gru.weight_ih_l${layer}${suffix}`] = [gates, inputs];
          shapes[`gru.weight_hh_l${layer}${suffix}`] = [gates, hidden];
          shapes[`gru.bias_ih_l${layer}${suffix}`] = [gates];
          shapes[`gru.bias_hh_l${layer}${suffix}`] = [gates];
        }
      }
      shapes['head.0.weight'] = [2 * hidden, 2 * hidden];
      shapes['head.0.bias'] = [2 * hidden];
      shapes['head.3.weight'] = [CONCEPTS.length, 2 * hidden];
      shapes['head.3.bias'] = [CONCEPTS.length];

      const order = Object.keys(shapes);
      const total = order.reduce(
        (sum, name) => sum + (shapes[name] ?? []).reduce((a, b) => a * b, 1),
        0,
      );
      const blob = new Float32Array(total);
      blob.set(
        probabilities.map((p) => Math.log(p)),
        total - CONCEPTS.length,
      );

      const manifest = {
        concepts: CONCEPTS,
        abstentionConcept,
        signatureLength: VOCABULARY_SIGNATURE_LENGTH,
        frames,
        hidden,
        layers: 2,
        order,
        shapes,
        testTop1: 0.7,
        testTop3: 0.86,
      };

      globalThis.fetch = (async (input: RequestInfo | URL) =>
        String(input).endsWith('.json')
          ? ({ json: async () => manifest } as Response)
          : ({ arrayBuffer: async () => blob.buffer } as Response)) as typeof fetch;
    }

    const oneWindow = () => [buildFrame(0, buildHand({ curls: [0.5, 0.2, 0.4, 0.6, 0.3] }))];

    it('says nothing at all when the model says nobody is signing', async () => {
      serveModel([0.03, 0.02, 0.95], '__NADA__');
      const engine = classifier();
      await engine.load();

      expect(await engine.classify(oneWindow())).toEqual([]);
      expect(engine.lastAbstained).toBe(true);
    });

    it('still says nothing when the runner-up would have cleared its floor', async () => {
      // Abstention, not filtering. Measured on held-out signers the two policies scored
      // identically — 38.1% recall, 10.9% of pause windows written — because under an
      // abstention no real concept ever cleared the floor. Identical is not interchangeable:
      // if the model answers "nobody is signing" at 0.52, writing the 0.46 runner-up
      // contradicts the answer it just gave.
      serveModel([0.46, 0.02, 0.52], '__NADA__');
      const engine = classifier();
      await engine.load();

      expect(await engine.classify(oneWindow())).toEqual([]);
    });

    it('never offers the abstention as a word, even ranked under a real sign', async () => {
      serveModel([0.6, 0.38, 0.02], '__NADA__');
      const engine = classifier();
      await engine.load();

      const candidates = await engine.classify(oneWindow());

      expect(candidates.map((candidate) => candidate.gloss.id)).toEqual(['DOLOR']);
      expect(engine.lastAbstained).toBe(false);
    });

    it('keeps the abstention in the diagnostics, where it is the whole story', async () => {
      serveModel([0.03, 0.02, 0.95], '__NADA__');
      const engine = classifier();
      await engine.load();
      await engine.classify(oneWindow());

      expect(engine.lastScores[0]?.text).toBe('sin signo');
      expect(engine.lastScores[0]?.confidence).toBeCloseTo(0.95, 2);
    });

    it('refuses a model carrying a reserved concept it does not declare', async () => {
      // This is how `__nada__` reached transcripts: the class arrived with the co-articulated
      // retrain and nothing on this side knew it was not a word.
      serveModel([0.03, 0.02, 0.95], null);

      await expect(classifier().load()).rejects.toThrow(AbstentionUndeclaredError);
    });
  });
});
