import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import fixture from '@/test/fixtures/alphabet-parity.json';
import { buildFrame, buildHand } from '@/test/handFixtures';
import type { AlphabetManifest } from '../CtcAlphabetClassifier';
import { CtcAlphabetClassifier } from '../CtcAlphabetClassifier';

/**
 * The trainer and the app each build the frame vector and run the GRU independently, and a
 * disagreement between them does not throw — it predicts noise. This drives the shipped blob
 * through the TypeScript engine and demands the numbers PyTorch produced from that same blob,
 * which pins the tensor order in the file as well as the arithmetic on top of it.
 *
 * Six frames, not one: a single frame would pass even if the engine dropped its hidden state,
 * and the hidden state is the whole reason this engine exists.
 *
 * Regenerate with `tools/train/make_alphabet_parity.py` whenever the model is retrained.
 */

const MODELS = path.resolve(import.meta.dirname, '../../../../public/models');

function shipped(): { manifest: AlphabetManifest; blob: ArrayBuffer } {
  const manifest = JSON.parse(
    readFileSync(path.join(MODELS, 'lse-alphabet.json'), 'utf-8'),
  ) as AlphabetManifest;
  const bytes = readFileSync(path.join(MODELS, 'lse-alphabet.bin'));
  const blob = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  return { manifest, blob };
}

describe('alphabet model parity', () => {
  it('matches the posteriors PyTorch reads from the same weights', async () => {
    const { manifest, blob } = shipped();
    const classifier = new CtcAlphabetClassifier();
    await classifier.loadFrom(manifest, blob);

    for (const [step, frame] of fixture.frames.entries()) {
      const hand = buildHand({
        curls: frame.curls as [number, number, number, number, number],
        offset: { x: frame.offset[0]!, y: frame.offset[1]! },
      });
      await classifier.classify([buildFrame(step * 40, hand)]);

      // `lastScores` is sorted by confidence; the fixture is in class order, so index by name.
      const byName = new Map(classifier.lastScores.map((s) => [s.text, s.confidence]));
      const expected = fixture.posteriors[step]!;
      expected.forEach((probability, klass) => {
        const name = klass === 0 ? 'blank' : fixture.letters[klass - 1]!;
        expect(byName.get(name), `frame ${step}, clase ${name}`).toBeCloseTo(probability, 5);
      });
    }
  });

  it('declares as many classes as the fixture carries', () => {
    const { manifest } = shipped();

    expect(manifest.classes).toBe(fixture.posteriors[0]!.length);
    expect(manifest.letters).toEqual(fixture.letters);
  });
});
