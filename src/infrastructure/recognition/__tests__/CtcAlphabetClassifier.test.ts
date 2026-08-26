import { beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import type { AlphabetManifest } from '../CtcAlphabetClassifier';
import { CtcAlphabetClassifier } from '../CtcAlphabetClassifier';

/**
 * The difference that matters against the handshape table: this engine is an automaton, not a
 * pure function of the last frame. A GRU carries hidden state, so the same hand can mean
 * different things depending on what came before — which is the whole reason it can tell a
 * held letter from a transition, and the whole reason it must be reset when the hand leaves.
 */

const HIDDEN = 4;
const INPUTS = 63;
const LETTERS = ['a', 'b', 'c'];
const CLASSES = LETTERS.length + 1;

function fill(count: number, seed: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = Math.sin((i + 1) * seed) * 0.4;
  return out;
}

const manifest: AlphabetManifest = {
  hidden: HIDDEN,
  layers: 1,
  inputs: INPUTS,
  classes: CLASSES,
  blank: 0,
  letters: LETTERS,
  order: [
    'gru.weight_ih_l0',
    'gru.weight_hh_l0',
    'gru.bias_ih_l0',
    'gru.bias_hh_l0',
    'head.weight',
    'head.bias',
  ],
  shapes: {
    'gru.weight_ih_l0': [3 * HIDDEN, INPUTS],
    'gru.weight_hh_l0': [3 * HIDDEN, HIDDEN],
    'gru.bias_ih_l0': [3 * HIDDEN],
    'gru.bias_hh_l0': [3 * HIDDEN],
    'head.weight': [CLASSES, HIDDEN],
    'head.bias': [CLASSES],
  },
};

function weights(): ArrayBuffer {
  const total = manifest.order.reduce(
    (sum, name) => sum + manifest.shapes[name]!.reduce((a, b) => a * b, 1),
    0,
  );
  return fill(total, 1.7).buffer as ArrayBuffer;
}

describe('CtcAlphabetClassifier', () => {
  let classifier: CtcAlphabetClassifier;

  beforeEach(async () => {
    classifier = new CtcAlphabetClassifier();
    await classifier.loadFrom(manifest, weights());
  });

  const hand = () => buildFrame(0, buildHand({ curls: [0, 0, 0.9, 0.9, 0.9] }));

  it('is not ready until its weights are loaded', () => {
    expect(new CtcAlphabetClassifier().isReady()).toBe(false);
    expect(classifier.isReady()).toBe(true);
  });

  it('answers nothing when there is no hand, and does not advance state', async () => {
    // A frame with no hand is not a shape to classify. Feeding zeros through the GRU instead
    // would move the state on nothing, which is how a pause would come to mean something.
    expect(await classifier.classify([buildFrame(0, null)])).toEqual([]);
    expect(classifier.lastScores).toEqual([]);
  });

  it('carries state between frames, so the same hand twice answers differently', async () => {
    const first = await classifier.classify([hand()]);
    const second = await classifier.classify([hand()]);

    expect(first[0]?.confidence).not.toBeCloseTo(second[0]?.confidence ?? -1, 12);
  });

  it('returns to the start when reset, which a new session and a lost hand both need', async () => {
    const first = await classifier.classify([hand()]);
    await classifier.classify([hand()]);
    classifier.reset();
    const afterReset = await classifier.classify([hand()]);

    expect(afterReset[0]?.gloss.text).toBe(first[0]?.gloss.text);
    expect(afterReset[0]?.confidence).toBeCloseTo(first[0]?.confidence ?? -1, 12);
  });

  it('exposes every class through lastScores, blank included', async () => {
    // The window into the floor that the table engine only grew once someone tried to measure
    // it. Blank is in here on purpose: "the model abstained" and "the model was unsure" are
    // different diagnoses.
    await classifier.classify([hand()]);

    expect(classifier.lastScores).toHaveLength(CLASSES);
    expect(classifier.lastScores.map((s) => s.text)).toContain('blank');
  });

  it('never offers blank as a candidate, because blank is not a letter', async () => {
    await classifier.classify([hand()]);
    const candidates = await classifier.classify([hand()]);

    expect(candidates.map((c) => c.gloss.text)).not.toContain('blank');
  });

  it('reads probabilities that sum to one across the classes', async () => {
    await classifier.classify([hand()]);
    const total = classifier.lastScores.reduce((sum, s) => sum + s.confidence, 0);

    expect(total).toBeCloseTo(1, 6);
  });
});
