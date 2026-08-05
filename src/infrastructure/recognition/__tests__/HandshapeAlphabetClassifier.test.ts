import { describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { HandshapeAlphabetClassifier } from '../HandshapeAlphabetClassifier';

type Curls = [number, number, number, number, number];

const classifier = new HandshapeAlphabetClassifier();

async function readLetter(curls: Curls): Promise<string | null> {
  const hand = buildHand({ curls });
  const candidates = await classifier.classify([buildFrame(0, hand)]);
  return candidates[0]?.gloss.text ?? null;
}

async function readAll(curls: Curls): Promise<string[]> {
  const hand = buildHand({ curls });
  const candidates = await classifier.classify([buildFrame(0, hand)]);
  return candidates.map((candidate) => candidate.gloss.text);
}

describe('HandshapeAlphabetClassifier', () => {
  it('reads L from an extended thumb and index', async () => {
    expect(await readLetter([0, 0, 0.9, 0.9, 0.9])).toBe('l');
  });

  it('reads Y from an extended thumb and pinky', async () => {
    expect(await readLetter([0, 0.9, 0.9, 0.9, 0])).toBe('y');
  });

  it('reads I from an extended pinky alone', async () => {
    expect(await readLetter([0.9, 0.9, 0.9, 0.9, 0])).toBe('i');
  });

  it('reads W from three extended fingers', async () => {
    expect(await readLetter([0.9, 0, 0, 0, 0.9])).toBe('w');
  });

  it('reads B from four extended fingers with the thumb folded in', async () => {
    expect(await readLetter([0.9, 0, 0, 0, 0])).toBe('b');
  });

  it('reads D from an extended index with the thumb held against the middle finger', async () => {
    expect(await readLetter([0.4, 0, 0.85, 0.85, 0.85])).toBe('d');
  });

  it('declines rather than guess when a defining feature is wrong', async () => {
    // Index out but thumb clenched into a fist: neither D (thumb against middle) nor L
    // (thumb extended). Averaging would have picked L on its three matching curled fingers.
    expect(await readAll([0.9, 0, 0.9, 0.9, 0.9])).toEqual([]);
  });

  it('declines on a plain fist, where A, E and S are genuinely ambiguous', async () => {
    expect(await readAll([0.9, 0.95, 0.95, 0.95, 0.95])).toEqual([]);
  });

  it('returns nothing when no hand is visible', async () => {
    expect(await classifier.classify([buildFrame(0, null)])).toEqual([]);
  });

  it('returns nothing for an empty window', async () => {
    expect(await classifier.classify([])).toEqual([]);
  });

  it('reads the same letter from a left hand', async () => {
    // The features are mirror-invariant, so handedness must not change the answer.
    const left = buildHand({ curls: [0, 0.9, 0.9, 0.9, 0], handedness: 'left' });
    const candidates = await classifier.classify([buildFrame(0, left)]);
    expect(candidates[0]?.gloss.text).toBe('y');
  });

  it('is unaffected by where the hand sits in frame', async () => {
    const moved = buildHand({ curls: [0, 0, 0.9, 0.9, 0.9], offset: { x: 0.2, y: -0.3 } });
    const candidates = await classifier.classify([buildFrame(0, moved)]);
    expect(candidates[0]?.gloss.text).toBe('l');
  });

  it('never offers more than three candidates', async () => {
    const letters = await readAll([0, 0, 0.9, 0.9, 0.9]);
    expect(letters.length).toBeLessThanOrEqual(3);
  });

  it('tags what it reads as alphabet, so the transcript spells rather than spaces', async () => {
    const hand = buildHand({ curls: [0, 0, 0.9, 0.9, 0.9] });
    const [top] = await classifier.classify([buildFrame(0, hand)]);
    expect(top?.source).toBe('alphabet');
  });
});
