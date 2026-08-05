import { describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { SIGNATURE_LENGTH, similarity, windowSignature } from '../windowSignature';

type Curls = [number, number, number, number, number];

/** A held handshape, repeated for `frames` — a static sign. */
function hold(curls: Curls, frames: number, handedness: 'left' | 'right' = 'right') {
  return Array.from({ length: frames }, (_, i) =>
    buildFrame(i * 33, buildHand({ curls, handedness })),
  );
}

/** A handshape travelling across the frame — a dynamic sign. */
function travel(curls: Curls, frames: number, step = 0.04) {
  return Array.from({ length: frames }, (_, i) =>
    buildFrame(i * 33, buildHand({ curls, offset: { x: i * step, y: 0 } })),
  );
}

const FIST: Curls = [0.9, 0.9, 0.9, 0.9, 0.9];
const OPEN: Curls = [0, 0, 0, 0, 0];
const POINT: Curls = [0.9, 0, 0.9, 0.9, 0.9];

describe('windowSignature', () => {
  it('always produces the same length, whatever the sign lasted', () => {
    expect(windowSignature(hold(OPEN, 3))).toHaveLength(SIGNATURE_LENGTH);
    expect(windowSignature(hold(OPEN, 40))).toHaveLength(SIGNATURE_LENGTH);
  });

  it('returns a zero signature for an empty window rather than throwing', () => {
    const signature = windowSignature([]);
    expect(signature).toHaveLength(SIGNATURE_LENGTH);
    expect(signature.every((value) => value === 0)).toBe(true);
  });

  it('matches the same sign performed at different speeds', () => {
    // The whole point of resampling: duration must not be what distinguishes two signs.
    const fast = windowSignature(travel(POINT, 6));
    const slow = windowSignature(travel(POINT, 30));
    expect(similarity(fast, slow)).toBeGreaterThan(0.95);
  });

  it('separates two genuinely different handshapes by a wide margin', () => {
    // Asserting a real number, not just an ordering: an earlier cosine metric ranked these
    // "correctly" while scoring them 0.965, which no threshold could have used.
    const fist = windowSignature(hold(FIST, 12));
    const open = windowSignature(hold(OPEN, 12));
    expect(similarity(fist, open)).toBeLessThan(0.3);
  });

  it('keeps the closest distinct pair below the recognition threshold', () => {
    // A fist and an index point share four curled fingers, so they are the hardest pair on
    // this fixture set. If even they scored above 0.86 the classifier would be guessing.
    const fist = windowSignature(hold(FIST, 12));
    const point = windowSignature(hold(POINT, 12));
    expect(similarity(fist, point)).toBeLessThan(0.86);
  });

  it('still matches a re-recording of the same sign with natural variation', () => {
    const first = windowSignature(hold(POINT, 12));
    const wobbly = Array.from({ length: 14 }, (_, i) =>
      buildFrame(
        i * 33,
        buildHand({
          curls: [0.85, 0.06, 0.96, 0.84, 0.95],
          offset: { x: 0.03, y: 0.02 },
        }),
      ),
    );
    expect(similarity(first, windowSignature(wobbly))).toBeGreaterThan(0.86);
  });

  it('is unaffected by where in frame the sign happens', () => {
    const here = windowSignature(hold(POINT, 12));
    const shifted = hold(POINT, 12).map((frame) =>
      buildFrame(frame.timestampMs, buildHand({ curls: POINT, offset: { x: 0.25, y: -0.2 } })),
    );
    expect(similarity(here, windowSignature(shifted))).toBeGreaterThan(0.99);
  });

  it('keeps the two hands in separate slots', () => {
    // A right-handed sign and its left-handed twin must not collide, or a two-handed sign
    // could never be told apart from a one-handed one.
    const right = windowSignature(hold(POINT, 10, 'right'));
    const left = windowSignature(hold(POINT, 10, 'left'));
    expect(similarity(right, left)).toBeLessThan(0.99);
  });
});

describe('similarity', () => {
  it('scores a vector against itself at the top of the range', () => {
    const signature = windowSignature(hold(POINT, 10));
    expect(similarity(signature, signature)).toBeCloseTo(1, 5);
  });

  it('scores zero against a signature where nothing was tracked', () => {
    expect(similarity(new Float32Array(SIGNATURE_LENGTH), windowSignature(hold(OPEN, 5)))).toBe(0);
  });

  it('does not call two empty signatures a perfect match', () => {
    // They sit at distance zero from each other, so the raw metric would say 1.0 — "I saw
    // no hand" agreeing with "I saw no hand" is not a recognised sign.
    const empty = new Float32Array(SIGNATURE_LENGTH);
    expect(similarity(empty, new Float32Array(SIGNATURE_LENGTH))).toBe(0);
  });

  it('refuses to compare signatures of different lengths', () => {
    expect(similarity(new Float32Array(4), new Float32Array(8))).toBe(0);
  });
});
