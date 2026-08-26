import { describe, expect, it } from 'vitest';
import { type GruDirection, gruPass, gruStep, matrix } from '../gru';

/**
 * The fingerspelling engine reads one live frame at a time and cannot wait for a sequence to
 * end, so it drives the GRU a step at a time and carries the hidden state itself. That only
 * stays honest if stepping and passing are the *same* arithmetic — a second GRU written
 * alongside the first would drift apart the moment either was touched, which is the porting
 * failure this repository has already made twice.
 */

const HIDDEN = 4;
const INPUTS = 3;

/** Deterministic pseudo-weights: readable, and nothing here depends on their values. */
function fill(count: number, seed: number): Float32Array {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = Math.sin((i + 1) * seed) * 0.5;
  return out;
}

const direction: GruDirection = {
  weightIh: matrix(3 * HIDDEN, INPUTS, fill(3 * HIDDEN * INPUTS, 1.1)),
  weightHh: matrix(3 * HIDDEN, HIDDEN, fill(3 * HIDDEN * HIDDEN, 2.3)),
  biasIh: fill(3 * HIDDEN, 0.7),
  biasHh: fill(3 * HIDDEN, 1.9),
};

const sequence = [fill(INPUTS, 3.1), fill(INPUTS, 4.7), fill(INPUTS, 5.3), fill(INPUTS, 6.1)];

describe('gruStep', () => {
  it('reproduces gruPass exactly when chained over the same sequence', () => {
    const passed = gruPass(sequence, direction, HIDDEN, false);

    let state = new Float32Array(HIDDEN);
    const stepped = sequence.map((frame) => {
      state = gruStep(frame, state, direction, HIDDEN);
      return state;
    });

    expect(stepped).toHaveLength(passed.length);
    stepped.forEach((out, step) => {
      out.forEach((value, i) => {
        expect(value).toBeCloseTo(passed[step]![i]!, 10);
      });
    });
  });

  it('carries state, so the same frame twice does not give the same output', () => {
    // What makes the streaming engine an automaton rather than a pure function, and the
    // reason it needs an explicit reset when the hand leaves frame.
    const first = gruStep(sequence[0]!, new Float32Array(HIDDEN), direction, HIDDEN);
    const second = gruStep(sequence[0]!, first, direction, HIDDEN);

    expect([...second]).not.toEqual([...first]);
  });

  it('starts from zero state the way gruPass does', () => {
    const [first] = gruPass([sequence[0]!], direction, HIDDEN, false);
    const stepped = gruStep(sequence[0]!, new Float32Array(HIDDEN), direction, HIDDEN);

    stepped.forEach((value, i) => {
      expect(value).toBeCloseTo(first![i]!, 10);
    });
  });
});
