import type { SignCandidate } from '@domain/recognition/value-objects/Gloss';
import { describe, expect, it } from 'vitest';
import { editOps, spell } from '../spelling';

/**
 * These two functions are the whole scoring surface of the alphabet bench, and they are where
 * a benchmark can quietly flatter itself: an edit distance that reports only its total hides
 * whether the engine is guessing wrong letters, inventing them, or missing them — three faults
 * with three different fixes. So the split is asserted piece by piece here, on strings short
 * enough to count by hand, rather than inferred from corpus output.
 */
describe('editOps', () => {
  it('counts a wrong letter as a substitution', () => {
    expect(editOps('CASA', 'CASO')).toMatchObject({
      distance: 1,
      substitutions: 1,
      insertions: 0,
      deletions: 0,
    });
  });

  it('names which letter was confused for which', () => {
    // The confusion matrix is built from this, and it is the output that says whether the
    // handshape table is wrong about a specific letter or merely noisy everywhere.
    expect(editOps('CASA', 'CASO').pairs).toEqual([['A', 'O']]);
  });

  it('counts a spare letter as an insertion', () => {
    expect(editOps('CASA', 'CASAS')).toMatchObject({
      distance: 1,
      substitutions: 0,
      insertions: 1,
      deletions: 0,
    });
  });

  it('counts a missing letter as a deletion', () => {
    expect(editOps('CASA', 'CAS')).toMatchObject({
      distance: 1,
      substitutions: 0,
      insertions: 0,
      deletions: 1,
    });
  });

  it('reads an empty answer as every letter deleted', () => {
    expect(editOps('AB', '')).toMatchObject({ distance: 2, deletions: 2, insertions: 0 });
  });

  it('reads an answer to nothing as every letter inserted', () => {
    expect(editOps('', 'AB')).toMatchObject({ distance: 2, insertions: 2, deletions: 0 });
  });

  it('splits a mixed error into the operations that actually happened', () => {
    // Cost 2, and the split is forced: the answer is one longer, so insertions minus
    // deletions must be 1, which leaves exactly one substitution. A total of 2 alone would
    // not distinguish this from two wrong letters.
    expect(editOps('ABC', 'AXCD')).toMatchObject({
      distance: 2,
      substitutions: 1,
      insertions: 1,
      deletions: 0,
    });
  });

  it('prefers substitutions when two decompositions cost the same', () => {
    // 'AB' → 'BA' is two substitutions or one insertion plus one deletion, both cost 2.
    // Pinned rather than left to chance: an unstable tie-break would move the three counts
    // between runs while the total stayed put, which is the hardest kind of drift to notice.
    expect(editOps('AB', 'BA')).toMatchObject({
      distance: 2,
      substitutions: 2,
      insertions: 0,
      deletions: 0,
    });
  });

  it('scores a perfect answer as no operations at all', () => {
    expect(editOps('CASA', 'CASA')).toMatchObject({ distance: 0, pairs: [] });
  });
});

/** A frame's winning candidate, at whatever confidence. */
function saw(letter: string, confidence = 0.9): SignCandidate {
  return {
    gloss: { id: letter, conceptId: letter, text: letter },
    confidence,
    source: 'alphabet',
  };
}

const NOTHING = { top: null, empty: true };

describe('spell', () => {
  it('writes a letter once the engine has agreed on it three frames running', () => {
    expect(spell([1, 2, 3].map(() => ({ top: saw('A'), empty: false })))).toBe('A');
  });

  it('writes nothing on two frames, which is the flicker the stabiliser exists to eat', () => {
    expect(spell([1, 2].map(() => ({ top: saw('A'), empty: false })))).toBe('');
  });

  it('does not repeat a held letter however long it is held', () => {
    expect(spell(Array.from({ length: 20 }, () => ({ top: saw('A'), empty: false })))).toBe('A');
  });

  it('lets the same letter repeat once the hand has left frame', () => {
    // How a double letter gets spelled at all. Without the release the latch would swallow
    // the second L of 'ELLA' and no amount of holding it would help.
    const held = Array.from({ length: 3 }, () => ({ top: saw('L'), empty: false }));
    expect(spell([...held, NOTHING, NOTHING, ...held])).toBe('LL');
  });

  it('ignores a candidate under the confidence floor the app ships', () => {
    expect(spell(Array.from({ length: 5 }, () => ({ top: saw('A', 0.55), empty: false })))).toBe(
      '',
    );
  });

  it('spells a whole word from a run of held letters', () => {
    const hold = (letter: string) =>
      Array.from({ length: 4 }, () => ({ top: saw(letter), empty: false }));
    expect(spell([...hold('S'), ...hold('O'), ...hold('L')])).toBe('SOL');
  });
});
