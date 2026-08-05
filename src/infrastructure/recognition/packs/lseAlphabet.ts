/**
 * LSE fingerspelling handshapes, described in the invariant features of `handShape`.
 *
 * Only the letters that are *separable by handshape alone* are here. Deliberately missing:
 *
 * - **J, Z, Ñ** trace a path through the air — a static frame cannot express them.
 * - **G, H, P, Q** depend on palm orientation, which these features intentionally discard
 *   so that distance and handedness stop mattering.
 * - **M, N, T** differ only in how many fingers fold over the thumb, a distinction
 *   MediaPipe's landmarks do not resolve reliably when the thumb is occluded.
 * - **R** crosses index over middle; crossing is invisible in per-finger curl.
 *
 * Those letters are the trained model's job, or the user's via a taught sign. Guessing them
 * here would be worse than declining: a confident wrong letter is harder to notice and undo
 * than a missing one.
 */
export interface HandshapeTemplate {
  readonly letter: string;
  /** Target curl per finger (thumb, index, middle, ring, pinky); null = irrelevant. */
  readonly curl: readonly (number | null)[];
  /** Target gaps between adjacent fingertips (index-middle, middle-ring, ring-pinky). */
  readonly spread?: readonly (number | null)[];
  readonly thumbIndexGap?: number;
  readonly thumbAcrossPalm?: number;
}

export const LSE_ALPHABET: readonly HandshapeTemplate[] = [
  { letter: 'a', curl: [0.15, 0.9, 0.9, 0.9, 0.9], thumbAcrossPalm: 1.3 },
  { letter: 'b', curl: [0.8, 0.05, 0.05, 0.05, 0.05], spread: [0.2, 0.2, 0.2] },
  { letter: 'c', curl: [0.4, 0.45, 0.45, 0.45, 0.45], thumbIndexGap: 0.9 },
  { letter: 'd', curl: [0.4, 0.05, 0.85, 0.85, 0.85] },
  { letter: 'e', curl: [0.75, 0.8, 0.8, 0.8, 0.8], thumbAcrossPalm: 0.7 },
  { letter: 'f', curl: [0.45, 0.5, 0.05, 0.05, 0.05], thumbIndexGap: 0.2 },
  { letter: 'i', curl: [0.7, 0.9, 0.9, 0.9, 0.05] },
  { letter: 'l', curl: [0.05, 0.05, 0.9, 0.9, 0.9] },
  { letter: 'o', curl: [0.5, 0.55, 0.55, 0.55, 0.55], thumbIndexGap: 0.15 },
  { letter: 's', curl: [0.6, 0.88, 0.88, 0.88, 0.88], thumbAcrossPalm: 0.55 },
  { letter: 'u', curl: [0.7, 0.05, 0.05, 0.9, 0.9], spread: [0.15, null, null] },
  { letter: 'v', curl: [0.7, 0.05, 0.05, 0.9, 0.9], spread: [0.75, null, null] },
  { letter: 'w', curl: [0.7, 0.05, 0.05, 0.05, 0.9] },
  { letter: 'y', curl: [0.05, 0.9, 0.9, 0.9, 0.05] },
];

/** Letters this engine will never emit, surfaced in the UI so the gap is visible. */
export const UNSUPPORTED_LETTERS = ['g', 'h', 'j', 'm', 'n', 'ñ', 'p', 'q', 'r', 't', 'x', 'z'];
