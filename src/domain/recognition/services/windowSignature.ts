import { toFeatureVector } from '@domain/landmarks/services/normalizeHand';
import { HAND_LANDMARK_COUNT } from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

/** Frames every signature is resampled to, whatever the sign's real duration. */
export const SIGNATURE_FRAMES = 8;

const HAND_FLOATS = HAND_LANDMARK_COUNT * 3;
/** Right hand then left hand, zero-filled when a hand is absent. */
const FRAME_FLOATS = HAND_FLOATS * 2;
export const SIGNATURE_LENGTH = SIGNATURE_FRAMES * FRAME_FLOATS;

/**
 * Reduces a variable-length sign to one fixed-length vector that can be compared to another.
 *
 * Resampling to a fixed frame count is what makes a fast signing of a word and a slow one
 * land near each other: the shape of the movement survives, its duration does not. Without
 * it, two recordings of the same sign would differ mostly in how long they took.
 *
 * Both hands are encoded from the start, in a stable right-then-left order rather than by
 * which hand is dominant in a given frame — many LSE signs are two-handed, and a signature
 * that only captured one would have to change later, invalidating every sign a user had
 * already taught.
 */
export function windowSignature(window: readonly LandmarkFrame[]): Float32Array {
  const signature = new Float32Array(SIGNATURE_LENGTH);
  if (window.length === 0) return signature;

  for (let slot = 0; slot < SIGNATURE_FRAMES; slot += 1) {
    const frame = window[sampleIndex(slot, window.length)];
    if (!frame) continue;

    const right = frame.hands.find((hand) => hand.handedness === 'right');
    const left = frame.hands.find((hand) => hand.handedness === 'left');
    const base = slot * FRAME_FLOATS;

    if (right) signature.set(toFeatureVector(right), base);
    if (left) signature.set(toFeatureVector(left), base + HAND_FLOATS);
  }

  return signature;
}

/** Evenly spaced picks across the window, always including its first and last frame. */
function sampleIndex(slot: number, length: number): number {
  if (length === 1) return 0;
  return Math.round((slot / (SIGNATURE_FRAMES - 1)) * (length - 1));
}

/**
 * Typical per-coordinate distance, in palm widths, at which two signs stop being the same
 * sign. Tuned so distinct handshapes land well below the classifier's threshold while
 * repeats of one sign stay above it.
 */
const DISTANCE_SCALE = 0.18;

/**
 * How alike two signatures are, from 0 to 1.
 *
 * Root-mean-square Euclidean distance, not cosine. Cosine looks like the natural choice and
 * is badly wrong here: every hand shares the same gross structure — wrist at the origin,
 * fingers extending away — so that common component dominates the vector and the angle
 * between two *different* handshapes stays tiny. Measured on this fixture set, cosine
 * scored a fist against an open hand at 0.965 and an index point against a Y at 0.962,
 * where an exact self-match scored 1.000. There is no threshold that separates those.
 *
 * Distance works because `normalizeHand` has already removed position and scale, so what
 * remains is shape, and shape differences show up as real displacement per coordinate.
 */
export function similarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let sum = 0;
  let energyA = 0;
  let energyB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    const delta = x - y;
    sum += delta * delta;
    energyA += x * x;
    energyB += y * y;
  }

  // Two signatures with nothing tracked are at distance zero from each other, which would
  // otherwise score a perfect match — "I saw no hand" agreeing with "I saw no hand".
  if (energyA < 1e-9 || energyB < 1e-9) return 0;

  const distance = Math.sqrt(sum / a.length);
  // Gaussian decay: identical signs score 1, and similarity falls away smoothly rather than
  // cutting off, so the classifier's threshold is the only place a decision gets made.
  return Math.exp(-((distance / DISTANCE_SCALE) ** 2));
}
