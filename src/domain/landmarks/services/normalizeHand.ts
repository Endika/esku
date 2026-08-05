import {
  HAND_LANDMARK_COUNT,
  type HandLandmarks,
  HandPoint,
  type Landmark,
  pointAt,
} from '../value-objects/Landmark';
import { palmWidth } from './handShape';

/**
 * Recentre a hand on its wrist and rescale it to unit palm width, mirroring left hands so
 * both handednesses land in the same space.
 *
 * This is the exact transform the training pipeline applies to SWL-LSE landmarks, and it
 * has to stay in lockstep: a model trained on normalised coordinates and fed raw ones
 * silently predicts noise rather than failing loudly. See `tools/train/README.md`.
 */
export function normalizeHand(hand: HandLandmarks): readonly Landmark[] {
  const wrist = pointAt(hand, HandPoint.wrist);
  const scale = palmWidth(hand);
  const mirror = hand.handedness === 'left' ? -1 : 1;

  return hand.points.map((point) => ({
    x: ((point.x - wrist.x) / scale) * mirror,
    y: (point.y - wrist.y) / scale,
    z: (point.z - wrist.z) / scale,
  }));
}

/** Floats `toFeatureVector` emits: 21 normalised points, then the wrist's raw position. */
export const HAND_FEATURE_LENGTH = HAND_LANDMARK_COUNT * 3 + 3;

/**
 * Flatten to the vector a model consumes. Order must match `tools/train/features.py`.
 *
 * The wrist's un-normalised position is appended because normalisation deliberately throws
 * position away. That is right for a fingerspelled letter — an A is an A anywhere in frame —
 * and wrong for a sign: in LSE the same handshape at the forehead and at the chest are
 * different words.
 */
export function toFeatureVector(hand: HandLandmarks): Float32Array {
  const normalized = normalizeHand(hand);
  const out = new Float32Array(HAND_FEATURE_LENGTH);
  normalized.forEach((point, i) => {
    out[i * 3] = point.x;
    out[i * 3 + 1] = point.y;
    out[i * 3 + 2] = point.z;
  });

  const wrist = pointAt(hand, HandPoint.wrist);
  out[HAND_LANDMARK_COUNT * 3] = wrist.x;
  out[HAND_LANDMARK_COUNT * 3 + 1] = wrist.y;
  out[HAND_LANDMARK_COUNT * 3 + 2] = wrist.z;
  return out;
}
