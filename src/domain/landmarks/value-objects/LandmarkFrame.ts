import type { FaceLandmarks, PoseLandmarks } from './BodyLandmarks';
import type { HandLandmarks } from './Landmark';

/**
 * One instant of tracking: whatever was visible, plus when it was seen.
 *
 * Body and face are optional because the app can run on hands alone — that is what the
 * fingerspelling engine needs, and it is what remains if the pose model has not loaded.
 */
export interface LandmarkFrame {
  readonly timestampMs: number;
  readonly hands: readonly HandLandmarks[];
  readonly pose?: PoseLandmarks | undefined;
  readonly face?: FaceLandmarks | undefined;
}

export function isEmpty(frame: LandmarkFrame): boolean {
  return frame.hands.length === 0;
}

/**
 * The hand a one-handed sign should be read from. LSE is predominantly right-dominant but
 * left-handed signers mirror everything, so "dominant" means "the one we can see", and on
 * two hands we take the one whose landmarks span the larger area — that is the one closer
 * to the camera and therefore the better-resolved.
 */
export function dominantHand(frame: LandmarkFrame): HandLandmarks | null {
  if (frame.hands.length === 0) return null;
  return frame.hands.reduce((best, hand) => (spread(hand) > spread(best) ? hand : best));
}

function spread(hand: { readonly points: readonly { x: number; y: number }[] }): number {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const p of hand.points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return (maxX - minX) * (maxY - minY);
}
