import {
  createHandLandmarks,
  HAND_LANDMARK_COUNT,
  type Handedness,
  type HandLandmarks,
  HandPoint,
  type Landmark,
} from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

/**
 * Builds anatomically plausible hands for tests without a camera.
 *
 * Fingers are laid out as straight chains from their knuckle; `curl` bends a finger by
 * folding its tip back toward the palm, which is what the real geometry helpers measure.
 */
const FINGER_CHAINS = [
  { mcp: HandPoint.thumbCmc, joints: [HandPoint.thumbMcp, HandPoint.thumbIp, HandPoint.thumbTip] },
  { mcp: HandPoint.indexMcp, joints: [HandPoint.indexPip, HandPoint.indexDip, HandPoint.indexTip] },
  {
    mcp: HandPoint.middleMcp,
    joints: [HandPoint.middlePip, HandPoint.middleDip, HandPoint.middleTip],
  },
  { mcp: HandPoint.ringMcp, joints: [HandPoint.ringPip, HandPoint.ringDip, HandPoint.ringTip] },
  { mcp: HandPoint.pinkyMcp, joints: [HandPoint.pinkyPip, HandPoint.pinkyDip, HandPoint.pinkyTip] },
] as const;

export interface HandOptions {
  /** Per finger (thumb first), 0 = straight, 1 = folded back onto the palm. */
  readonly curls?: readonly [number, number, number, number, number];
  readonly handedness?: Handedness;
  /** Translate the whole hand, to simulate movement between frames. */
  readonly offset?: { readonly x: number; readonly y: number };
}

export function buildHand(options: HandOptions = {}): HandLandmarks {
  const curls = options.curls ?? [0, 0, 0, 0, 0];
  const offset = options.offset ?? { x: 0, y: 0 };
  const points: Landmark[] = Array.from({ length: HAND_LANDMARK_COUNT }, () => ({
    x: 0,
    y: 0,
    z: 0,
  }));

  points[HandPoint.wrist] = { x: 0.5, y: 0.9, z: 0 };

  // Knuckles spread across a palm 0.2 wide, so palmWidth() has a stable non-zero unit.
  const knuckleXs = [0.38, 0.44, 0.5, 0.56, 0.6];
  FINGER_CHAINS.forEach((chain, finger) => {
    const baseX = knuckleXs[finger]!;
    const baseY = finger === 0 ? 0.82 : 0.72;
    points[chain.mcp] = { x: baseX, y: baseY, z: 0 };

    const curl = curls[finger] ?? 0;
    chain.joints.forEach((joint, segment) => {
      const reach = 0.06 * (segment + 1);
      // A curled finger travels back down toward the wrist instead of away from it.
      const direction = 1 - 2 * curl;
      points[joint] = { x: baseX, y: baseY - reach * direction, z: 0 };
    });
  });

  const translated = points.map((point) => ({
    x: point.x + offset.x,
    y: point.y + offset.y,
    z: point.z,
  }));
  return createHandLandmarks(options.handedness ?? 'right', translated);
}

export function buildFrame(timestampMs: number, hand: HandLandmarks | null): LandmarkFrame {
  return { timestampMs, hands: hand ? [hand] : [] };
}
