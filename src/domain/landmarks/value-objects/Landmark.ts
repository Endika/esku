/** A single normalised landmark as produced by a pose/hand estimator. */
export interface Landmark {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export type Handedness = 'left' | 'right';

/** MediaPipe's hand model always emits exactly this many points per hand. */
export const HAND_LANDMARK_COUNT = 21;

/**
 * Indices into a hand's point list. Named because `points[8]` on its own is unreadable
 * and every geometry helper below depends on getting these right.
 */
export const HandPoint = {
  wrist: 0,
  thumbCmc: 1,
  thumbMcp: 2,
  thumbIp: 3,
  thumbTip: 4,
  indexMcp: 5,
  indexPip: 6,
  indexDip: 7,
  indexTip: 8,
  middleMcp: 9,
  middlePip: 10,
  middleDip: 11,
  middleTip: 12,
  ringMcp: 13,
  ringPip: 14,
  ringDip: 15,
  ringTip: 16,
  pinkyMcp: 17,
  pinkyPip: 18,
  pinkyDip: 19,
  pinkyTip: 20,
} as const;

export interface HandLandmarks {
  readonly handedness: Handedness;
  readonly points: readonly Landmark[];
}

export class InvalidHandLandmarksError extends Error {
  constructor(count: number) {
    super(`A hand needs exactly ${HAND_LANDMARK_COUNT} landmarks, got ${count}`);
    this.name = 'InvalidHandLandmarksError';
  }
}

export function createHandLandmarks(
  handedness: Handedness,
  points: readonly Landmark[],
): HandLandmarks {
  if (points.length !== HAND_LANDMARK_COUNT) {
    throw new InvalidHandLandmarksError(points.length);
  }
  return { handedness, points };
}

export function pointAt(hand: HandLandmarks, index: number): Landmark {
  const point = hand.points[index];
  if (!point) throw new InvalidHandLandmarksError(hand.points.length);
  return point;
}
