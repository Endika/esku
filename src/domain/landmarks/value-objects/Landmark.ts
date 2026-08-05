/** A single normalised landmark as produced by a pose/hand estimator. */
export interface Landmark {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  /**
   * How sure the estimator is that this point is actually in frame, 0..1.
   *
   * Pose reports it; hands and face do not. It matters because MediaPipe's pose model
   * *extrapolates* landmarks it cannot see rather than omitting them — film someone from
   * the chest up and it will still hand you hip coordinates, invented, somewhere below the
   * bottom of the picture.
   */
  readonly visibility?: number | undefined;
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

/**
 * Which landmarks are joined by bone, as index pairs. This is the hand model's topology, not
 * a drawing choice — anything rendering or measuring a skeleton needs the same pairs.
 */
export const HAND_CONNECTIONS: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

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
