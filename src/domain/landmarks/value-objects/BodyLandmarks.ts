import type { Landmark } from './Landmark';

/**
 * MediaPipe Pose indices for the parts that matter while signing.
 *
 * Legs are ignored: nothing below the hips carries meaning in LSE, and drawing them would
 * spend pixels and inference on noise.
 */
export const PosePoint = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;

/** Drawn as the torso: the shoulder line and the box down to the hips. */
export const TORSO_CONNECTIONS: readonly (readonly [number, number])[] = [
  [PosePoint.leftShoulder, PosePoint.rightShoulder],
  [PosePoint.leftShoulder, PosePoint.leftHip],
  [PosePoint.rightShoulder, PosePoint.rightHip],
  [PosePoint.leftHip, PosePoint.rightHip],
];

/** Drawn as the arms: shoulder to elbow to wrist, both sides. */
export const ARM_CONNECTIONS: readonly (readonly [number, number])[] = [
  [PosePoint.leftShoulder, PosePoint.leftElbow],
  [PosePoint.leftElbow, PosePoint.leftWrist],
  [PosePoint.rightShoulder, PosePoint.rightElbow],
  [PosePoint.rightElbow, PosePoint.rightWrist],
];

/** Neck: the head sitting on the shoulder line, drawn as its own segment. */
export const NECK_CONNECTIONS: readonly (readonly [number, number])[] = [
  [PosePoint.nose, PosePoint.leftShoulder],
  [PosePoint.nose, PosePoint.rightShoulder],
];

export interface PoseLandmarks {
  readonly points: readonly Landmark[];
}

/**
 * The face, reduced to the landmarks that carry grammar.
 *
 * Non-manual markers are not decoration in LSE: raised eyebrows mark a question, a head
 * shake negates, and mouth gestures separate minimal pairs. The full 468-point mesh is far
 * more than that needs, and far more than a model trained on 6,336 examples can use.
 */
export interface FaceLandmarks {
  readonly points: readonly Landmark[];
}
