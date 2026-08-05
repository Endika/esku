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

/**
 * Drawn as the torso: the shoulder line, and the box down to the hips when the hips are
 * genuinely in shot.
 *
 * A signer is usually framed from the chest up, so the hip edges are dropped far more often
 * than they are drawn. `isVisible` decides, per edge, at render time.
 */
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

/**
 * Below this, a pose landmark is the model's guess rather than something it saw.
 *
 * Filming a signer from the chest up leaves the hips invisible, and MediaPipe answers with
 * extrapolated coordinates instead of nothing. Drawing those produces a torso box running
 * off the bottom of the picture and arms pointing at the frame edges.
 */
export const MIN_POSE_VISIBILITY = 0.6;

export function isVisible(point: Landmark | undefined): boolean {
  // Absent visibility means the estimator does not report it (hands, face) — trust those.
  return point !== undefined && (point.visibility ?? 1) >= MIN_POSE_VISIBILITY;
}

/**
 * The neck: one segment from the middle of the shoulders up to the head.
 *
 * MediaPipe has no neck landmark, so it has to be derived. Drawing nose-to-each-shoulder
 * instead — the obvious shortcut — paints a wide triangle across the chest that looks
 * nothing like a neck and hides the signing space behind it.
 */
export function neckSegment(pose: readonly Landmark[]): readonly [Landmark, Landmark] | null {
  const left = pose[PosePoint.leftShoulder];
  const right = pose[PosePoint.rightShoulder];
  const nose = pose[PosePoint.nose];
  if (!isVisible(left) || !isVisible(right) || !isVisible(nose) || !left || !right || !nose) {
    return null;
  }

  const centre: Landmark = {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2,
    z: (left.z + right.z) / 2,
  };
  // Stop short of the nose: the neck ends at the chin, and running the line into the face
  // mesh just clutters it.
  const chin: Landmark = {
    x: centre.x + (nose.x - centre.x) * 0.6,
    y: centre.y + (nose.y - centre.y) * 0.6,
    z: centre.z + (nose.z - centre.z) * 0.6,
  };
  return [centre, chin];
}

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
