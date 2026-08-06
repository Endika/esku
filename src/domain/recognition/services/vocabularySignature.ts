import { PosePoint } from '@domain/landmarks/value-objects/BodyLandmarks';
import { HandPoint, type Landmark } from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type {
  SignatureBlock,
  SignatureProfile,
} from '@domain/recognition/value-objects/RecognitionDiagnostics';

/**
 * What the trained LSE vocabulary model reads.
 *
 * MUST stay byte-for-byte equivalent to `tools/train/vocabulary_features.py`. A model
 * trained on one layout and fed another does not fail — it predicts noise, quietly.
 *
 * Deliberately separate from `windowSignature`, which describes a *taught* sign. The two
 * used to be one function, and every improvement to this model invalidated every sign the
 * user had recorded. They ship on different clocks, so they get different code.
 *
 * Each element earned its place on the held-out test split: hand position relative to the
 * torso (+5.7 top-1), sixteen frames instead of eight (+3.0), torso and head orientation
 * (+1.0), facial expression (+1.2). Motion deltas, raw face coordinates and input
 * augmentation were all measured and all made it worse.
 */
export const VOCABULARY_FRAMES = 16;

const HAND_FLOATS = 21 * 3 + 3 + 3;
const TORSO_FLOATS = 5;
const FACE_FLOATS = 6;
const FRAME_FLOATS = HAND_FLOATS * 2 + TORSO_FLOATS + FACE_FLOATS;
export const VOCABULARY_SIGNATURE_LENGTH = VOCABULARY_FRAMES * FRAME_FLOATS;

/** Where each part lives inside one frame's slice, for reading a built signature back. */
const BLOCKS = {
  rightHand: [0, HAND_FLOATS],
  leftHand: [HAND_FLOATS, HAND_FLOATS * 2],
  torso: [HAND_FLOATS * 2, HAND_FLOATS * 2 + TORSO_FLOATS],
  face: [HAND_FLOATS * 2 + TORSO_FLOATS, FRAME_FLOATS],
} as const;

/**
 * Face Mesh indices for the landmarks that carry grammar while signing. MediaPipe's
 * FaceLandmarker returns 478 points; 0–467 are the same mesh these indices refer to.
 */
const FACE = {
  browLeftInner: 107,
  browRightInner: 336,
  eyeLeftUpper: 159,
  eyeLeftLower: 145,
  eyeRightUpper: 386,
  eyeRightLower: 374,
  mouthLeft: 61,
  mouthRight: 291,
  lipUpper: 13,
  lipLower: 14,
  cheekLeft: 234,
  cheekRight: 454,
} as const;

export function vocabularySignature(window: readonly LandmarkFrame[]): Float32Array {
  const signature = new Float32Array(VOCABULARY_SIGNATURE_LENGTH);
  if (window.length === 0) return signature;

  for (let slot = 0; slot < VOCABULARY_FRAMES; slot += 1) {
    const frame = window[sampleIndex(slot, window.length)];
    if (!frame) continue;

    const pose = frame.pose?.points ?? [];
    const face = frame.face?.points ?? [];
    const right = frame.hands.find((hand) => hand.handedness === 'right');
    const left = frame.hands.find((hand) => hand.handedness === 'left');

    let offset = slot * FRAME_FLOATS;
    signature.set(handBlock(right?.points, 'right', pose), offset);
    offset += HAND_FLOATS;
    signature.set(handBlock(left?.points, 'left', pose), offset);
    offset += HAND_FLOATS;
    signature.set(torsoBlock(pose), offset);
    offset += TORSO_FLOATS;
    signature.set(expression(face), offset);
  }

  return signature;
}

/**
 * Reads a built signature back as four per-part summaries.
 *
 * The model scores near-noise in the browser while measuring 0.741 offline, and the feature
 * code has verified parity with the trainer — so what differs is the input, not the maths.
 * Comparing these four numbers against the same statistics over SWL-LSE's test split says
 * which part is wrong without guessing: a part that is empty here and never empty in
 * training, or an order-of-magnitude gap, is the answer.
 */
export function profileSignature(signature: Float32Array): SignatureProfile {
  const read = (from: number, to: number): SignatureBlock => {
    let empty = 0;
    let total = 0;
    let count = 0;
    for (let slot = 0; slot < VOCABULARY_FRAMES; slot += 1) {
      const base = slot * FRAME_FLOATS;
      let magnitude = 0;
      for (let i = from; i < to; i += 1) magnitude += Math.abs(signature[base + i] ?? 0);
      if (magnitude === 0) empty += 1;
      else {
        total += magnitude / (to - from);
        count += 1;
      }
    }
    return {
      emptyFrames: empty / VOCABULARY_FRAMES,
      meanMagnitude: count ? total / count : 0,
    };
  };

  return {
    rightHand: read(...BLOCKS.rightHand),
    leftHand: read(...BLOCKS.leftHand),
    torso: read(...BLOCKS.torso),
    face: read(...BLOCKS.face),
  };
}

function sampleIndex(slot: number, length: number): number {
  if (length === 1) return 0;
  return Math.round((slot / (VOCABULARY_FRAMES - 1)) * (length - 1));
}

/**
 * Origin and scale taken from the torso, not the image.
 *
 * The whole point of using pose: a wrist at chin height must read the same whether the
 * signer is close to the camera or across the room.
 */
function bodyFrame(pose: readonly Landmark[]): { centre: Landmark; width: number } | null {
  const left = pose[PosePoint.leftShoulder];
  const right = pose[PosePoint.rightShoulder];
  if (!left || !right) return null;

  const centre = {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2,
    z: (left.z + right.z) / 2,
  };
  const width = distance(left, right);
  return { centre, width: width > 1e-6 ? width : 1e-6 };
}

function handBlock(
  points: readonly Landmark[] | undefined,
  side: 'left' | 'right',
  pose: readonly Landmark[],
): Float32Array {
  const out = new Float32Array(HAND_FLOATS);
  const body = bodyFrame(pose);
  const wrist = points?.[HandPoint.wrist];
  if (!points || !wrist || !body) return out;

  const indexMcp = points[HandPoint.indexMcp];
  const pinkyMcp = points[HandPoint.pinkyMcp];
  if (!indexMcp || !pinkyMcp) return out;

  const palm = Math.max(distance(indexMcp, pinkyMcp), 1e-6);
  const mirror = side === 'left' ? -1 : 1;

  points.forEach((point, i) => {
    out[i * 3] = ((point.x - wrist.x) / palm) * mirror;
    out[i * 3 + 1] = (point.y - wrist.y) / palm;
    out[i * 3 + 2] = (point.z - wrist.z) / palm;
  });

  out[63] = (wrist.x - body.centre.x) / body.width;
  out[64] = (wrist.y - body.centre.y) / body.width;
  out[65] = (wrist.z - body.centre.z) / body.width;
  out[66] = wrist.x;
  out[67] = wrist.y;
  out[68] = wrist.z;
  return out;
}

function torsoBlock(pose: readonly Landmark[]): Float32Array {
  const out = new Float32Array(TORSO_FLOATS);
  const body = bodyFrame(pose);
  const left = pose[PosePoint.leftShoulder];
  const right = pose[PosePoint.rightShoulder];
  const nose = pose[PosePoint.nose];
  if (!body || !left || !right || !nose) return out;

  out[0] = Math.atan2(left.y - right.y, left.x - right.x);
  out[1] = (left.z - right.z) / body.width;
  out[2] = (nose.x - body.centre.x) / body.width;
  out[3] = (nose.y - body.centre.y) / body.width;
  out[4] = (nose.z - body.centre.z) / body.width;
  return out;
}

/**
 * Non-manual markers, as six ratios rather than sixty coordinates.
 *
 * Raised eyebrows mark a question in LSE, a head shake negates, and mouth gestures separate
 * minimal pairs. Handing the model the ratios directly beat giving it the raw face points by
 * four points of top-1 — with only 6,336 training examples, coordinates it would have to
 * derive these from are capacity spent memorising faces.
 */
function expression(face: readonly Landmark[]): Float32Array {
  const out = new Float32Array(FACE_FLOATS);
  const cheekLeft = face[FACE.cheekLeft];
  const cheekRight = face[FACE.cheekRight];
  if (!cheekLeft || !cheekRight) return out;

  const span = Math.max(distance(cheekLeft, cheekRight), 1e-6);
  const gap = (a: number, b: number) => {
    const first = face[a];
    const second = face[b];
    return first && second ? distance(first, second) / span : 0;
  };

  out[0] = gap(FACE.browLeftInner, FACE.eyeLeftUpper);
  out[1] = gap(FACE.browRightInner, FACE.eyeRightUpper);
  out[2] = gap(FACE.eyeLeftUpper, FACE.eyeLeftLower);
  out[3] = gap(FACE.eyeRightUpper, FACE.eyeRightLower);
  out[4] = gap(FACE.lipUpper, FACE.lipLower);
  out[5] = gap(FACE.mouthLeft, FACE.mouthRight);
  return out;
}

function distance(a: Landmark, b: Landmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
