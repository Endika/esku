import { type HandLandmarks, HandPoint, type Landmark, pointAt } from '../value-objects/Landmark';

/**
 * Orientation- and size-invariant description of a handshape.
 *
 * Everything here is a ratio or an angle, never a raw coordinate, so the same sign scores
 * the same whether the hand is near or far, left or right, high or low in frame. That is
 * what lets a handshape table be written once instead of per camera distance.
 */
export interface HandShape {
  /** Per finger, 0 = fully straight, 1 = fully curled. Thumb first, then index → pinky. */
  readonly curl: readonly [number, number, number, number, number];
  /** Gap between adjacent extended fingertips, in palm widths. index-middle first. */
  readonly spread: readonly [number, number, number];
  /** Distance from thumb tip to index tip, in palm widths. Small means a pinch. */
  readonly thumbIndexGap: number;
  /** Distance from thumb tip to the pinky MCP, in palm widths. Small means thumb tucked across the palm. */
  readonly thumbAcrossPalm: number;
}

const FINGERS = [
  [HandPoint.thumbCmc, HandPoint.thumbMcp, HandPoint.thumbTip],
  [HandPoint.indexMcp, HandPoint.indexPip, HandPoint.indexTip],
  [HandPoint.middleMcp, HandPoint.middlePip, HandPoint.middleTip],
  [HandPoint.ringMcp, HandPoint.ringPip, HandPoint.ringTip],
  [HandPoint.pinkyMcp, HandPoint.pinkyPip, HandPoint.pinkyTip],
] as const;

const TIPS = [HandPoint.indexTip, HandPoint.middleTip, HandPoint.ringTip, HandPoint.pinkyTip];

export function describeHandShape(hand: HandLandmarks): HandShape {
  const width = palmWidth(hand);
  const curl = FINGERS.map(([mcp, pip, tip]) =>
    curlOf(pointAt(hand, mcp), pointAt(hand, pip), pointAt(hand, tip)),
  ) as unknown as HandShape['curl'];

  const spread = [0, 1, 2].map(
    (i) => distance(pointAt(hand, TIPS[i]!), pointAt(hand, TIPS[i + 1]!)) / width,
  ) as unknown as HandShape['spread'];

  return {
    curl,
    spread,
    thumbIndexGap:
      distance(pointAt(hand, HandPoint.thumbTip), pointAt(hand, HandPoint.indexTip)) / width,
    thumbAcrossPalm:
      distance(pointAt(hand, HandPoint.thumbTip), pointAt(hand, HandPoint.pinkyMcp)) / width,
  };
}

/**
 * Palm width is the reference length for every ratio above. Using the MCP span rather than
 * the wrist-to-fingertip distance keeps it stable while fingers move, which is the whole
 * point — a curling index finger must not change the unit it is measured in.
 */
export function palmWidth(hand: HandLandmarks): number {
  const width = distance(pointAt(hand, HandPoint.indexMcp), pointAt(hand, HandPoint.pinkyMcp));
  // Guard against a degenerate hand collapsing the unit and producing Infinity ratios.
  return width > 1e-6 ? width : 1e-6;
}

/**
 * Curl as the normalised turn at the middle joint: a straight finger has its tip in line
 * with the knuckle (angle ~180°, curl 0); a fully folded one doubles back (angle ~0°, curl 1).
 */
function curlOf(mcp: Landmark, pip: Landmark, tip: Landmark): number {
  const a = subtract(mcp, pip);
  const b = subtract(tip, pip);
  const magnitudes = magnitude(a) * magnitude(b);
  if (magnitudes < 1e-9) return 0;
  const cosine = clamp(dot(a, b) / magnitudes, -1, 1);
  const angle = Math.acos(cosine);
  return clamp(1 - angle / Math.PI, 0, 1);
}

export function isExtended(shape: HandShape, finger: number): boolean {
  return (shape.curl[finger] ?? 1) < 0.35;
}

function subtract(a: Landmark, b: Landmark): Landmark {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function dot(a: Landmark, b: Landmark): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function magnitude(a: Landmark): number {
  return Math.sqrt(dot(a, a));
}

export function distance(a: Landmark, b: Landmark): number {
  return magnitude(subtract(a, b));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
