/**
 * What one frame of the vision pipeline costs, per model.
 *
 * Frame rate is not a comfort metric in this app: below the segmenter's floor no window can
 * satisfy both `minSignMs` and `minFrames`, so the app writes nothing at all. Deciding which
 * model to make cheaper needs its share of the budget measured on a real device, not guessed
 * from model file sizes — the face model is the smallest of the three on disk.
 */
export interface FrameCost {
  /** Median wall-clock milliseconds of one `detectForVideo` pass. */
  readonly handsMs: number;
  readonly poseMs: number;
  readonly faceMs: number;
  /**
   * Frames per face pass, as observed. The face detector is throttled — its six scalars are
   * worth nothing beyond seed noise, and holding the last reading for a full second costs
   * 0.001 top-1 — so `faceMs` is what one pass costs, not what every frame pays. Multiply
   * accordingly before comparing it with the other two.
   */
  readonly faceEvery: number;
  /** How many frames the medians are taken over. */
  readonly samples: number;
}

/**
 * Median of the samples taken so far, per model.
 *
 * Medians, not means: a frame that lands on a garbage collection or a lost animation frame is
 * several times the typical cost, and this project has already been misled once by comparing
 * against means over data with rare extreme outliers.
 */
export class FrameCostMeter {
  private readonly hands: number[] = [];
  private readonly pose: number[] = [];
  private readonly face: number[] = [];

  constructor(private readonly window = 120) {}

  /** `faceMs` is null on the frames where the throttled face detector did not run. */
  record(handsMs: number, poseMs: number, faceMs: number | null): void {
    this.push(this.hands, handsMs);
    this.push(this.pose, poseMs);
    if (faceMs !== null) this.push(this.face, faceMs);
  }

  reset(): void {
    this.hands.length = 0;
    this.pose.length = 0;
    this.face.length = 0;
  }

  read(): FrameCost | null {
    if (this.hands.length === 0) return null;
    return {
      handsMs: median(this.hands),
      poseMs: median(this.pose),
      faceMs: this.face.length > 0 ? median(this.face) : 0,
      faceEvery: this.face.length > 0 ? this.hands.length / this.face.length : 0,
      samples: this.hands.length,
    };
  }

  private push(into: number[], value: number): void {
    into.push(value);
    if (into.length > this.window) into.shift();
  }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}
