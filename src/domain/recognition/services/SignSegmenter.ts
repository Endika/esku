import { distance, palmWidth } from '@domain/landmarks/services/handShape';
import { HandPoint, pointAt } from '@domain/landmarks/value-objects/Landmark';
import { dominantHand, type LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

export interface SegmenterOptions {
  /** Movement above this (palm widths per frame) counts as "signing". */
  readonly motionThreshold: number;
  /** Frames of stillness that close a sign. */
  readonly settleFrames: number;
  /** Shortest accepted sign; below this it is camera noise, not a sign. */
  readonly minFrames: number;
  /** Longest window kept. Bounds memory and matches the model's input length. */
  readonly maxFrames: number;
}

export const DEFAULT_SEGMENTER_OPTIONS: SegmenterOptions = {
  motionThreshold: 0.08,
  settleFrames: 6,
  minFrames: 4,
  maxFrames: 48,
};

/**
 * Turns a continuous landmark stream into discrete sign windows.
 *
 * The rule is motion-energy gating: a sign starts when the hand begins moving, and ends
 * once it holds still for `settleFrames`. Without this the classifier would fire on every
 * frame, including the transitions *between* signs, which are not signs at all.
 *
 * Deliberately a pure state machine over frames — no timers, no clock, no I/O — so its
 * behaviour is reproducible in tests by pushing a scripted frame sequence.
 */
export class SignSegmenter {
  private readonly options: SegmenterOptions;
  private window: LandmarkFrame[] = [];
  private stillFrames = 0;
  private active = false;

  constructor(options: Partial<SegmenterOptions> = {}) {
    this.options = { ...DEFAULT_SEGMENTER_OPTIONS, ...options };
  }

  /** Feed one frame. Returns a completed window when a sign just ended, else null. */
  push(frame: LandmarkFrame): readonly LandmarkFrame[] | null {
    const hand = dominantHand(frame);
    if (!hand) return this.handleHandLost();

    const motion = this.motionSince(frame);
    this.window.push(frame);
    if (this.window.length > this.options.maxFrames) this.window.shift();

    if (motion > this.options.motionThreshold) {
      this.active = true;
      this.stillFrames = 0;
      return null;
    }

    if (!this.active) {
      // Still idle: keep only a short tail so the next sign's start is not truncated.
      if (this.window.length > this.options.minFrames) this.window.shift();
      return null;
    }

    this.stillFrames += 1;
    return this.stillFrames >= this.options.settleFrames ? this.close() : null;
  }

  /** A hand leaving frame ends the sign as surely as stillness does. */
  private handleHandLost(): readonly LandmarkFrame[] | null {
    if (!this.active) {
      this.window = [];
      return null;
    }
    return this.close();
  }

  private close(): readonly LandmarkFrame[] | null {
    const completed = this.window;
    this.reset();
    return completed.length >= this.options.minFrames ? completed : null;
  }

  reset(): void {
    this.window = [];
    this.stillFrames = 0;
    this.active = false;
  }

  /**
   * Mean fingertip displacement from the previous frame, in palm widths so that moving the
   * phone closer does not read as faster signing.
   */
  private motionSince(frame: LandmarkFrame): number {
    const previous = this.window.at(-1);
    if (!previous) return 0;

    const current = dominantHand(frame);
    const before = dominantHand(previous);
    if (!current || !before) return 0;

    const tips = [
      HandPoint.thumbTip,
      HandPoint.indexTip,
      HandPoint.middleTip,
      HandPoint.ringTip,
      HandPoint.pinkyTip,
    ];
    const scale = palmWidth(current);
    const total = tips.reduce(
      (sum, tip) => sum + distance(pointAt(current, tip), pointAt(before, tip)),
      0,
    );
    return total / tips.length / scale;
  }
}
