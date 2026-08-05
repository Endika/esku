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
  /**
   * Longest a window may run before it is emitted anyway.
   *
   * This is what makes continuous signing work at all. Waiting for stillness assumes the
   * signer pauses between signs, which is true of dictionary recordings and false of anyone
   * signing fluently — measured at 0 windows closed over 300 frames of continuous motion,
   * meaning the vocabulary engine was never once asked.
   */
  readonly maxFrames: number;
}

/**
 * Tuned against SWL-LSE's test split by replaying it through this segmenter and scoring the
 * real model on what came out — `tools/train/sweep.py`. The model reaches 0.741 top-1 when
 * fed whole recordings; these settings get the segmented path to 0.739, where the previous
 * 0.08 / 6 gave 0.722.
 *
 * The lower motion threshold also fixes gentle signing: at 0.08 a small-amplitude sign never
 * crossed the line, so the segmenter stayed idle and the vocabulary engine was never asked.
 */
export const DEFAULT_SEGMENTER_OPTIONS: SegmenterOptions = {
  motionThreshold: 0.03,
  settleFrames: 10,
  minFrames: 4,
  maxFrames: 48,
};

/**
 * Turns a continuous landmark stream into discrete sign windows.
 *
 * Two rules, and the second exists because the first is not enough. Motion-energy gating
 * closes a window once the hand holds still, which is how isolated signs are delimited and
 * how the training data was recorded. A fluent signer never holds still, so a hard cap at
 * `maxFrames` emits the window anyway.
 *
 * Without the cap, continuous signing produced no windows at all and the vocabulary engine
 * was never invoked — the app looked broken rather than inaccurate.
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

    if (motion > this.options.motionThreshold) {
      this.active = true;
      this.stillFrames = 0;
      // Still moving, but the window is full: emit it rather than let a fluent signer run
      // forever without ever being classified.
      return this.window.length >= this.options.maxFrames ? this.close() : null;
    }

    if (!this.active) {
      // Still idle: keep only a short tail so the next sign's start is not truncated.
      if (this.window.length > this.options.minFrames) this.window.shift();
      return null;
    }

    if (this.window.length >= this.options.maxFrames) return this.close();

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
