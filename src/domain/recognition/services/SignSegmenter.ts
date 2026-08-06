import { distance, palmWidth } from '@domain/landmarks/services/handShape';
import { HandPoint, pointAt } from '@domain/landmarks/value-objects/Landmark';
import { dominantHand, type LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

export interface SegmenterOptions {
  /** Movement above this (palm widths per frame) counts as "signing". */
  readonly motionThreshold: number;
  /**
   * Fraction of the window's peak speed below which the signer is decelerating.
   *
   * The boundary that works on fluent signing. Waiting for stillness assumes a pause that
   * only isolated dictionary recordings contain; a signer in conversation never stops, but
   * does visibly slow between signs. Stillness is just the extreme of this, so this rule
   * subsumes the old one rather than sitting beside it.
   */
  readonly decelerationDrop: number;
  /** Frames the deceleration must persist before it counts as a boundary, not a wobble. */
  readonly decelerationHold: number;
  /**
   * Frames a window must reach before a deceleration may close it.
   *
   * Signs decelerate internally — a two-part sign slows at its hinge — so without a floor
   * near the typical sign length the rule chops signs in half. Measured: dropping this from
   * 24 to 18 costs 8 points of isolated top-1 to buy 3 points of continuous recovery.
   */
  readonly minSignFrames: number;
  /** Shortest accepted sign; below this it is camera noise, not a sign. */
  readonly minFrames: number;
  /**
   * Longest a window may run before it is emitted anyway.
   *
   * A backstop now rather than the main event: signing at a near-constant speed never
   * decelerates enough to close, and a window that runs forever is never classified. It was
   * doing all the work while stillness was the only other rule, which is precisely why
   * windows were cut at a fixed length instead of at sign boundaries.
   */
  readonly maxFrames: number;
}

/**
 * Tuned against two benchmarks, because one of them was missing and hid the real failure.
 *
 * `tools/train/sweep.py` replays SWL-LSE's isolated recordings; `tools/train/continuous.py`
 * splices them into unbroken streams and asks how many signs survive. The shipped
 * stillness rule scored 0.739 on the first and **0.146 on the second** — it never closes
 * without a pause, so it only closed at `maxFrames`, and with a median sign of 30 frames
 * nearly every 48-frame window straddled a boundary. That is why the app read a signing
 * video and wrote nothing.
 *
 * These settings give 0.696 isolated and 0.384 continuous at higher precision (0.755 against
 * 0.710). Four points of the validated path bought twenty-four of the one people actually
 * use. Note the continuous benchmark splices isolated recordings and so cannot reproduce
 * real co-articulation: read it as an upper bound.
 */
export const DEFAULT_SEGMENTER_OPTIONS: SegmenterOptions = {
  motionThreshold: 0.03,
  decelerationDrop: 0.45,
  decelerationHold: 1,
  minSignFrames: 24,
  minFrames: 4,
  maxFrames: 48,
};

/**
 * Turns a continuous landmark stream into discrete sign windows.
 *
 * A window closes where the signer decelerates off their own peak speed, with `maxFrames`
 * as a backstop. Deceleration rather than stillness because stillness is a property of
 * dictionary recordings, not of signing: measured on spliced continuous streams, waiting
 * for a pause recovered 14.6% of signs against 38.4% for this rule.
 *
 * `minSignFrames` is what keeps it honest — signs decelerate internally too, so a boundary
 * is only believed once the window is already about as long as a sign.
 *
 * Deliberately a pure state machine over frames — no timers, no clock, no I/O — so its
 * behaviour is reproducible in tests by pushing a scripted frame sequence.
 */
export class SignSegmenter {
  private readonly options: SegmenterOptions;
  private window: LandmarkFrame[] = [];
  private slowFrames = 0;
  private peakMotion = 0;
  private active = false;
  private shortWindows = 0;

  constructor(options: Partial<SegmenterOptions> = {}) {
    this.options = { ...DEFAULT_SEGMENTER_OPTIONS, ...options };
  }

  /** Mid-sign right now. Read-only; the diagnostics panel shows it live. */
  get isActive(): boolean {
    return this.active;
  }

  /** Frames buffered towards the current window. */
  get pendingFrames(): number {
    return this.window.length;
  }

  /**
   * Windows completed and then discarded for being shorter than `minFrames`.
   *
   * Invisible from outside otherwise: `push` returns null both when nothing ended and when
   * something ended and was judged too short to be a sign. Those are opposite diagnoses.
   */
  get discardedShortWindows(): number {
    return this.shortWindows;
  }

  /** Feed one frame. Returns a completed window when a sign just ended, else null. */
  push(frame: LandmarkFrame): readonly LandmarkFrame[] | null {
    const hand = dominantHand(frame);
    if (!hand) return this.handleHandLost();

    const motion = this.motionSince(frame);
    this.window.push(frame);

    if (motion > this.options.motionThreshold) {
      this.active = true;
      this.peakMotion = Math.max(this.peakMotion, motion);
    }

    if (!this.active) {
      // Still idle: keep only a short tail so the next sign's start is not truncated.
      if (this.window.length > this.options.minFrames) this.window.shift();
      return null;
    }

    if (this.window.length >= this.options.maxFrames) return this.close();

    const decelerating =
      this.window.length >= this.options.minSignFrames &&
      this.peakMotion > 0 &&
      motion < this.peakMotion * this.options.decelerationDrop;

    if (!decelerating) {
      this.slowFrames = 0;
      return null;
    }

    this.slowFrames += 1;
    return this.slowFrames >= this.options.decelerationHold ? this.close() : null;
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
    if (completed.length >= this.options.minFrames) return completed;
    this.shortWindows += 1;
    return null;
  }

  reset(): void {
    this.window = [];
    this.slowFrames = 0;
    this.peakMotion = 0;
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
