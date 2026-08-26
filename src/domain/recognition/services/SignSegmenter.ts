import { distance, palmWidth } from '@domain/landmarks/services/handShape';
import { HandPoint, pointAt } from '@domain/landmarks/value-objects/Landmark';
import { dominantHand, type LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';

export interface SegmenterOptions {
  /**
   * Movement above this (palm widths per **second**) counts as "signing".
   *
   * Per second, not per frame: a frame-relative floor is a speed floor divided by the frame
   * rate, so the same unhurried sign counts as motion on a slow phone and as stillness on a
   * fast one.
   */
  readonly motionRate: number;
  /**
   * Fraction of the window's peak speed below which the signer is decelerating.
   *
   * The boundary that works on fluent signing. Waiting for stillness assumes a pause that
   * only isolated dictionary recordings contain; a signer in conversation never stops, but
   * does visibly slow between signs. Stillness is just the extreme of this, so this rule
   * subsumes the old one rather than sitting beside it.
   */
  readonly decelerationDrop: number;
  /** How long the deceleration must persist before it counts as a boundary, not a wobble. */
  readonly decelerationHoldMs: number;
  /**
   * How long a window must run before a deceleration may close it.
   *
   * Two opposed pressures. Signs decelerate internally — a two-part sign slows at its hinge —
   * so without a floor near the typical sign length the rule chops signs in half. But the
   * floor also puts short signs out of reach as arithmetic: a window forced to last F can
   * only reach IoU 0.5 against a sign of length L if L >= F/2, and half of real signing is
   * shorter than 480 ms.
   *
   * Which pressure wins depends on the classifier, and it has changed sides. Against the
   * dictionary-only model, 1150 -> 850 cost 8 points of isolated top-1 to buy 3 of continuous
   * recovery, so the floor stayed high. Re-measured on the co-articulated model over the same
   * four seeds, the move buys +6.9 points of continuous word recall at an unchanged gate
   * (34.9% -> 41.9%) for 5.4 of isolated. Shorter windows only pay once the classifier has
   * seen a sign glued to the sign before it; the two levers multiply.
   *
   * Part of that +6.9 is not recall, though: shorter windows also write more into pauses,
   * from 31% of the gaps between annotated sentences to 36%. The shipped pairing raises the
   * window gate to 0.60 to give that back, and banks **+2.3 points at a flat 30%** — see
   * `RecognizeSignsUseCase`, which holds the other half of this decision.
   */
  readonly minSignMs: number;
  /** Shortest accepted sign; below this it is camera noise, not a sign. */
  readonly minMs: number;
  /**
   * Fewest frames a window may contain, whatever its duration.
   *
   * A separate concern from `minMs`, and both are needed. `minMs` rejects what is too brief
   * to be a sign; this rejects what has too few samples to *describe* one — sixteen feature
   * slots resampled from two frames is a vector the model has never seen. On a device slow
   * enough to trip this, recognition is genuinely impossible, and the discarded-window
   * counter is what makes that visible instead of silent.
   */
  readonly minFrames: number;
  /**
   * Longest a window may run before it is emitted anyway.
   *
   * A backstop now rather than the main event: signing at a near-constant speed never
   * decelerates enough to close, and a window that runs forever is never classified. It was
   * doing all the work while stillness was the only other rule, which is precisely why
   * windows were cut at a fixed length instead of at sign boundaries.
   */
  readonly maxMs: number;
}

/**
 * Tuned against two benchmarks, because one of them was missing and hid the real failure.
 *
 * `tools/train/sweep.py` replays SWL-LSE's isolated recordings; `tools/train/continuous.py`
 * splices them into unbroken streams and asks how many signs survive. The shipped
 * stillness rule scored 0.739 on the first and **0.146 on the second** — it never closes
 * without a pause, so it only closed at the cap, and with a median sign of 30 frames
 * nearly every 48-frame window straddled a boundary. That is why the app read a signing
 * video and wrote nothing.
 *
 * These settings give 0.696 isolated and 0.384 continuous at higher precision (0.755 against
 * 0.710). Four points of the validated path bought twenty-four of the one people actually
 * use. Note the continuous benchmark splices isolated recordings and so cannot reproduce
 * real co-articulation: read it as an upper bound.
 *
 * **Expressed in time, not frames, and that was a bug for a while.** Every threshold here
 * used to be a frame count, swept against SWL-LSE — which is 20.00 fps in all 300 of its
 * recordings. The live pipeline runs three MediaPipe models per frame and reaches whatever
 * the device allows, so those counts silently meant a different duration on every phone.
 * Below the dataset's rate a sign never reached that 24-frame floor and every window was
 * discarded as too short: measured in a real browser, 23 frames in 16 s, six windows
 * discarded, the vocabulary engine asked **zero** times, nothing written. The same build
 * recognised "Dolor" at 68% as soon as the input was slowed to restore the sampling density.
 * `LandmarkFrame` has always carried `timestampMs`; it just was not read.
 *
 * The values below are the swept frame counts converted at 20.00 fps — and N frames span
 * N-1 intervals, so the old 24-frame floor is 1150 ms, not 1200. Converting exactly is what
 * lets `simulate_app.py` reproduce the pre-conversion scores and prove the port faithful.
 * `minSignMs` is the one value that no longer comes from that sweep: it was re-measured on
 * real continuous signing and lowered to 850, which is 18 frames at the dataset's rate.
 */
export const DEFAULT_SEGMENTER_OPTIONS: SegmenterOptions = {
  motionRate: 0.6,
  decelerationDrop: 0.45,
  decelerationHoldMs: 50,
  minSignMs: 850,
  minMs: 150,
  minFrames: 4,
  maxMs: 2350,
};

/**
 * Turns a continuous landmark stream into discrete sign windows.
 *
 * A window closes where the signer decelerates off their own peak speed, with `maxMs`
 * as a backstop. Deceleration rather than stillness because stillness is a property of
 * dictionary recordings, not of signing: measured on spliced continuous streams, waiting
 * for a pause recovered 14.6% of signs against 38.4% for this rule.
 *
 * `minSignMs` is what keeps it honest — signs decelerate internally too, so a boundary
 * is only believed once the window is already about as long as a sign.
 *
 * Deliberately a pure state machine over the frames' own `timestampMs` — no timers, no
 * clock, no I/O — so its behaviour is reproducible in tests by pushing a scripted sequence,
 * and identical at any frame rate.
 */
export class SignSegmenter {
  private readonly options: SegmenterOptions;
  private window: LandmarkFrame[] = [];
  private slowMs = 0;
  private peakRate = 0;
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
   * Windows completed and then discarded for being too brief or too sparsely sampled.
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

    const sinceLastMs = this.gapBefore(frame);
    const rate = this.motionRateSince(frame, sinceLastMs);
    this.window.push(frame);

    if (rate > this.options.motionRate) {
      this.active = true;
      this.peakRate = Math.max(this.peakRate, rate);
    }

    if (!this.active) {
      // Still idle: keep only a short tail so the next sign's start is not truncated.
      while (this.window.length > 1 && this.span() > this.options.minMs) this.window.shift();
      return null;
    }

    if (this.span() >= this.options.maxMs) return this.close();

    const decelerating =
      this.span() >= this.options.minSignMs &&
      this.peakRate > 0 &&
      rate < this.peakRate * this.options.decelerationDrop;

    if (!decelerating) {
      this.slowMs = 0;
      return null;
    }

    // Accumulated rather than counted: one frame is 50 ms at the frame rate this was tuned
    // at and 17 ms at 60 fps, so counting frames would make the hold three times stricter on
    // a fast device — the same units mistake one level down.
    this.slowMs += sinceLastMs;
    return this.slowMs >= this.options.decelerationHoldMs ? this.close() : null;
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
    const spanMs = spanOf(completed);
    this.reset();

    const longEnough = spanMs >= this.options.minMs;
    const sampledEnough = completed.length >= this.options.minFrames;
    if (longEnough && sampledEnough) return completed;

    this.shortWindows += 1;
    return null;
  }

  reset(): void {
    this.window = [];
    this.slowMs = 0;
    this.peakRate = 0;
    this.active = false;
  }

  /** Wall-clock span of the frames buffered so far. */
  private span(): number {
    return spanOf(this.window);
  }

  /** Time since the previous buffered frame, from the frames themselves. */
  private gapBefore(frame: LandmarkFrame): number {
    const previous = this.window.at(-1);
    return previous ? frame.timestampMs - previous.timestampMs : 0;
  }

  /**
   * Mean fingertip speed since the previous frame, in palm widths per second.
   *
   * Palm widths so that moving the phone closer does not read as faster signing; per second
   * so that a slow phone does not read as faster signing either.
   */
  private motionRateSince(frame: LandmarkFrame, sinceLastMs: number): number {
    const previous = this.window.at(-1);
    if (!previous || sinceLastMs <= 0) return 0;

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
    return total / tips.length / scale / (sinceLastMs / 1000);
  }
}

function spanOf(frames: readonly LandmarkFrame[]): number {
  if (frames.length < 2) return 0;
  return frames.at(-1)!.timestampMs - frames[0]!.timestampMs;
}
