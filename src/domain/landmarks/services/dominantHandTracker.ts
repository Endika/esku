import type { Handedness, HandLandmarks } from '../value-objects/Landmark';
import { HandPoint, pointAt } from '../value-objects/Landmark';
import { dominantHand, type LandmarkFrame } from '../value-objects/LandmarkFrame';

/** How fast an idle hand's score fades. One frame of stillness costs it a tenth. */
const DECAY = 0.9;

/**
 * Follows which hand is doing the signing, across frames.
 *
 * `dominantHand` picks the larger bounding box, which is the best a single frame allows and
 * not very good: audited against LSE-FS-UVigo's own `handness` labels, it takes the wrong hand
 * in 23.7% of the frames where both are visible. Those are only 11.8% of frames, but the cost
 * is out of proportion — the fingerspelling engine carries hidden state, so a wrong hand does
 * not lose one frame, it corrupts the GRU's state and everything after it. Measured end to
 * end, swapping this rule alone took character error from 0.307 to 0.166.
 *
 * The signal is accumulated wrist motion: fingerspelling moves, a resting hand does not. It is
 * decayed rather than summed so a signer who changes hands is followed instead of remembered.
 *
 * Deliberately *not* "prefer the right hand". That would score 99% on this corpus, which is
 * 199/200 right-dominant, and fail every left-handed signer — the same shape of mistake as the
 * handedness inversion this repository already shipped once.
 */
export class DominantHandTracker {
  #motion = new Map<Handedness, number>();
  #previous = new Map<Handedness, { x: number; y: number }>();

  /** Clear the history. A new session, or a hand that left frame, shares nothing with before. */
  reset(): void {
    this.#motion.clear();
    this.#previous.clear();
  }

  pick(frame: LandmarkFrame): HandLandmarks | null {
    const wrists = new Map<Handedness, { x: number; y: number }>();
    for (const hand of frame.hands) {
      const wrist = pointAt(hand, HandPoint.wrist);
      wrists.set(hand.handedness, { x: wrist.x, y: wrist.y });
    }

    for (const [side, wrist] of wrists) {
      const before = this.#previous.get(side);
      const travelled = before ? Math.hypot(wrist.x - before.x, wrist.y - before.y) : 0;
      this.#motion.set(side, (this.#motion.get(side) ?? 0) * DECAY + travelled);
    }
    this.#previous = wrists;

    if (frame.hands.length <= 1) return frame.hands[0] ?? null;

    // Before any motion has been seen the scores are all zero and this would pick arbitrarily,
    // so fall back to the stateless rule for as long as that is true.
    const best = frame.hands.reduce((a, b) =>
      (this.#motion.get(b.handedness) ?? 0) > (this.#motion.get(a.handedness) ?? 0) ? b : a,
    );
    return (this.#motion.get(best.handedness) ?? 0) > 0 ? best : dominantHand(frame);
  }
}
