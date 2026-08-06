import type { SignCandidate } from '../value-objects/Gloss';

/**
 * Suppresses flicker in a live prediction stream.
 *
 * A per-frame classifier jitters between neighbours, so emitting its raw argmax would spell
 * "AAAABAAAA" for a held A. A candidate is only accepted once the same concept has won
 * `agreementFrames` times in a row above `minConfidence`, and the same concept is not
 * emitted twice until something else intervenes — otherwise holding one handshape would
 * repeat that letter forever.
 */
export class CandidateStabilizer {
  private streakConceptId: string | null = null;
  private streak = 0;
  private lastEmitted: string | null = null;

  constructor(
    private readonly agreementFrames = 3,
    private readonly minConfidence = 0.6,
  ) {}

  /** Exposed so a rejection can be attributed to this floor rather than the engine's own. */
  get threshold(): number {
    return this.minConfidence;
  }

  /** Returns the candidate to append to the transcript, or null while unstable. */
  accept(candidate: SignCandidate | null): SignCandidate | null {
    if (!candidate || candidate.confidence < this.minConfidence) {
      this.streakConceptId = null;
      this.streak = 0;
      return null;
    }

    const conceptId = candidate.gloss.conceptId;
    if (conceptId === this.streakConceptId) {
      this.streak += 1;
    } else {
      this.streakConceptId = conceptId;
      this.streak = 1;
    }

    if (this.streak < this.agreementFrames || conceptId === this.lastEmitted) return null;

    this.lastEmitted = conceptId;
    return candidate;
  }

  /**
   * Call when the hand leaves frame or a sign boundary passes: it clears the
   * "already emitted" latch so the same sign can legitimately be signed twice in a row.
   */
  release(): void {
    this.streakConceptId = null;
    this.streak = 0;
    this.lastEmitted = null;
  }
}
