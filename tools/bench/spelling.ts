import { CandidateStabilizer } from '@domain/recognition/services/CandidateStabilizer';
import type { SignCandidate } from '@domain/recognition/value-objects/Gloss';

/** What one frame contributed: the engine's best letter, and whether a hand was there at all. */
export interface SpellStep {
  readonly top: SignCandidate | null;
  readonly empty: boolean;
}

export interface EditOps {
  readonly distance: number;
  readonly substitutions: number;
  readonly insertions: number;
  readonly deletions: number;
  /** `[expected, written]` for each substitution, left to right. Feeds the confusion matrix. */
  readonly pairs: readonly (readonly [string, string])[];
}

/**
 * Levenshtein with a backtrace, because the total on its own decides nothing.
 *
 * A distance of 40 over the corpus could be forty wrong letters, forty invented ones or forty
 * missed ones, and each points at a different part of the engine: substitutions at the
 * handshape table, insertions at the stabiliser letting transitions through, deletions at the
 * confidence floor. Reporting only the sum is what would let a bad engine and a strict one
 * look alike.
 *
 * Ties prefer the diagonal, so an equal-cost choice is read as a substitution rather than as
 * an insertion plus a deletion. The preference is arbitrary but must be *fixed*: a wobbling
 * tie-break moves the three counts around while the total sits still, which is the hardest
 * kind of drift to spot.
 */
export function editOps(expected: string, actual: string): EditOps {
  const rows = expected.length;
  const columns = actual.length;
  const cost: number[][] = Array.from({ length: rows + 1 }, (_, i) =>
    Array.from({ length: columns + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
  );

  for (let i = 1; i <= rows; i++) {
    for (let j = 1; j <= columns; j++) {
      const same = expected[i - 1] === actual[j - 1];
      cost[i]![j] = Math.min(
        cost[i - 1]![j - 1]! + (same ? 0 : 1),
        cost[i - 1]![j]! + 1,
        cost[i]![j - 1]! + 1,
      );
    }
  }

  const pairs: [string, string][] = [];
  let substitutions = 0;
  let insertions = 0;
  let deletions = 0;
  let i = rows;
  let j = columns;

  while (i > 0 || j > 0) {
    const here = cost[i]![j]!;
    if (i > 0 && j > 0) {
      const same = expected[i - 1] === actual[j - 1];
      if (cost[i - 1]![j - 1]! + (same ? 0 : 1) === here) {
        if (!same) {
          substitutions += 1;
          pairs.push([expected[i - 1]!, actual[j - 1]!]);
        }
        i -= 1;
        j -= 1;
        continue;
      }
    }
    if (i > 0 && cost[i - 1]![j]! + 1 === here) {
      deletions += 1;
      i -= 1;
      continue;
    }
    insertions += 1;
    j -= 1;
  }

  pairs.reverse();
  return { distance: cost[rows]![columns]!, substitutions, insertions, deletions, pairs };
}

/**
 * Collapses a stream of per-frame guesses into the word the app would have written.
 *
 * Uses the shipped `CandidateStabilizer` rather than reimplementing the collapse, and drives
 * it the way `RecognizeSignsUseCase.onFrame` does: a step that yields no candidate releases
 * the latch first — no hand, or an engine that abstained — and every step is then offered.
 * That release is the only reason a doubled letter can be spelled at all.
 *
 * The stabiliser is passed in because the two engines need different constructions, and the
 * bench must measure whichever the app actually ships. A handshape table flickers and wants
 * three agreeing frames; a CTC head spikes and wants one. Hard-coding either would make this
 * bench measure a configuration nobody runs.
 */
export function spell(
  steps: readonly SpellStep[],
  stabilizer: CandidateStabilizer = new CandidateStabilizer(),
  releaseOnAbstain = false,
): string {
  let written = '';
  for (const step of steps) {
    // Two different app versions, two different boundary rules, and they must not be mixed.
    // The table engine released only when the hand left frame; the CTC engine also releases
    // when the model says "no letter here", because that abstention *is* a boundary. Scoring
    // the table under the CTC rule measures a build nobody ever shipped.
    if (step.empty || (releaseOnAbstain && step.top === null)) stabilizer.release();
    const accepted = stabilizer.accept(step.top);
    if (accepted) written += accepted.gloss.text;
  }
  return written;
}
