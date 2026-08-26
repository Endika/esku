import { describeHandShape, type HandShape } from '@domain/landmarks/services/handShape';
import { dominantHand, type LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import {
  byConfidenceDescending,
  type SignCandidate,
} from '@domain/recognition/value-objects/Gloss';
import type { RawScore } from '@domain/recognition/value-objects/RecognitionDiagnostics';
import { type HandshapeTemplate, LSE_ALPHABET } from './packs/lseAlphabet';

/** How far a feature may drift before it contributes nothing. Wider = more forgiving. */
const TOLERANCE = { curl: 0.45, spread: 0.5, gap: 0.6 } as const;

/** Below this the match is not worth showing; the shape is probably a transition. */
const MIN_SCORE = 0.72;

const MAX_CANDIDATES = 3;

/**
 * Recognises fingerspelled letters by scoring the live handshape against a table.
 *
 * Rule-based on purpose: the LSE alphabet is a small set of static, well-documented
 * handshapes, and a table needs no dataset, no training run and no download — so the app is
 * useful on first launch, before the vocabulary model exists. It also degrades honestly:
 * an ambiguous shape scores low everywhere rather than picking a confident wrong letter.
 *
 * Reads only the last frame of the window; letters are held, not travelled.
 */
export class HandshapeAlphabetClassifier implements ISignClassifier {
  readonly id = 'lse-alphabet';
  readonly granularity = 'frame' as const;

  #scores: readonly RawScore[] = [];

  constructor(private readonly templates: readonly HandshapeTemplate[] = LSE_ALPHABET) {}

  /**
   * Every letter's score from the last `classify`, before `MIN_SCORE` threw any away.
   *
   * Without it a shape that scored 0.71 on the right letter and a frame with no hand in it
   * are the same empty array from outside, and they want opposite fixes — a looser table
   * against a stricter one. `RecognizeSignsUseCase` only reads this from `window` engines,
   * so nothing in the app changes; it exists for the bench that measures this engine against
   * a real fingerspelling corpus, where separating those two is the whole question.
   */
  get lastScores(): readonly RawScore[] {
    return this.#scores;
  }

  isReady(): boolean {
    return true;
  }

  async load(): Promise<void> {
    // Nothing to load: the whole model is the table.
  }

  async classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]> {
    const frame = window.at(-1);
    const hand = frame ? dominantHand(frame) : null;
    if (!hand) {
      this.#scores = [];
      return [];
    }

    const shape = describeHandShape(hand);
    const scored = this.templates
      .map((template) => ({ template, score: scoreTemplate(shape, template) }))
      .sort((a, b) => b.score - a.score);
    this.#scores = scored.map(({ template, score }) => ({
      text: template.letter,
      confidence: score,
    }));

    return scored
      .filter(({ score }) => score >= MIN_SCORE)
      .map(({ template, score }) => ({
        gloss: { id: template.letter, conceptId: template.letter, text: template.letter },
        confidence: score,
        source: 'alphabet' as const,
      }))
      .sort(byConfidenceDescending)
      .slice(0, MAX_CANDIDATES);
  }
}

/**
 * Mean closeness across every feature the template constrains.
 *
 * Unconstrained features (`null`, or simply absent) are skipped rather than scored as a
 * perfect match — otherwise a loosely-specified letter would out-score a precise one just
 * by asking for less.
 */
function scoreTemplate(shape: HandShape, template: HandshapeTemplate): number {
  const scores: number[] = [];
  /**
   * Any single feature landing outside tolerance vetoes the whole letter.
   *
   * Averaging alone is not enough: `d` and `l` differ only in the thumb, so a completely
   * wrong thumb would cost just one fifth of the score and `l` would win on its three
   * matching curled fingers. Vetoing makes the engine answer "nothing" for a shape that is
   * neither letter, which is the honest answer and the recoverable one — a missing letter is
   * obvious to the signer, a confidently wrong one is not.
   */
  let vetoed = false;

  const add = (actual: number, target: number, tolerance: number) => {
    const score = closeness(actual, target, tolerance);
    if (score <= 0) vetoed = true;
    scores.push(score);
  };

  template.curl.forEach((target, finger) => {
    if (target === null) return;
    add(shape.curl[finger] ?? 0, target, TOLERANCE.curl);
  });

  template.spread?.forEach((target, gap) => {
    if (target === null) return;
    add(shape.spread[gap] ?? 0, target, TOLERANCE.spread);
  });

  if (template.thumbIndexGap !== undefined) {
    add(shape.thumbIndexGap, template.thumbIndexGap, TOLERANCE.gap);
  }
  if (template.thumbAcrossPalm !== undefined) {
    add(shape.thumbAcrossPalm, template.thumbAcrossPalm, TOLERANCE.gap);
  }

  if (vetoed || scores.length === 0) return 0;
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

function closeness(actual: number, target: number, tolerance: number): number {
  return Math.max(0, 1 - Math.abs(actual - target) / tolerance);
}
