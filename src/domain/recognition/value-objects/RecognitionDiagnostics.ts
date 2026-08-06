/**
 * What the pipeline actually did, so a silent app can be told apart from a wrong one.
 *
 * Esku draws a correct skeleton and writes nothing, and on screen that looks identical
 * whether the segmenter never closes a window, the vocabulary engine is never asked, or it
 * is asked and scores too low to speak. Three very different bugs, one symptom. Everything
 * measured so far was measured offline in Node and Python; these counters are the first
 * evidence taken from a running browser.
 */
export interface RawScore {
  readonly text: string;
  /** 0..1, straight from the softmax — deliberately *not* thresholded. */
  readonly confidence: number;
}

/**
 * Who swallowed the last completed sign.
 *
 * `classifier` means every concept scored below the engine's own floor. `stabilizer` means
 * the engine did answer and the stabiliser's higher floor rejected it — the two floors are
 * different numbers, which is exactly the confusion this distinguishes.
 */
export type WindowVeto = 'classifier' | 'stabilizer' | 'duplicate';

export interface RecognitionDiagnostics {
  readonly framesSeen: number;
  readonly framesWithHands: number;
  /** Windows the segmenter completed and handed on. */
  readonly windowsClosed: number;
  /** Windows it completed and threw away for being shorter than `minFrames`. */
  readonly windowsTooShort: number;
  readonly lastWindowFrames: number;
  /** True while the segmenter is mid-sign; with `pendingFrames`, shows it is alive. */
  readonly segmenterActive: boolean;
  readonly pendingFrames: number;
  readonly vocabularyReady: boolean;
  readonly vocabularyInvocations: number;
  /** Unfiltered best guesses for the last window, below the thresholds included. */
  readonly lastRawTop: readonly RawScore[];
  readonly lastVeto: WindowVeto | null;
  readonly wordsEmitted: number;
}

export const EMPTY_DIAGNOSTICS: RecognitionDiagnostics = {
  framesSeen: 0,
  framesWithHands: 0,
  windowsClosed: 0,
  windowsTooShort: 0,
  lastWindowFrames: 0,
  segmenterActive: false,
  pendingFrames: 0,
  vocabularyReady: false,
  vocabularyInvocations: 0,
  lastRawTop: [],
  lastVeto: null,
  wordsEmitted: 0,
};
