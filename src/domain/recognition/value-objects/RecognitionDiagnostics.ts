import type { FrameCost } from '@domain/landmarks/value-objects/FrameCost';

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

export interface SignatureBlock {
  /** 0..1 of the 16 sampled frames where this part contributed nothing at all. */
  readonly emptyFrames: number;
  /** Mean absolute value across the part's floats, over the frames that had it. */
  readonly meanMagnitude: number;
}

/**
 * The feature vector the browser actually built, summarised per body part.
 *
 * Measured over SWL-LSE's test split for comparison — right hand 3.37, left 3.61, torso
 * 0.515, face 0.157, and torso empty on 0.0% of frames. A part that reads far from its
 * number here is receiving something training never saw.
 */
export interface SignatureProfile {
  readonly rightHand: SignatureBlock;
  readonly leftHand: SignatureBlock;
  readonly torso: SignatureBlock;
  readonly face: SignatureBlock;
}

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
  /** Vocabulary words only. Fingerspelled letters are counted separately below. */
  readonly wordsEmitted: number;
  /** Letters from the alphabet engine, which speaks even while the vocabulary is silent. */
  readonly lettersEmitted: number;
  /** What the model was actually fed for the last window. Null until one is classified. */
  readonly lastSignature: SignatureProfile | null;
  /** Per-model cost of a frame. Null until the camera has produced one. */
  readonly frameCost: FrameCost | null;
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
  lettersEmitted: 0,
  lastSignature: null,
  frameCost: null,
};
