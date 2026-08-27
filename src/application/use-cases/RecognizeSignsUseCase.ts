import type { ILandmarkSource } from '@domain/landmarks/services/ILandmarkSource';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import { CandidateStabilizer } from '@domain/recognition/services/CandidateStabilizer';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import { SignSegmenter } from '@domain/recognition/services/SignSegmenter';
import {
  byConfidenceDescending,
  type SignCandidate,
} from '@domain/recognition/value-objects/Gloss';
import type {
  RawScore,
  RecognitionDiagnostics,
  SignatureProfile,
  WindowVeto,
} from '@domain/recognition/value-objects/RecognitionDiagnostics';
import { Transcript } from '@domain/transcript/entities/Transcript';

export interface RecognitionUpdate {
  readonly transcript: Transcript;
  /** What the engines are currently seeing, best first. For the live hint, not the text. */
  readonly candidates: readonly SignCandidate[];
  /** The frame these candidates came from, so the UI can draw what was tracked. */
  readonly frame: LandmarkFrame;
  /** Counters behind the pipeline, for the diagnostics panel. Never shown by default. */
  readonly diagnostics: RecognitionDiagnostics;
}

export type RecognitionListener = (update: RecognitionUpdate) => void;

/** Emitted alongside edits, which change the text without any new camera input. */
const EMPTY_FRAME: LandmarkFrame = { timestampMs: 0, hands: [] };

/**
 * Drives the whole live pipeline: camera → segmenter → classifiers → stabiliser → transcript.
 *
 * The stabiliser is what makes the output readable: without it a held handshape would append
 * its letter on every single frame. Frame- and window-granularity engines get their own
 * stabiliser, because a letter being held is not evidence about the last completed sign.
 */
export class RecognizeSignsUseCase {
  private readonly segmenter = new SignSegmenter();
  /**
   * One agreeing frame, and 0.50.
   *
   * The old handshape table needed three frames of agreement to damp its flicker. The CTC
   * head does not flicker, it *spikes*: the loss is invariant to how long a letter is held,
   * so the model marks each letter on a frame or two and blanks the rest. Measured on
   * LSE-FS-UVigo, non-blank runs are 26% one frame long and only 23% reach three — and a
   * three-frame rule over those posteriors writes 550 letters of 3,758 and not one whole
   * word, against 2,981 and 53 words at one frame.
   *
   * With `release()` on abstention, one-frame agreement plus the no-repeat latch *is* CTC's
   * greedy collapse. The rule was already here; it just had the wrong constant.
   */
  private readonly frameStabilizer = new CandidateStabilizer(1, 0.5);
  /**
   * 0.45, and it is the number that decides: it is a *second* floor, applied on top of the
   * 0.30 the vocabulary engine enforces, so a word has to clear this one.
   *
   * It was 0.60, set when this figure looked unaffordable: a third of the windows landing in
   * gaps between annotated sentences got a word written, and the gate barely discriminated.
   * Over half of that third was the model's own `__NADA__` class being written as a word.
   * The negatives an earlier version of this comment said the corpus could not supply had
   * been trained and shipped all along, and nothing on this side read them. With the
   * abstention honoured, 0.60 writes into 13.3% of pause windows (sd 2.1, four held-out
   * signers' worth of seeds) rather than 30.0%, and that is what freed this knob.
   *
   * 0.45 spends the room on recall: **37.3% → 44.2%** (sd 0.7) of signs written correctly on
   * real continuous signing, with pause babble at 19.9% (sd 1.9) — still well under the 30%
   * that shipped for months. All four seeds gain. Lower gates keep paying (46.4% at 0.40,
   * 50.9% at 0.30) but babble climbs with them past what this corpus can defend.
   */
  private readonly windowStabilizer = new CandidateStabilizer(1, 0.45);
  private transcript = new Transcript();
  private listener: RecognitionListener | null = null;
  /** Guards against overlapping async classify calls piling up behind a slow frame. */
  private busy = false;
  /** Set while teaching: the next completed sign is handed over instead of transcribed. */
  private capture: ((window: readonly LandmarkFrame[]) => void) | null = null;
  /**
   * A finished sign waiting for the classifier to free up.
   *
   * A window closes on exactly one frame. Dropping it because a previous classification was
   * still running loses the whole sign, permanently and silently — and with three MediaPipe
   * models plus a GRU running per frame on a phone, that is the common case, not the rare
   * one. Holding it here means a slow device recognises late instead of not at all.
   */
  private pendingWindow: readonly LandmarkFrame[] | null = null;

  private framesSeen = 0;
  private framesWithHands = 0;
  private windowsClosed = 0;
  private lastWindowFrames = 0;
  private vocabularyInvocations = 0;
  private lastRawTop: readonly RawScore[] = [];
  private lastVeto: WindowVeto | null = null;
  private wordsEmitted = 0;
  private lettersEmitted = 0;
  private lastSignature: SignatureProfile | null = null;
  /** The segmenter outlives a session, so short-window counts are read as a delta. */
  private shortWindowsAtStart = 0;

  constructor(
    private readonly source: ILandmarkSource,
    private readonly classifiers: readonly ISignClassifier[],
  ) {}

  async start(listener: RecognitionListener): Promise<void> {
    this.listener = listener;
    this.resetDiagnostics();
    await this.source.start((frame) => void this.onFrame(frame));
  }

  stop(): void {
    this.source.stop();
    this.pendingWindow = null;
    this.segmenter.reset();
    this.frameStabilizer.release();
    this.windowStabilizer.release();
    this.listener = null;
  }

  clear(): void {
    this.transcript = this.transcript.clear();
    this.frameStabilizer.release();
    this.windowStabilizer.release();
    this.emit([], EMPTY_FRAME);
  }

  undo(): void {
    this.transcript = this.transcript.removeLast();
    // Releasing lets the user re-sign what they just deleted, which is usually the point.
    this.frameStabilizer.release();
    this.windowStabilizer.release();
    this.emit([], EMPTY_FRAME);
  }

  get current(): Transcript {
    return this.transcript;
  }

  /**
   * Resolves with the next completed sign instead of transcribing it.
   *
   * Recording reuses the live segmenter rather than a separate timed capture, so a taught
   * example is delimited exactly the way a recognised sign will be. Capturing on a stopwatch
   * would train the app on windows it never sees at recognition time.
   */
  captureWindow(): Promise<readonly LandmarkFrame[]> {
    return new Promise((resolve) => {
      this.capture = resolve;
    });
  }

  cancelCapture(): void {
    this.capture = null;
  }

  get isCapturing(): boolean {
    return this.capture !== null;
  }

  private async onFrame(frame: LandmarkFrame): Promise<void> {
    this.framesSeen += 1;
    if (frame.hands.length === 0) {
      // A hand leaving frame is a deliberate boundary: it lets the same letter repeat.
      this.frameStabilizer.release();
    } else {
      this.framesWithHands += 1;
    }

    const closedWindow = this.segmenter.push(frame);
    if (closedWindow) {
      this.windowsClosed += 1;
      this.lastWindowFrames = closedWindow.length;
      this.pendingWindow = closedWindow;
    }

    if (this.capture) {
      // Teaching: hand the finished sign over, and transcribe nothing meanwhile — the user
      // is demonstrating a sign, not dictating.
      if (closedWindow) {
        const deliver = this.capture;
        this.capture = null;
        this.pendingWindow = null;
        deliver(closedWindow);
      }
      this.emit([], frame);
      return;
    }

    if (this.busy) return;
    this.busy = true;
    try {
      const live = await this.classifyFrame([frame]);
      // No candidate means a boundary: either no hand, or the model said "no letter here".
      // Releasing on it is what lets the second L of ELLA through — without it the latch
      // that stops a held letter repeating also swallows a genuine double.
      if (live.length === 0) this.frameStabilizer.release();
      const accepted = this.frameStabilizer.accept(live[0] ?? null);
      if (accepted) this.append(accepted, frame.timestampMs);

      const pending = this.pendingWindow;
      this.pendingWindow = null;
      if (pending) {
        this.vocabularyInvocations += 1;
        const words = await this.classifyWindow(pending);
        this.lastRawTop = this.collectRawScores();
        this.lastSignature = this.collectSignatureProfile();
        const top = words[0] ?? null;
        const word = this.windowStabilizer.accept(top);
        if (word) this.append(word, frame.timestampMs);
        this.lastVeto = this.attributeVeto(top, word);
        this.windowStabilizer.release();
      }

      this.emit(live, frame);
    } finally {
      this.busy = false;
    }
  }

  private async classifyFrame(window: readonly LandmarkFrame[]) {
    return this.runAll('frame', window);
  }

  private async classifyWindow(window: readonly LandmarkFrame[]) {
    return this.runAll('window', window);
  }

  private async runAll(
    granularity: 'frame' | 'window',
    window: readonly LandmarkFrame[],
  ): Promise<readonly SignCandidate[]> {
    const engines = this.classifiers.filter(
      (engine) => engine.granularity === granularity && engine.isReady(),
    );
    const results = await Promise.all(engines.map((engine) => engine.classify(window)));
    return results.flat().sort(byConfidenceDescending);
  }

  private append(candidate: SignCandidate, atMs: number): void {
    // Counted apart: a panel reading "2 words" while the vocabulary was vetoed every time
    // says the alphabet spoke, and conflating them hid exactly that.
    if (candidate.source === 'alphabet') this.lettersEmitted += 1;
    else this.wordsEmitted += 1;
    this.transcript = this.transcript.append({
      text: candidate.gloss.text,
      source: candidate.source,
      confidence: candidate.confidence,
      atMs,
    });
  }

  /**
   * Which floor rejected a completed sign.
   *
   * An empty `classify` means the engine's own floor took it — unless the engine abstained,
   * which is the model answering "nobody is signing" and not a near miss. A candidate that
   * survived the floor and still produced no word was stopped by the stabiliser's, a different
   * and higher number. Anything else is the already-emitted latch.
   */
  private attributeVeto(
    top: SignCandidate | null,
    emitted: SignCandidate | null,
  ): WindowVeto | null {
    if (emitted) return null;
    if (!top) return this.windowAbstained() ? 'abstention' : 'classifier';
    return top.confidence < this.windowStabilizer.threshold ? 'stabilizer' : 'duplicate';
  }

  private windowAbstained(): boolean {
    return this.classifiers.some(
      (engine) => engine.granularity === 'window' && engine.lastAbstained === true,
    );
  }

  private collectRawScores(): readonly RawScore[] {
    for (const engine of this.classifiers) {
      if (engine.granularity === 'window' && engine.lastScores?.length) return engine.lastScores;
    }
    return [];
  }

  private collectSignatureProfile(): SignatureProfile | null {
    for (const engine of this.classifiers) {
      if (engine.granularity === 'window' && engine.lastSignatureProfile) {
        return engine.lastSignatureProfile;
      }
    }
    return null;
  }

  private resetDiagnostics(): void {
    this.framesSeen = 0;
    this.framesWithHands = 0;
    this.windowsClosed = 0;
    this.lastWindowFrames = 0;
    this.vocabularyInvocations = 0;
    this.lastRawTop = [];
    this.lastVeto = null;
    this.wordsEmitted = 0;
    this.lettersEmitted = 0;
    this.lastSignature = null;
    this.shortWindowsAtStart = this.segmenter.discardedShortWindows;
  }

  private get diagnostics(): RecognitionDiagnostics {
    return {
      framesSeen: this.framesSeen,
      framesWithHands: this.framesWithHands,
      windowsClosed: this.windowsClosed,
      windowsTooShort: this.segmenter.discardedShortWindows - this.shortWindowsAtStart,
      lastWindowFrames: this.lastWindowFrames,
      segmenterActive: this.segmenter.isActive,
      pendingFrames: this.segmenter.pendingFrames,
      vocabularyReady: this.classifiers.some(
        (engine) => engine.granularity === 'window' && engine.isReady(),
      ),
      vocabularyInvocations: this.vocabularyInvocations,
      lastRawTop: this.lastRawTop,
      lastVeto: this.lastVeto,
      wordsEmitted: this.wordsEmitted,
      lettersEmitted: this.lettersEmitted,
      lastSignature: this.lastSignature,
      frameCost: this.source.frameCost?.() ?? null,
    };
  }

  private emit(candidates: readonly SignCandidate[], frame: LandmarkFrame): void {
    this.listener?.({
      transcript: this.transcript,
      candidates,
      frame,
      diagnostics: this.diagnostics,
    });
  }
}
