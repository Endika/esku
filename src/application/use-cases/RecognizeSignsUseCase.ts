import type { ILandmarkSource } from '@domain/landmarks/services/ILandmarkSource';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import { CandidateStabilizer } from '@domain/recognition/services/CandidateStabilizer';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import { SignSegmenter } from '@domain/recognition/services/SignSegmenter';
import {
  byConfidenceDescending,
  type SignCandidate,
} from '@domain/recognition/value-objects/Gloss';
import { Transcript } from '@domain/transcript/entities/Transcript';

export interface RecognitionUpdate {
  readonly transcript: Transcript;
  /** What the engines are currently seeing, best first. For the live hint, not the text. */
  readonly candidates: readonly SignCandidate[];
  /** The frame these candidates came from, so the UI can draw what was tracked. */
  readonly frame: LandmarkFrame;
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
  private readonly frameStabilizer = new CandidateStabilizer();
  private readonly windowStabilizer = new CandidateStabilizer(1, 0.55);
  private transcript = new Transcript();
  private listener: RecognitionListener | null = null;
  /** Guards against overlapping async classify calls piling up behind a slow frame. */
  private busy = false;
  /** Set while teaching: the next completed sign is handed over instead of transcribed. */
  private capture: ((window: readonly LandmarkFrame[]) => void) | null = null;

  constructor(
    private readonly source: ILandmarkSource,
    private readonly classifiers: readonly ISignClassifier[],
  ) {}

  async start(listener: RecognitionListener): Promise<void> {
    this.listener = listener;
    await this.source.start((frame) => void this.onFrame(frame));
  }

  stop(): void {
    this.source.stop();
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
    if (frame.hands.length === 0) {
      // A hand leaving frame is a deliberate boundary: it lets the same letter repeat.
      this.frameStabilizer.release();
    }

    const closedWindow = this.segmenter.push(frame);

    if (this.capture) {
      // Teaching: hand the finished sign over, and transcribe nothing meanwhile — the user
      // is demonstrating a sign, not dictating.
      if (closedWindow) {
        const deliver = this.capture;
        this.capture = null;
        deliver(closedWindow);
      }
      this.emit([], frame);
      return;
    }

    if (this.busy) return;
    this.busy = true;
    try {
      const live = await this.classifyFrame([frame]);
      const accepted = this.frameStabilizer.accept(live[0] ?? null);
      if (accepted) this.append(accepted, frame.timestampMs);

      if (closedWindow) {
        const words = await this.classifyWindow(closedWindow);
        const word = this.windowStabilizer.accept(words[0] ?? null);
        if (word) this.append(word, frame.timestampMs);
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
    this.transcript = this.transcript.append({
      text: candidate.gloss.text,
      source: candidate.source,
      confidence: candidate.confidence,
      atMs,
    });
  }

  private emit(candidates: readonly SignCandidate[], frame: LandmarkFrame): void {
    this.listener?.({ transcript: this.transcript, candidates, frame });
  }
}
