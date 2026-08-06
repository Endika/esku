import type { ILandmarkSource, LandmarkListener } from '@domain/landmarks/services/ILandmarkSource';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import { createGloss, type SignCandidate } from '@domain/recognition/value-objects/Gloss';
import { beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { type RecognitionUpdate, RecognizeSignsUseCase } from '../RecognizeSignsUseCase';

/** Drives frames by hand instead of waiting on a camera. */
class ScriptedSource implements ILandmarkSource {
  private listener: LandmarkListener | null = null;

  async start(listener: LandmarkListener): Promise<void> {
    this.listener = listener;
  }
  stop(): void {
    this.listener = null;
  }
  isRunning(): boolean {
    return this.listener !== null;
  }
  push(frame: LandmarkFrame): void {
    this.listener?.(frame);
  }
}

/**
 * The per-frame engine, held open on demand.
 *
 * This is what actually blocks in the app: it runs on every frame, and on a phone the
 * MediaPipe passes behind it are slow enough that a sign routinely finishes mid-call.
 */
class SlowFrameClassifier implements ISignClassifier {
  readonly id = 'slow-frame';
  readonly granularity = 'frame' as const;
  blocking = false;
  private release: (() => void) | null = null;

  isReady(): boolean {
    return true;
  }
  async load(): Promise<void> {}

  async classify(): Promise<readonly SignCandidate[]> {
    if (this.blocking) {
      await new Promise<void>((resolve) => {
        this.release = resolve;
      });
    }
    return [];
  }

  finish(): void {
    this.release?.();
    this.release = null;
  }
}

/**
 * The vocabulary engine, counting how often it is actually consulted.
 *
 * `confidence` is what it answers with, and `lastScores` mirrors the real engine: the raw
 * softmax survives even when the engine's own floor leaves `classify` empty.
 */
class CountingWindowClassifier implements ISignClassifier {
  readonly id = 'window';
  readonly granularity = 'window' as const;
  calls = 0;
  confidence = 0.9;
  /** Below this the real engine returns nothing at all, keeping only the raw score. */
  floor = 0.45;
  lastScores: readonly { text: string; confidence: number }[] = [];

  isReady(): boolean {
    return true;
  }
  async load(): Promise<void> {}

  async classify(): Promise<readonly SignCandidate[]> {
    this.calls += 1;
    this.lastScores = [{ text: 'dolor', confidence: this.confidence }];
    if (this.confidence < this.floor) return [];
    return [{ gloss: createGloss('DOLOR'), confidence: this.confidence, source: 'vocabulary' }];
  }
}

function movingFrames(count: number, from = 0) {
  return Array.from({ length: count }, (_, i) =>
    buildFrame((from + i) * 33, buildHand({ offset: { x: (from + i) * 0.06, y: 0 } })),
  );
}

function stillFrames(count: number, at: number) {
  return Array.from({ length: count }, () => buildFrame(0, buildHand({ offset: { x: at, y: 0 } })));
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('RecognizeSignsUseCase', () => {
  let source: ScriptedSource;
  let slow: SlowFrameClassifier;
  let vocabulary: CountingWindowClassifier;
  let recognize: RecognizeSignsUseCase;

  beforeEach(() => {
    source = new ScriptedSource();
    slow = new SlowFrameClassifier();
    vocabulary = new CountingWindowClassifier();
    recognize = new RecognizeSignsUseCase(source, [slow, vocabulary]);
  });

  it('does not lose a sign that finishes while the per-frame engine is busy', async () => {
    // The bug this exists for: a window closes on exactly one frame, and that frame arriving
    // mid-classification used to hit `if (busy) return` and be discarded outright. On a
    // phone running three MediaPipe models per frame that is the common case, not the rare
    // one — and the whole sign was lost silently. The app looked simply dead.
    await recognize.start(() => {});

    slow.blocking = true;
    source.push(buildFrame(0, buildHand()));
    await tick();

    // The sign now completes entirely while that first call is still in flight.
    for (const frame of [...movingFrames(20), ...stillFrames(14, 1.2)]) source.push(frame);
    expect(vocabulary.calls).toBe(0);

    slow.blocking = false;
    slow.finish();
    await tick();

    expect(vocabulary.calls).toBe(1);
  });

  it('transcribes a recognised sign as a word', async () => {
    await recognize.start(() => {});
    for (const frame of [...movingFrames(20), ...stillFrames(14, 1.2)]) source.push(frame);
    await tick();

    expect(recognize.current.toText().toLowerCase()).toContain('dolor');
  });

  describe('diagnostics', () => {
    /** Runs one complete sign and returns what the panel would be showing afterwards. */
    async function signOnce() {
      let last: RecognitionUpdate | null = null;
      await recognize.start((update) => {
        last = update;
      });
      for (const frame of [...movingFrames(20), ...stillFrames(14, 1.2)]) source.push(frame);
      await tick();
      return last!.diagnostics;
    }

    it('separates a window never closing from an engine never asked', async () => {
      // The app draws a correct skeleton and writes nothing in both cases. Offline metrics
      // cannot tell them apart, which is how three previous diagnoses went wrong.
      let last: RecognitionUpdate | null = null;
      await recognize.start((update) => {
        last = update;
      });
      for (const frame of movingFrames(3)) source.push(frame);
      await tick();

      expect(last!.diagnostics.windowsClosed).toBe(0);
      expect(last!.diagnostics.vocabularyInvocations).toBe(0);
      expect(last!.diagnostics.segmenterActive).toBe(true);
    });

    it('reports the engine being asked, and what it answered', async () => {
      const diagnostics = await signOnce();

      expect(diagnostics.windowsClosed).toBe(1);
      expect(diagnostics.vocabularyInvocations).toBe(1);
      expect(diagnostics.lastWindowFrames).toBeGreaterThan(0);
      expect(diagnostics.wordsEmitted).toBe(1);
      expect(diagnostics.lastVeto).toBeNull();
    });

    it('blames the engine floor when nothing reached it', async () => {
      vocabulary.confidence = 0.2;
      const diagnostics = await signOnce();

      expect(diagnostics.vocabularyInvocations).toBe(1);
      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('classifier');
      // The number the panel exists to show: without it, 0.2 and "never classified" look the
      // same from outside, and they need opposite fixes.
      expect(diagnostics.lastRawTop[0]?.confidence).toBeCloseTo(0.2);
    });

    it('blames the stabiliser when the engine answered and the higher floor rejected it', async () => {
      // The two floors are different numbers — 0.45 in the engine, 0.55 here. A sign landing
      // between them is recognised and then silently dropped.
      vocabulary.confidence = 0.5;
      const diagnostics = await signOnce();

      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('stabilizer');
      expect(diagnostics.lastRawTop[0]?.confidence).toBeCloseTo(0.5);
    });

    it('starts a new session from zero rather than carrying the last one over', async () => {
      await signOnce();
      recognize.stop();
      const diagnostics = await signOnce();

      expect(diagnostics.windowsClosed).toBe(1);
    });
  });

  it('drops the queued sign on stop, so it cannot surface in the next session', async () => {
    await recognize.start(() => {});
    slow.blocking = true;
    source.push(buildFrame(0, buildHand()));
    await tick();
    for (const frame of [...movingFrames(20), ...stillFrames(14, 1.2)]) source.push(frame);

    recognize.stop();
    slow.blocking = false;
    slow.finish();
    await tick();

    expect(vocabulary.calls).toBe(0);
  });
});
