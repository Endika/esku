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
  /** The model's own "nobody is signing" class winning: empty, but not a near miss. */
  abstain = false;
  lastAbstained = false;
  lastScores: readonly { text: string; confidence: number }[] = [];

  isReady(): boolean {
    return true;
  }
  async load(): Promise<void> {}

  async classify(): Promise<readonly SignCandidate[]> {
    this.calls += 1;
    this.lastAbstained = this.abstain;
    if (this.abstain) {
      this.lastScores = [{ text: 'sin signo', confidence: this.confidence }];
      return [];
    }
    this.lastScores = [{ text: 'dolor', confidence: this.confidence }];
    if (this.confidence < this.floor) return [];
    return [{ gloss: createGloss('DOLOR'), confidence: this.confidence, source: 'vocabulary' }];
  }
}

const FRAME_MS = 33;

function movingFrames(count: number, from = 0) {
  return Array.from({ length: count }, (_, i) =>
    buildFrame((from + i) * FRAME_MS, buildHand({ offset: { x: (from + i) * 0.06, y: 0 } })),
  );
}

/**
 * One complete sign: moving, then held still long enough to read as a boundary.
 *
 * The frame count is chosen for *duration* — 40 frames at 33 ms is 1.3 s, past the
 * segmenter's `minSignMs`. Timestamps have to be real and continuous: these frames used to
 * be stamped 0 and the segmenter counted frames, so the lie was invisible. It is not any
 * more, and a collapsed timestamp span now silently means "no sign happened".
 */
function scriptedSign(moving = 40, still = 6) {
  const heldAt = (moving - 1) * 0.06;
  return [
    ...movingFrames(moving),
    ...Array.from({ length: still }, (_, i) =>
      buildFrame((moving + i) * FRAME_MS, buildHand({ offset: { x: heldAt, y: 0 } })),
    ),
  ];
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
    for (const frame of scriptedSign()) source.push(frame);
    expect(vocabulary.calls).toBe(0);

    slow.blocking = false;
    slow.finish();
    await tick();

    expect(vocabulary.calls).toBe(1);
  });

  it('transcribes a recognised sign as a word', async () => {
    await recognize.start(() => {});
    for (const frame of scriptedSign()) source.push(frame);
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
      for (const frame of scriptedSign()) source.push(frame);
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

    it('writes nothing when the model abstains, and says so instead of blaming its floor', async () => {
      // The abstention is an answer, not a near miss, and the shipped model has had one all
      // along: 131 of 1,477 words on held-out signers came out as the literal `__nada__`.
      // Blaming `classifier` here would send anyone reading the panel after a threshold that
      // was never involved.
      vocabulary.abstain = true;
      const diagnostics = await signOnce();

      expect(diagnostics.vocabularyInvocations).toBe(1);
      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('abstention');
    });

    it('blames the stabiliser when the engine answered and the higher floor rejected it', async () => {
      // Two floors, and the higher one decides: 0.45 in the engine, 0.60 here. A sign landing
      // between them is recognised and then silently dropped.
      vocabulary.confidence = 0.47;
      const diagnostics = await signOnce();

      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('stabilizer');
      expect(diagnostics.lastRawTop[0]?.confidence).toBeCloseTo(0.47);
    });

    // The two below pin the window floor at 0.60. It moved up from 0.50 when `minSignMs` came
    // down: on real discourse a third of the windows landing in gaps between annotated
    // sentences got a word written, and 0.60 is what holds that rate flat while the shorter
    // windows recover more signs. Pinned because nothing else in the suite would notice it
    // drifting back, and 0.55 is exactly the confidence that used to be written.
    it('drops a sign just under the window floor', async () => {
      vocabulary.confidence = 0.55;
      const diagnostics = await signOnce();

      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('stabilizer');
    });

    it('writes a sign just over it', async () => {
      vocabulary.confidence = 0.65;
      const diagnostics = await signOnce();

      expect(diagnostics.wordsEmitted).toBe(1);
      expect(diagnostics.lastVeto).toBeNull();
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
    for (const frame of scriptedSign()) source.push(frame);

    recognize.stop();
    slow.blocking = false;
    slow.finish();
    await tick();

    expect(vocabulary.calls).toBe(0);
  });
});
