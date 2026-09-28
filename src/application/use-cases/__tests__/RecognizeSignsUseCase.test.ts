import type { ILandmarkSource, LandmarkListener } from '@domain/landmarks/services/ILandmarkSource';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import { createGloss, type SignCandidate } from '@domain/recognition/value-objects/Gloss';
import { beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import {
  CaptureCancelledError,
  type RecognitionUpdate,
  RecognizeSignsUseCase,
} from '../RecognizeSignsUseCase';

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
  floor = 0.3;
  /** The model's own "nobody is signing" class winning: empty, but not a near miss. */
  abstain = false;
  lastAbstained = false;
  lastScores: readonly { text: string; confidence: number }[] = [];
  blocking = false;
  private release: (() => void) | null = null;

  isReady(): boolean {
    return true;
  }
  async load(): Promise<void> {}

  async classify(): Promise<readonly SignCandidate[]> {
    this.calls += 1;
    if (this.blocking) {
      await new Promise<void>((resolve) => {
        this.release = resolve;
      });
    }
    this.lastAbstained = this.abstain;
    if (this.abstain) {
      this.lastScores = [{ text: 'sin signo', confidence: this.confidence }];
      return [];
    }
    this.lastScores = [{ text: 'dolor', confidence: this.confidence }];
    if (this.confidence < this.floor) return [];
    return [{ gloss: createGloss('DOLOR'), confidence: this.confidence, source: 'vocabulary' }];
  }

  finish(): void {
    this.release?.();
    this.release = null;
  }
}

/** A vocabulary engine that throws while `broken`, the way a bad tensor or a lost GPU does. */
class BreakableWindowClassifier extends CountingWindowClassifier {
  broken = true;

  override async classify(): Promise<readonly SignCandidate[]> {
    if (this.broken) {
      this.calls += 1;
      throw new Error('tensor dolor.weight is missing');
    }
    return super.classify();
  }
}

/** The alphabet engine: a letter on every frame with a hand, and a record of being reset. */
class LetterClassifier implements ISignClassifier {
  readonly id = 'letters';
  readonly granularity = 'frame' as const;
  calls = 0;
  resets = 0;

  isReady(): boolean {
    return true;
  }
  async load(): Promise<void> {}
  reset(): void {
    this.resets += 1;
  }

  async classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]> {
    this.calls += 1;
    if (window[0]?.hands.length === 0) return [];
    return [{ gloss: createGloss('A'), confidence: 0.9, source: 'alphabet' }];
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

/** A second sign, well after the first, so the segmenter reads it as its own. */
function laterSign() {
  return scriptedSign().map((frame) => ({ ...frame, timestampMs: frame.timestampMs + 5000 }));
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('RecognizeSignsUseCase', () => {
  let source: ScriptedSource;
  let vocabulary: CountingWindowClassifier;
  let recognize: RecognizeSignsUseCase;

  beforeEach(() => {
    source = new ScriptedSource();
    vocabulary = new CountingWindowClassifier();
    recognize = new RecognizeSignsUseCase(source, [vocabulary]);
  });

  it('does not lose a sign that finishes while the previous one is still being read', async () => {
    // A window closes on exactly one frame, and that frame arriving mid-classification used
    // to hit `if (busy) return` and be discarded outright — the whole sign lost silently.
    await recognize.start(() => {});

    vocabulary.blocking = true;
    for (const frame of scriptedSign()) source.push(frame);
    await tick();
    expect(vocabulary.calls).toBe(1);

    for (const frame of laterSign()) source.push(frame);
    vocabulary.blocking = false;
    vocabulary.finish();
    await tick();
    source.push(buildFrame(9000, buildHand()));
    await tick();

    expect(vocabulary.calls).toBe(2);
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
      // Two floors, and the higher one decides: 0.30 in the engine, 0.45 here. A sign landing
      // between them is recognised and then silently dropped.
      vocabulary.confidence = 0.35;
      const diagnostics = await signOnce();

      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('stabilizer');
      expect(diagnostics.lastRawTop[0]?.confidence).toBeCloseTo(0.35);
    });

    // The two below pin the window floor at 0.45. It came down from 0.60 once the model's own
    // abstention stopped being written as a word: over half the words this app wrote into
    // pauses were the `__NADA__` class itself, so honouring it took pause babble from 30.0%
    // to 13.3% and paid for a lower gate. 0.45 buys 37.3% → 44.2% of signs written correctly
    // on continuous signing, all four seeds gaining, at 19.9% babble. Pinned because nothing
    // else in the suite would notice it drifting, in either direction.
    it('drops a sign just under the window floor', async () => {
      vocabulary.confidence = 0.44;
      const diagnostics = await signOnce();

      expect(diagnostics.wordsEmitted).toBe(0);
      expect(diagnostics.lastVeto).toBe('stabilizer');
    });

    it('writes a sign just over it', async () => {
      vocabulary.confidence = 0.46;
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

  describe('when an engine throws', () => {
    let breakable: BreakableWindowClassifier;
    let updates: RecognitionUpdate[];

    beforeEach(async () => {
      breakable = new BreakableWindowClassifier();
      recognize = new RecognizeSignsUseCase(source, [breakable]);
      updates = [];
      await recognize.start((update) => updates.push(update));
      for (const frame of scriptedSign()) source.push(frame);
      await tick();
    });

    it('says so to the listener instead of going quiet', () => {
      const last = updates.at(-1)!;

      expect(last.failing).toBe(true);
      expect(last.diagnostics.framesFailed).toBe(1);
      expect(last.diagnostics.lastFailure).toMatch(/dolor.weight/);
    });

    it('keeps reading, and clears the failure once a sign goes through', async () => {
      breakable.broken = false;
      for (const frame of laterSign()) source.push(frame);
      await tick();

      const last = updates.at(-1)!;
      expect(last.failing).toBe(false);
      expect(last.diagnostics.lastFailure).toBeNull();
      expect(last.diagnostics.framesFailed).toBe(1);
      expect(recognize.current.toText().toLowerCase()).toContain('dolor');
    });
  });

  describe('capturing a sign to teach', () => {
    it('rejects the pending capture when it is cancelled, so the caller is not left waiting', async () => {
      await recognize.start(() => {});
      const capture = recognize.captureWindow();

      recognize.cancelCapture();

      await expect(capture).rejects.toBeInstanceOf(CaptureCancelledError);
      expect(recognize.isCapturing).toBe(false);
    });

    it('rejects the pending capture when the camera stops', async () => {
      await recognize.start(() => {});
      const capture = recognize.captureWindow();

      recognize.stop();

      await expect(capture).rejects.toBeInstanceOf(CaptureCancelledError);
    });

    it('still hands over the next completed sign', async () => {
      await recognize.start(() => {});
      const capture = recognize.captureWindow();

      for (const frame of scriptedSign()) source.push(frame);

      expect((await capture).length).toBeGreaterThan(0);
    });
  });

  it('drops the queued sign on stop, so it cannot surface in the next session', async () => {
    await recognize.start(() => {});
    vocabulary.blocking = true;
    for (const frame of scriptedSign()) source.push(frame);
    await tick();
    for (const frame of laterSign()) source.push(frame);

    recognize.stop();
    vocabulary.blocking = false;
    vocabulary.finish();
    await recognize.start(() => {});
    source.push(buildFrame(9000, buildHand()));
    await tick();

    expect(vocabulary.calls).toBe(1);
  });

  describe('spelling', () => {
    let letters: LetterClassifier;

    beforeEach(() => {
      letters = new LetterClassifier();
      recognize = new RecognizeSignsUseCase(source, [letters, vocabulary]);
    });

    it('writes no letters while reading words, and never asks the alphabet', async () => {
      await recognize.start(() => {});
      for (const frame of scriptedSign()) source.push(frame);
      await tick();

      expect(letters.calls).toBe(0);
      expect(recognize.current.toText()).toBe('Dolor');
    });

    it('writes letters and no words while spelling', async () => {
      await recognize.start(() => {});
      recognize.setSpelling(true);
      for (const frame of scriptedSign()) {
        source.push(frame);
        await tick();
      }

      expect(vocabulary.calls).toBe(0);
      expect(recognize.current.toText()).toBe('A');
    });

    it('starts the alphabet afresh each time spelling is switched on', async () => {
      recognize.setSpelling(true);
      recognize.setSpelling(false);
      recognize.setSpelling(true);

      expect(letters.resets).toBe(3);
    });

    it('drops a sign half-read by the vocabulary when spelling starts', async () => {
      await recognize.start(() => {});
      const [moving, still] = [scriptedSign().slice(0, 40), scriptedSign().slice(40)];
      for (const frame of moving) source.push(frame);
      recognize.setSpelling(true);
      recognize.setSpelling(false);
      for (const frame of still) source.push(frame);
      await tick();

      expect(vocabulary.calls).toBe(0);
    });
  });
});
