import type { ILandmarkSource, LandmarkListener } from '@domain/landmarks/services/ILandmarkSource';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import { createGloss, type SignCandidate } from '@domain/recognition/value-objects/Gloss';
import { beforeEach, describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { RecognizeSignsUseCase } from '../RecognizeSignsUseCase';

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

/** The vocabulary engine, counting how often it is actually consulted. */
class CountingWindowClassifier implements ISignClassifier {
  readonly id = 'window';
  readonly granularity = 'window' as const;
  calls = 0;

  isReady(): boolean {
    return true;
  }
  async load(): Promise<void> {}

  async classify(): Promise<readonly SignCandidate[]> {
    this.calls += 1;
    return [{ gloss: createGloss('DOLOR'), confidence: 0.9, source: 'vocabulary' }];
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
