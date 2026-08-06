import { describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { SignSegmenter } from '../SignSegmenter';

const OPTIONS = { motionThreshold: 0.08, settleFrames: 3, minFrames: 4, maxFrames: 20 };

/** Feeds frames and collects every window the segmenter closes. */
function run(segmenter: SignSegmenter, frames: ReturnType<typeof buildFrame>[]) {
  const closed: readonly unknown[][] = [];
  const out: (readonly unknown[])[] = [...closed];
  frames.forEach((frame) => {
    const window = segmenter.push(frame);
    if (window) out.push(window);
  });
  return out;
}

function movingFrames(count: number, startAt = 0, step = 0.05) {
  return Array.from({ length: count }, (_, i) =>
    buildFrame((startAt + i) * 33, buildHand({ offset: { x: i * step, y: 0 } })),
  );
}

function stillFrames(count: number, startAt = 0, atX = 0) {
  return Array.from({ length: count }, (_, i) =>
    buildFrame((startAt + i) * 33, buildHand({ offset: { x: atX, y: 0 } })),
  );
}

describe('SignSegmenter', () => {
  it('emits nothing while the hand is merely still', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    expect(run(segmenter, stillFrames(20))).toHaveLength(0);
  });

  it('closes a window once movement is followed by stillness', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    const frames = [...movingFrames(8), ...stillFrames(4, 8, 0.35)];
    expect(run(segmenter, frames)).toHaveLength(1);
  });

  it('does not close while the hand is still moving', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    expect(run(segmenter, movingFrames(15))).toHaveLength(0);
  });

  it('separates two signs into two windows', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    const frames = [
      ...movingFrames(8),
      ...stillFrames(4, 8, 0.35),
      ...movingFrames(8, 12).map((f, i) =>
        buildFrame(f.timestampMs, buildHand({ offset: { x: 0.35 + i * 0.05, y: 0 } })),
      ),
      ...stillFrames(4, 20, 0.7),
    ];
    expect(run(segmenter, frames)).toHaveLength(2);
  });

  it('closes the window when the hand leaves the frame mid-sign', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    const frames = [...movingFrames(8), buildFrame(300, null)];
    expect(run(segmenter, frames)).toHaveLength(1);
  });

  it('discards a twitch shorter than minFrames', () => {
    const segmenter = new SignSegmenter({ ...OPTIONS, minFrames: 12 });
    const frames = [...movingFrames(5), ...stillFrames(4, 5, 0.2)];
    expect(run(segmenter, frames)).toHaveLength(0);
  });

  it('counts a discarded twitch, so silence can be told from never ending a sign', () => {
    // Both cases return null from push(). "The sign ended and was judged noise" and "nothing
    // has ended yet" call for opposite fixes, so the panel has to be able to say which.
    const segmenter = new SignSegmenter({ ...OPTIONS, minFrames: 12 });
    run(segmenter, [...movingFrames(5), ...stillFrames(4, 5, 0.2)]);
    expect(segmenter.discardedShortWindows).toBe(1);

    const idle = new SignSegmenter({ ...OPTIONS, minFrames: 12 });
    run(idle, stillFrames(20));
    expect(idle.discardedShortWindows).toBe(0);
  });

  it('reports how much of a sign it is holding while one is in progress', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    run(segmenter, movingFrames(6));
    expect(segmenter.isActive).toBe(true);
    expect(segmenter.pendingFrames).toBe(6);
  });

  it('emits windows while the hand keeps moving, without ever holding still', () => {
    // The failure that made the app look broken on video: a fluent signer never pauses, so
    // waiting for stillness closed nothing at all and the vocabulary engine was never asked.
    // Measured at 0 windows over 300 frames before the maxFrames cap existed.
    const segmenter = new SignSegmenter(OPTIONS);
    const continuous = Array.from({ length: 200 }, (_, i) =>
      buildFrame(
        i * 33,
        buildHand({ offset: { x: Math.sin(i / 3) * 0.12, y: Math.cos(i / 4) * 0.12 } }),
      ),
    );

    expect(run(segmenter, continuous).length).toBeGreaterThan(0);
  });

  it('caps a forced window at maxFrames', () => {
    const segmenter = new SignSegmenter({ ...OPTIONS, maxFrames: 12 });
    const continuous = Array.from({ length: 120 }, (_, i) =>
      buildFrame(i * 33, buildHand({ offset: { x: Math.sin(i / 3) * 0.12, y: 0 } })),
    );

    for (const window of run(segmenter, continuous)) {
      expect(window.length).toBeLessThanOrEqual(12);
    }
  });

  it('still prefers stillness as the boundary when the signer does pause', () => {
    // Isolated signs must not be chopped at maxFrames when a real boundary exists first.
    const segmenter = new SignSegmenter(OPTIONS);
    const frames = [...movingFrames(8), ...stillFrames(4, 8, 0.35)];
    const [window] = run(segmenter, frames);
    expect(window?.length).toBeLessThan(OPTIONS.maxFrames);
  });

  it('never grows a window past maxFrames', () => {
    const segmenter = new SignSegmenter({ ...OPTIONS, maxFrames: 10 });
    const frames = [...movingFrames(30), ...stillFrames(4, 30, 1.45)];
    const [window] = run(segmenter, frames);
    expect(window?.length).toBeLessThanOrEqual(10);
  });

  it('ignores an empty stream', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    expect(run(segmenter, [buildFrame(0, null), buildFrame(33, null)])).toHaveLength(0);
  });
});
