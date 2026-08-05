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
