import { describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import { DEFAULT_SEGMENTER_OPTIONS, SignSegmenter } from '../SignSegmenter';

/** `buildHand`'s knuckle span, the unit every motion measurement is expressed in. */
const PALM_WIDTH = 0.2203;

/**
 * The same physical sign, sampled at a given frame rate.
 *
 * `rates` are palm widths per *second* — a property of the signer, not of the camera. The
 * frame count follows from `fps`, which is exactly the confound being tested: a 1.5 s sign
 * is 30 frames at 20 fps and 12 at 8 fps, and the segmenter must not care.
 */
function signAt(fps: number, durationMs: number, rates: readonly number[]) {
  const step = 1000 / fps;
  const frames = Math.round(durationMs / step);
  let x = 0;
  return Array.from({ length: frames }, (_, i) => {
    const rate = rates[Math.min(Math.floor((i / frames) * rates.length), rates.length - 1)]!;
    x += (rate * PALM_WIDTH) / fps;
    return buildFrame(i * step, buildHand({ offset: { x, y: 0 } }));
  });
}

/**
 * The scripted frames below are 33 ms apart, so these are the old frame-count settings at
 * that spacing: motion 0.08 per frame is 2.42 per second, a 6-frame floor is 198 ms, and a
 * 20-frame cap is 660 ms. Kept equivalent on purpose — the point of the conversion was to
 * change the units, not the tuning.
 */
const OPTIONS = {
  motionRate: 2.42,
  decelerationDrop: 0.45,
  decelerationHoldMs: 33,
  minSignMs: 198,
  minMs: 130,
  minFrames: 4,
  maxMs: 660,
};

/** A window's wall-clock span, which is what the duration caps are now expressed in. */
function spanMs(window: readonly unknown[] | undefined): number {
  const frames = window as readonly { timestampMs: number }[] | undefined;
  if (!frames || frames.length < 2) return 0;
  return frames.at(-1)!.timestampMs - frames[0]!.timestampMs;
}

/**
 * Frames whose per-step travel is scripted, so a test can slow the signer down without
 * stopping them. `buildHand`'s palm is 0.22 wide, so a step of 0.05 reads as motion 0.23
 * and a step of 0.02 as 0.09 — still above `motionRate`, still visibly "signing".
 */
function pacedFrames(steps: readonly number[]) {
  let x = 0;
  return steps.map((step, i) => {
    x += step;
    return buildFrame(i * 33, buildHand({ offset: { x, y: 0 } }));
  });
}

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
    // Measured at 0 windows over 300 frames before the duration cap existed.
    const segmenter = new SignSegmenter(OPTIONS);
    const continuous = Array.from({ length: 200 }, (_, i) =>
      buildFrame(
        i * 33,
        buildHand({ offset: { x: Math.sin(i / 3) * 0.12, y: Math.cos(i / 4) * 0.12 } }),
      ),
    );

    expect(run(segmenter, continuous).length).toBeGreaterThan(0);
  });

  it('caps a forced window at maxMs', () => {
    const segmenter = new SignSegmenter({ ...OPTIONS, maxMs: 400 });
    const continuous = Array.from({ length: 120 }, (_, i) =>
      buildFrame(i * 33, buildHand({ offset: { x: Math.sin(i / 3) * 0.12, y: 0 } })),
    );

    for (const window of run(segmenter, continuous)) {
      // The cap is crossed *by* a frame, so the window may overshoot by one interval.
      expect(spanMs(window)).toBeLessThan(400 + 33);
    }
  });

  it('still prefers stillness as the boundary when the signer does pause', () => {
    // Isolated signs must not be chopped at maxMs when a real boundary exists first.
    const segmenter = new SignSegmenter(OPTIONS);
    const frames = [...movingFrames(8), ...stillFrames(4, 8, 0.35)];
    const [window] = run(segmenter, frames);
    expect(spanMs(window)).toBeLessThan(OPTIONS.maxMs);
  });

  it('never grows a window past maxMs', () => {
    const segmenter = new SignSegmenter({ ...OPTIONS, maxMs: 330 });
    const frames = [...movingFrames(30), ...stillFrames(4, 30, 1.45)];
    const [window] = run(segmenter, frames);
    expect(spanMs(window)).toBeLessThan(330 + 33);
  });

  it('ignores an empty stream', () => {
    const segmenter = new SignSegmenter(OPTIONS);
    expect(run(segmenter, [buildFrame(0, null), buildFrame(33, null)])).toHaveLength(0);
  });

  it('closes on a deceleration, without the signer ever holding still', () => {
    // The failure that made the app read a signing video and write nothing. A fluent signer
    // never stops, so a stillness rule only ever closed at the cap — and with a median
    // sign of 30 frames every fixed-length window straddled a boundary. Measured on spliced
    // continuous streams: 14.6% of signs recovered before this rule, 38.4% after.
    const segmenter = new SignSegmenter(OPTIONS);
    // Fast throughout, then slower — but never below the threshold that counts as signing.
    const frames = pacedFrames([...Array(10).fill(0.05), ...Array(4).fill(0.02)]);

    const closed = run(segmenter, frames);

    expect(closed).toHaveLength(1);
    expect(spanMs(closed[0])).toBeLessThan(OPTIONS.maxMs);
  });

  describe('frame rate', () => {
    // Why this block exists: every threshold was swept against SWL-LSE, which is 20.00 fps
    // in all 300 recordings. The live pipeline runs three MediaPipe models per frame and
    // reaches whatever the device allows. Measured in a real browser, the app saw 23 frames
    // in 16 s, discarded all six windows as too short, and never once asked the vocabulary
    // engine — while the identical build recognised "Dolor" at 68% when the same video was
    // slowed to give the pipeline the dataset's spatial sampling density.
    const SIGN_MS = 1500;
    // Brisk, then decelerating to a third of peak — a boundary any rule should believe.
    const RATES = [3, 3, 3, 3, 1];

    it('closes one window for the same sign at the dataset frame rate', () => {
      const segmenter = new SignSegmenter(DEFAULT_SEGMENTER_OPTIONS);
      expect(run(segmenter, signAt(20, SIGN_MS, RATES))).toHaveLength(1);
    });

    it('closes one window for that same sign on a slower device', () => {
      // 8 fps puts the sign at 12 frames, under the old 24-frame floor, so a frame-counting rule
      // can never close it: the sign is silently swallowed on exactly the phones that need
      // recognition most.
      const segmenter = new SignSegmenter(DEFAULT_SEGMENTER_OPTIONS);
      expect(run(segmenter, signAt(8, SIGN_MS, RATES))).toHaveLength(1);
    });

    it('still registers a slow sign as signing on a fast device', () => {
      // The mirror failure. A per-frame motion floor is a *speed* floor divided by fps, so at
      // 60 fps an unhurried sign moves too little between frames to count as motion at all
      // and the segmenter never activates.
      const segmenter = new SignSegmenter(DEFAULT_SEGMENTER_OPTIONS);
      run(segmenter, signAt(60, 800, [1, 1, 1]));
      expect(segmenter.isActive).toBe(true);
    });

    it('gives the same sign a window of the same duration at any frame rate', () => {
      const at = (fps: number) => {
        const segmenter = new SignSegmenter(DEFAULT_SEGMENTER_OPTIONS);
        const [window] = run(segmenter, signAt(fps, SIGN_MS, RATES));
        const frames = window as readonly { timestampMs: number }[] | undefined;
        if (!frames?.length) return 0;
        return frames.at(-1)!.timestampMs - frames[0]!.timestampMs;
      };

      // Frame counts differ by 2.5x; the wall-clock span of the window must not.
      expect(at(8)).toBeGreaterThan(0);
      expect(at(20)).toBeGreaterThan(0);
      expect(Math.abs(at(8) - at(20))).toBeLessThan(300);
    });
  });

  it('does not cut a sign at its own internal slow-down', () => {
    // Signs decelerate mid-way too — a two-part sign slows at its hinge. Believing that is
    // a boundary halves every sign, and costs 8 points of isolated accuracy.
    const segmenter = new SignSegmenter({ ...OPTIONS, minSignMs: 396 });
    const frames = pacedFrames([0.05, 0.05, 0.02, 0.05, 0.05, 0.05]);

    expect(run(segmenter, frames)).toHaveLength(0);
  });
});
