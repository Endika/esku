import { describe, expect, it } from 'vitest';
import { FrameCostMeter } from '../FrameCost';

describe('FrameCostMeter', () => {
  it('reads nothing before a frame has been measured', () => {
    expect(new FrameCostMeter().read()).toBeNull();
  });

  it('reports the median of each model, not the mean', () => {
    const meter = new FrameCostMeter();
    meter.record(10, 5, 2);
    meter.record(12, 6, 3);
    // One frame landing on a garbage collection must not move the reading
    meter.record(400, 200, 100);

    expect(meter.read()).toEqual({
      handsMs: 12,
      poseMs: 6,
      faceMs: 3,
      faceEvery: 1,
      samples: 3,
    });
  });

  it('averages the two middle samples when the count is even', () => {
    const meter = new FrameCostMeter();
    meter.record(10, 4, 1);
    meter.record(20, 6, 3);

    expect(meter.read()).toMatchObject({ handsMs: 15, poseMs: 5, faceMs: 2 });
  });

  it('keeps only the most recent frames, so a slow start stops counting', () => {
    const meter = new FrameCostMeter(4);
    for (const ms of [100, 100, 100, 100]) meter.record(ms, ms, ms);
    for (const ms of [10, 10, 10, 10]) meter.record(ms, ms, ms);

    expect(meter.read()).toEqual({
      handsMs: 10,
      poseMs: 10,
      faceMs: 10,
      faceEvery: 1,
      samples: 4,
    });
  });

  it('counts a throttled face pass without charging it to every frame', () => {
    const meter = new FrameCostMeter();
    // The face detector runs on its own clock, so most frames report no face cost at all.
    // Averaging zeros into it would hide what one pass costs; charging every frame would
    // invent a cost that was never paid. Both are wrong, so they are reported apart.
    meter.record(10, 5, 40);
    meter.record(10, 5, null);
    meter.record(10, 5, null);
    meter.record(10, 5, null);

    expect(meter.read()).toEqual({
      handsMs: 10,
      poseMs: 5,
      faceMs: 40,
      faceEvery: 4,
      samples: 4,
    });
  });

  it('keeps the face ratio honest once the window saturates', () => {
    // The reading that exposed the bug: 495 frames on a phone reported "every 2" for a detector
    // running every 8, because a saturated hands array was divided by an unsaturated face one.
    const meter = new FrameCostMeter(8);
    for (let frame = 0; frame < 40; frame += 1) {
      meter.record(70, 38, frame % 8 === 0 ? 37 : null);
    }

    expect(meter.read()).toMatchObject({ faceMs: 37, faceEvery: 8, samples: 8 });
  });

  it('reports no face cost when the detector never ran, rather than pretending', () => {
    const meter = new FrameCostMeter();
    meter.record(10, 5, null);

    expect(meter.read()).toMatchObject({ faceMs: 0, faceEvery: 0 });
  });

  it('forgets everything on reset, since the cameras do not cost the same', () => {
    const meter = new FrameCostMeter();
    meter.record(10, 5, 2);
    meter.reset();

    expect(meter.read()).toBeNull();
  });
});
