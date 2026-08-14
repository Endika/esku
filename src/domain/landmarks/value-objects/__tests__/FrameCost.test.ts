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

    expect(meter.read()).toEqual({ handsMs: 12, poseMs: 6, faceMs: 3, samples: 3 });
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

    expect(meter.read()).toEqual({ handsMs: 10, poseMs: 10, faceMs: 10, samples: 4 });
  });

  it('forgets everything on reset, since the cameras do not cost the same', () => {
    const meter = new FrameCostMeter();
    meter.record(10, 5, 2);
    meter.reset();

    expect(meter.read()).toBeNull();
  });
});
