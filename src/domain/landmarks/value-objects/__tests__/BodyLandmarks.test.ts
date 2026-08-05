import { describe, expect, it } from 'vitest';
import { isVisible, neckSegment, PosePoint } from '../BodyLandmarks';
import type { Landmark } from '../Landmark';

/** A pose where everything is confidently seen unless overridden. */
function buildPose(overrides: Record<number, Partial<Landmark>> = {}): Landmark[] {
  return Array.from({ length: 33 }, (_, i) => ({
    x: 0.5,
    y: 0.5,
    z: 0,
    visibility: 0.95,
    ...overrides[i],
  }));
}

describe('isVisible', () => {
  it('accepts a confidently seen landmark', () => {
    expect(isVisible({ x: 0, y: 0, z: 0, visibility: 0.9 })).toBe(true);
  });

  it('rejects a landmark the model only guessed at', () => {
    // Framing a signer from the chest up leaves the hips invisible, and MediaPipe answers
    // with extrapolated coordinates rather than nothing.
    expect(isVisible({ x: 0, y: 0, z: 0, visibility: 0.1 })).toBe(false);
  });

  it('trusts landmarks from estimators that report no visibility at all', () => {
    // Hands and face do not report it; treating absent as invisible would erase them.
    expect(isVisible({ x: 0, y: 0, z: 0 })).toBe(true);
  });

  it('rejects a missing landmark', () => {
    expect(isVisible(undefined)).toBe(false);
  });
});

describe('neckSegment', () => {
  it('runs from the middle of the shoulders toward the head', () => {
    const pose = buildPose({
      [PosePoint.leftShoulder]: { x: 0.4, y: 0.6 },
      [PosePoint.rightShoulder]: { x: 0.6, y: 0.6 },
      [PosePoint.nose]: { x: 0.5, y: 0.2 },
    });
    const segment = neckSegment(pose);

    expect(segment?.[0].x).toBeCloseTo(0.5, 5);
    expect(segment?.[0].y).toBeCloseTo(0.6, 5);
    // Upward, and stopping short of the nose so it does not run into the face mesh.
    expect(segment?.[1].y).toBeLessThan(0.6);
    expect(segment?.[1].y).toBeGreaterThan(0.2);
  });

  it('is one segment, not a triangle across the chest', () => {
    // Drawing nose-to-each-shoulder was the obvious shortcut and looked nothing like a neck.
    const segment = neckSegment(buildPose());
    expect(segment).toHaveLength(2);
  });

  it('declines when a shoulder is not really visible', () => {
    const pose = buildPose({ [PosePoint.rightShoulder]: { visibility: 0.1 } });
    expect(neckSegment(pose)).toBeNull();
  });

  it('declines when the head is not really visible', () => {
    const pose = buildPose({ [PosePoint.nose]: { visibility: 0.2 } });
    expect(neckSegment(pose)).toBeNull();
  });

  it('declines on an empty pose rather than throwing', () => {
    expect(neckSegment([])).toBeNull();
  });
});
