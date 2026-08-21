import { PosePoint } from '@domain/landmarks/value-objects/BodyLandmarks';
import { describe, expect, it } from 'vitest';
import { handednessFor, mirroringFrom } from '../MediaPipeLandmarkSource';

/** A pose with only the two shoulders placed, which is all the mirroring test reads. */
function shoulders(leftX: number, rightX: number): { x: number }[] {
  const pose: { x: number }[] = [];
  pose[PosePoint.leftShoulder] = { x: leftX };
  pose[PosePoint.rightShoulder] = { x: rightX };
  return pose;
}

describe('mirroringFrom', () => {
  it('reads a frame as direct when the anatomical left shoulder is on the viewer right', () => {
    // Facing the lens, your left shoulder is on the viewer's right, so it carries the larger x.
    expect(mirroringFrom(shoulders(0.7, 0.3))).toBe('direct');
  });

  it('reads a frame as mirrored when that order is reversed', () => {
    expect(mirroringFrom(shoulders(0.3, 0.7))).toBe('mirrored');
  });

  it('declines to answer when the signer is turned too far to tell', () => {
    // Side-on, the shoulders collapse onto each other and their order is noise.
    expect(mirroringFrom(shoulders(0.5, 0.48))).toBe('unknown');
  });

  it('declines to answer without a pose, rather than assuming one', () => {
    expect(mirroringFrom(undefined)).toBe('unknown');
    expect(mirroringFrom([])).toBe('unknown');
  });
});

describe('handednessFor', () => {
  it("takes MediaPipe's label at face value on a direct frame", () => {
    // The measured default. `getUserMedia` hands over the sensor's frames — the selfie mirror is
    // a CSS convention on the preview — so MediaPipe's label is anatomical, front camera included.
    expect(handednessFor('Right', false)).toBe('right');
    expect(handednessFor('Left', false)).toBe('left');
  });

  it('inverts the label only when the frame really is mirrored', () => {
    expect(handednessFor('Right', true)).toBe('left');
    expect(handednessFor('Left', true)).toBe('right');
  });

  it('puts a right hand in the right slot on a front camera, which it did not', () => {
    // The bug this replaces, reproduced: on a phone's front camera a right hand landed in the
    // left slot, because the code assumed selfie meant mirrored. That is not a swap — the
    // signature reflects left hands into the right hand's space, so the model was handed a
    // mirror image of the handshape.
    const frame = mirroringFrom(shoulders(0.68, 0.32));
    expect(frame).toBe('direct');
    expect(handednessFor('Right', frame === 'mirrored')).toBe('right');
  });

  it('falls back to left when the label is missing rather than throwing', () => {
    expect(handednessFor(undefined, false)).toBe('left');
    expect(handednessFor(undefined, true)).toBe('left');
  });
});
