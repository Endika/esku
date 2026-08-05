import { describe, expect, it } from 'vitest';
import { handednessFor } from '../MediaPipeLandmarkSource';

describe('handednessFor', () => {
  it('reads the selfie view as mirrored', () => {
    // What MediaPipe calls "Left" in a mirrored frame is the signer's right hand.
    expect(handednessFor('Left', 'user')).toBe('right');
    expect(handednessFor('Right', 'user')).toBe('left');
  });

  it('reads the rear camera as-is', () => {
    expect(handednessFor('Right', 'environment')).toBe('right');
    expect(handednessFor('Left', 'environment')).toBe('left');
  });

  it('inverts between the two cameras for the same label', () => {
    // The whole point: one label, two answers. If these ever agree, the fix has been undone.
    expect(handednessFor('Left', 'user')).not.toBe(handednessFor('Left', 'environment'));
  });

  it('falls back to left when the label is missing rather than throwing', () => {
    expect(handednessFor(undefined, 'user')).toBe('left');
    expect(handednessFor(undefined, 'environment')).toBe('left');
  });
});
