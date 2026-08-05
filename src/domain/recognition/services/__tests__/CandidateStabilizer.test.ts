import { describe, expect, it } from 'vitest';
import { createGloss, type SignCandidate } from '../../value-objects/Gloss';
import { CandidateStabilizer } from '../CandidateStabilizer';

function candidate(label: string, confidence: number): SignCandidate {
  return { gloss: createGloss(label), confidence, source: 'vocabulary' };
}

describe('CandidateStabilizer', () => {
  it('waits for the agreement streak before emitting', () => {
    const stabilizer = new CandidateStabilizer(3, 0.6);
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).toBeNull();
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).toBeNull();
    expect(stabilizer.accept(candidate('DOLOR', 0.9))?.gloss.conceptId).toBe('DOLOR');
  });

  it('never emits a candidate below the confidence floor', () => {
    const stabilizer = new CandidateStabilizer(2, 0.6);
    expect(stabilizer.accept(candidate('DOLOR', 0.5))).toBeNull();
    expect(stabilizer.accept(candidate('DOLOR', 0.5))).toBeNull();
    expect(stabilizer.accept(candidate('DOLOR', 0.5))).toBeNull();
  });

  it('resets the streak when the prediction flickers to another sign', () => {
    const stabilizer = new CandidateStabilizer(3, 0.6);
    stabilizer.accept(candidate('DOLOR', 0.9));
    stabilizer.accept(candidate('DOLOR', 0.9));
    stabilizer.accept(candidate('FIEBRE2', 0.9));
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).toBeNull();
  });

  it('does not repeat a held sign every frame', () => {
    const stabilizer = new CandidateStabilizer(2, 0.6);
    stabilizer.accept(candidate('DOLOR', 0.9));
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).not.toBeNull();
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).toBeNull();
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).toBeNull();
  });

  it('allows the same sign again after a release', () => {
    const stabilizer = new CandidateStabilizer(2, 0.6);
    stabilizer.accept(candidate('DOLOR', 0.9));
    stabilizer.accept(candidate('DOLOR', 0.9));
    stabilizer.release();
    stabilizer.accept(candidate('DOLOR', 0.9));
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).not.toBeNull();
  });

  it('treats variants of one concept as agreement, not flicker', () => {
    // The model wobbling between AZUCAR and AZUCAR2 is still confidently saying "sugar".
    const stabilizer = new CandidateStabilizer(3, 0.6);
    stabilizer.accept(candidate('AZUCAR', 0.9));
    stabilizer.accept(candidate('AZUCAR2', 0.9));
    expect(stabilizer.accept(candidate('AZUCAR(2M)', 0.9))?.gloss.conceptId).toBe('AZUCAR');
  });

  it('clears the streak when nothing is detected', () => {
    const stabilizer = new CandidateStabilizer(2, 0.6);
    stabilizer.accept(candidate('DOLOR', 0.9));
    stabilizer.accept(null);
    expect(stabilizer.accept(candidate('DOLOR', 0.9))).toBeNull();
  });
});
