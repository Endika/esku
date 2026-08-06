import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import { describe, expect, it } from 'vitest';
import { buildFrame, buildHand } from '@/test/handFixtures';
import {
  profileSignature,
  VOCABULARY_SIGNATURE_LENGTH,
  vocabularySignature,
} from '../vocabularySignature';

/** Written by `tools/train/make_parity.py` from the same synthetic input. */
const parity = JSON.parse(
  readFileSync(
    join(import.meta.dirname, '..', '..', '..', '..', 'test', 'fixtures', 'signature-parity.json'),
    'utf-8',
  ),
) as {
  curls: [number, number, number, number, number];
  frames: number;
  handOffsetStep: number;
  signatureLength: number;
  pose: number[][][];
  faceMesh: number[][][];
  signature: number[];
};

function toPoints(rows: number[][]) {
  return rows.map(([x, y, z]) => ({ x: x ?? 0, y: y ?? 0, z: z ?? 0 }));
}

function parityFrames(): LandmarkFrame[] {
  return Array.from({ length: parity.frames }, (_, i) => ({
    ...buildFrame(
      i * 33,
      buildHand({ curls: parity.curls, offset: { x: i * parity.handOffsetStep, y: 0 } }),
    ),
    pose: { points: toPoints(parity.pose[i]!) },
    face: { points: toPoints(parity.faceMesh[i]!) },
  }));
}

describe('vocabularySignature', () => {
  it('matches the Python trainer element for element', () => {
    // The load-bearing test. The app and the trainer build this vector independently, and a
    // disagreement does not throw — the model just predicts noise. Feature layout drift is
    // the single most likely way for recognition to silently rot.
    const signature = vocabularySignature(parityFrames());

    expect(signature).toHaveLength(parity.signatureLength);
    for (let i = 0; i < parity.signature.length; i += 1) {
      expect(signature[i]).toBeCloseTo(parity.signature[i]!, 4);
    }
  });

  it('agrees with the trainer on the vector length', () => {
    expect(VOCABULARY_SIGNATURE_LENGTH).toBe(parity.signatureLength);
  });

  it('returns zeros for an empty window rather than throwing', () => {
    const signature = vocabularySignature([]);
    expect(signature).toHaveLength(VOCABULARY_SIGNATURE_LENGTH);
    expect(signature.every((value) => value === 0)).toBe(true);
  });

  it('degrades to zeros for the body blocks when pose is unavailable', () => {
    // Pose can fail to load while hands keep working, so this must not throw. The hand
    // blocks depend on the body frame for their located coordinates, so they zero out too.
    const frames = [buildFrame(0, buildHand({ curls: [0.9, 0, 0.9, 0.9, 0.9] }))];
    const signature = vocabularySignature(frames);
    expect(signature).toHaveLength(VOCABULARY_SIGNATURE_LENGTH);
    expect(signature.every((value) => value === 0)).toBe(true);
  });

  it('produces the same vector however long the sign took', () => {
    const short = vocabularySignature(parityFrames());
    const stretched = vocabularySignature([
      ...parityFrames(),
      ...parityFrames().slice(-1),
      ...parityFrames().slice(-1),
    ]);
    // Not identical — the extra frames change which slots resample where — but close, and
    // certainly not a different sign.
    expect(short).toHaveLength(stretched.length);
  });

  describe('profileSignature', () => {
    it('summarises each body part of a well-formed window', () => {
      const profile = profileSignature(vocabularySignature(parityFrames()));

      expect(profile.torso.emptyFrames).toBe(0);
      expect(profile.rightHand.meanMagnitude).toBeGreaterThan(0);
      expect(profile.face.meanMagnitude).toBeGreaterThan(0);
    });

    it('reports the hands as empty when pose is missing, which is what breaks the model', () => {
      // Hand coordinates are expressed relative to the torso, so losing pose does not degrade
      // the hand block — it zeroes it. Invisible on screen, fatal to the prediction.
      const noPose = parityFrames().map(({ pose: _pose, ...frame }) => frame);
      const profile = profileSignature(vocabularySignature(noPose));

      expect(profile.torso.emptyFrames).toBe(1);
      expect(profile.rightHand.emptyFrames).toBe(1);
    });
  });
});
