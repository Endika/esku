import { CameraUnavailableError } from '@domain/landmarks/services/ILandmarkSource';
import { ModelFileError } from '@infrastructure/recognition/modelFiles';
import { SignatureLayoutMismatchError } from '@infrastructure/recognition/VocabularySignClassifier';
import {
  missingTrackersNotice,
  startFailureMessage,
  untrackedParts,
} from '@presentation/engineNotices';
import { describe, expect, it } from 'vitest';

describe('engine notices', () => {
  it('says nothing when every tracker came up', () => {
    expect(missingTrackersNotice({ pose: true, face: true })).toBeNull();
    expect(untrackedParts({ pose: true, face: true }).size).toBe(0);
  });

  it('names the face when only its model is missing', () => {
    expect(missingTrackersNotice({ pose: true, face: false })).toMatch(/cara/);
    expect([...untrackedParts({ pose: true, face: false })]).toEqual(['face']);
  });

  it('names neck, torso and arms when the body model is missing', () => {
    expect(missingTrackersNotice({ pose: false, face: true })).toMatch(/cuello, torso ni brazos/);
    expect([...untrackedParts({ pose: false, face: true })]).toEqual(['neck', 'torso', 'arms']);
  });

  it('says reading is hands only when both are missing', () => {
    expect(missingTrackersNotice({ pose: false, face: false })).toMatch(/solo con las manos/);
    expect(untrackedParts({ pose: false, face: false }).size).toBe(4);
  });

  it('tells a broken model apart from a missing camera and from anything else', () => {
    const broken = startFailureMessage(new ModelFileError('truncated'));
    const drifted = startFailureMessage(new SignatureLayoutMismatchError(1, 2));
    const camera = startFailureMessage(new CameraUnavailableError());
    const other = startFailureMessage(new Error('boom'));

    expect(broken).toMatch(/Liberar espacio/);
    expect(drifted).toBe(broken);
    expect(camera).toMatch(/cámara/);
    expect(new Set([broken, camera, other]).size).toBe(3);
  });
});
