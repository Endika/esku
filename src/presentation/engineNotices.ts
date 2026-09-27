import { CameraUnavailableError } from '@domain/landmarks/services/ILandmarkSource';
import { AlphabetLayoutMismatchError } from '@infrastructure/recognition/CtcAlphabetClassifier';
import { ModelFileError } from '@infrastructure/recognition/modelFiles';
import {
  AbstentionUndeclaredError,
  SignatureLayoutMismatchError,
} from '@infrastructure/recognition/VocabularySignClassifier';
import type { BodyPart } from '@presentation/components/LandmarkOverlay';

export interface TrackerAvailability {
  readonly pose: boolean;
  readonly face: boolean;
}

/** The parts no model is watching, so their chips must not read as merely out of frame. */
export function untrackedParts(available: TrackerAvailability): ReadonlySet<BodyPart> {
  const parts = new Set<BodyPart>();
  if (!available.face) parts.add('face');
  if (!available.pose) for (const part of ['neck', 'torso', 'arms'] as const) parts.add(part);
  return parts;
}

export function missingTrackersNotice(available: TrackerAvailability): string | null {
  const retry = 'Recarga para reintentarlo.';
  if (!available.pose && !available.face) {
    return `No cargaron los modelos de cara y cuerpo: leo solo con las manos y fallaré más. ${retry}`;
  }
  if (!available.face) return `No cargó el modelo de la cara: leo sin ella y fallaré más. ${retry}`;
  if (!available.pose) {
    return `No cargó el modelo del cuerpo: leo sin cuello, torso ni brazos y fallaré más. ${retry}`;
  }
  return null;
}

export function startFailureMessage(error: unknown): string {
  if (error instanceof CameraUnavailableError) {
    return 'No hay cámara o se denegó el permiso. Revísalo en los ajustes del navegador.';
  }
  if (
    error instanceof ModelFileError ||
    error instanceof SignatureLayoutMismatchError ||
    error instanceof AbstentionUndeclaredError ||
    error instanceof AlphabetLayoutMismatchError
  ) {
    return (
      'El modelo de signos está incompleto o no es de esta versión. Recarga la página; si sigue ' +
      'igual, usa «Liberar espacio» en Herramientas y vuelve a descargarlo.'
    );
  }
  return 'No se pudo iniciar el reconocimiento.';
}
