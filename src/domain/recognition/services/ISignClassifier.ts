import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { SignCandidate } from '../value-objects/Gloss';

/**
 * The single port every recognition engine implements, so the application layer never
 * knows whether an answer came from a trained ONNX head, a geometric handshape table or
 * the user's own taught examples.
 *
 * `classify` takes a window rather than a frame because dynamic signs only exist over
 * time; a static-handshape engine is free to look at the last frame alone.
 */
export interface ISignClassifier {
  readonly id: string;
  /** Whether this engine can answer right now (weights loaded, prototypes present…). */
  isReady(): boolean;
  /** Loads whatever the engine needs. Safe to call twice. */
  load(): Promise<void>;
  classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]>;
}
