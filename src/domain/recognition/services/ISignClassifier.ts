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
/**
 * `frame` engines read a held pose and are asked on every frame — a fingerspelled letter is
 * a shape you hold. `window` engines read a whole movement and are asked only once a sign
 * has visibly ended. Routing on this is the difference between spelling a letter as you hold
 * it and waiting for a sign to finish.
 */
export type Granularity = 'frame' | 'window';

export interface ISignClassifier {
  readonly id: string;
  readonly granularity: Granularity;
  /** Whether this engine can answer right now (weights loaded, prototypes present…). */
  isReady(): boolean;
  /** Loads whatever the engine needs. Safe to call twice. */
  load(): Promise<void>;
  classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]>;
}
