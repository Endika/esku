import type { LandmarkFrame } from '../value-objects/LandmarkFrame';

export type LandmarkListener = (frame: LandmarkFrame) => void;

/**
 * Port over "something that produces hand landmarks over time" — the camera in production,
 * a scripted frame list in tests. Keeping the camera behind this is what makes the whole
 * recognition pipeline testable in a jsdom environment with no webcam.
 */
export interface ILandmarkSource {
  start(listener: LandmarkListener): Promise<void>;
  stop(): void;
  isRunning(): boolean;
}

export class CameraUnavailableError extends Error {
  constructor(cause?: unknown) {
    super('No camera available or permission denied');
    this.name = 'CameraUnavailableError';
    this.cause = cause;
  }
}
