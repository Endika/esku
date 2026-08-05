/**
 * A sign the user taught the app themselves.
 *
 * Stored as prototype feature vectors rather than as retrained weights: matching by
 * nearest-prototype needs a handful of examples and no training loop, where fine-tuning a
 * network in the browser would need far more of both. It is also the only path that lets
 * someone add a sign in a language we ship no model for.
 */
export interface CustomSign {
  readonly id: string;
  readonly text: string;
  /** One entry per recorded example, each already normalised for scale and handedness. */
  readonly prototypes: readonly Float32Array[];
  readonly createdAtMs: number;
}

/** Below this, nearest-prototype matching is too unreliable to be worth offering. */
export const MIN_PROTOTYPES_PER_SIGN = 3;

export class NotEnoughExamplesError extends Error {
  constructor(given: number) {
    super(`A taught sign needs at least ${MIN_PROTOTYPES_PER_SIGN} examples, got ${given}`);
    this.name = 'NotEnoughExamplesError';
  }
}

export class DuplicateSignTextError extends Error {
  constructor(text: string) {
    super(`A sign called "${text}" already exists`);
    this.name = 'DuplicateSignTextError';
  }
}
