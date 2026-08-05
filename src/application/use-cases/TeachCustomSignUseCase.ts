import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import {
  type CustomSign,
  DuplicateSignTextError,
  MIN_PROTOTYPES_PER_SIGN,
  NotEnoughExamplesError,
} from '@domain/recognition/entities/CustomSign';
import type { ICustomSignRepository } from '@domain/recognition/repositories/ICustomSignRepository';
import { windowSignature } from '@domain/recognition/services/windowSignature';

export class EmptySignTextError extends Error {
  constructor() {
    super('A taught sign needs a word to write when it is recognised');
    this.name = 'EmptySignTextError';
  }
}

/**
 * Turns a handful of recordings of one sign into something the app can recognise.
 *
 * Clock and id generator are injected so a test can assert on exact stored values instead of
 * whatever the wall clock happened to say.
 */
export class TeachCustomSignUseCase {
  constructor(
    private readonly repository: ICustomSignRepository,
    private readonly now: () => number = () => Date.now(),
    private readonly newId: () => string = () => crypto.randomUUID(),
  ) {}

  async execute(
    text: string,
    examples: readonly (readonly LandmarkFrame[])[],
  ): Promise<CustomSign> {
    const trimmed = text.trim();
    if (trimmed.length === 0) throw new EmptySignTextError();

    // Empty recordings are dropped first: three attempts where two caught no hand is not
    // three examples, and accepting them would produce a sign that matches almost anything.
    const usable = examples.filter((example) => example.some((frame) => frame.hands.length > 0));
    if (usable.length < MIN_PROTOTYPES_PER_SIGN) {
      throw new NotEnoughExamplesError(usable.length);
    }

    const existing = await this.repository.findAll();
    if (existing.some((sign) => sign.text.toLowerCase() === trimmed.toLowerCase())) {
      throw new DuplicateSignTextError(trimmed);
    }

    const sign: CustomSign = {
      id: this.newId(),
      text: trimmed,
      prototypes: usable.map(windowSignature),
      createdAtMs: this.now(),
    };

    await this.repository.save(sign);
    return sign;
  }
}
