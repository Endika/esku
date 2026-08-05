import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { CustomSign } from '@domain/recognition/entities/CustomSign';
import type { ICustomSignRepository } from '@domain/recognition/repositories/ICustomSignRepository';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import {
  SIGNATURE_LENGTH,
  similarity,
  windowSignature,
} from '@domain/recognition/services/windowSignature';
import {
  byConfidenceDescending,
  type SignCandidate,
} from '@domain/recognition/value-objects/Gloss';

/** Below this a "nearest" prototype is just the least-bad of a bad set. */
const MIN_SIMILARITY = 0.86;

const MAX_CANDIDATES = 3;

/**
 * Recognises signs the user taught, by nearest prototype.
 *
 * Nearest-prototype rather than fine-tuning a network in the browser: matching needs a
 * handful of examples and no training loop, where gradient descent would need far more of
 * both and a much longer wait on a phone. It is also the only engine that works for a sign
 * language we ship no model for — the user supplies the data.
 *
 * A sign scores as its *best* matching example, not its average. Recordings of one sign vary
 * (a hand higher, a beat slower); averaging would punish a sign for having varied examples,
 * which is exactly the variety that makes it robust.
 */
export class PrototypeSignClassifier implements ISignClassifier {
  readonly id = 'taught-signs';
  readonly granularity = 'window' as const;

  private signs: CustomSign[] = [];
  private loaded = false;

  constructor(private readonly repository: ICustomSignRepository) {}

  isReady(): boolean {
    return this.loaded && this.signs.length > 0;
  }

  async load(): Promise<void> {
    const stored = await this.repository.findAll();

    // Signatures gained the wrist position when the trained vocabulary model landed, so
    // prototypes recorded before that are a different length and can never match. Dropping
    // them here means the sign simply disappears from the list and can be re-taught, rather
    // than sitting there looking fine and never firing.
    this.signs = stored.filter((sign) =>
      sign.prototypes.every((prototype) => prototype.length === SIGNATURE_LENGTH),
    );
    this.loaded = true;
  }

  /** Call after teaching or deleting, so the live engine sees the change immediately. */
  async refresh(): Promise<void> {
    this.loaded = false;
    await this.load();
  }

  async classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]> {
    if (!this.isReady() || window.length === 0) return [];

    const signature = windowSignature(window);
    return this.signs
      .map((sign) => ({ sign, score: bestMatch(signature, sign) }))
      .filter(({ score }) => score >= MIN_SIMILARITY)
      .map(({ sign, score }) => ({
        gloss: { id: sign.id, conceptId: sign.id, text: sign.text },
        confidence: score,
        source: 'taught' as const,
      }))
      .sort(byConfidenceDescending)
      .slice(0, MAX_CANDIDATES);
  }
}

function bestMatch(signature: Float32Array, sign: CustomSign): number {
  let best = 0;
  for (const prototype of sign.prototypes) {
    const score = similarity(signature, prototype);
    if (score > best) best = score;
  }
  return best;
}
