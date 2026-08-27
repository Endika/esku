import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import {
  profileSignature,
  VOCABULARY_SIGNATURE_LENGTH,
  vocabularySignature,
} from '@domain/recognition/services/vocabularySignature';
import {
  byConfidenceDescending,
  createGloss,
  type SignCandidate,
} from '@domain/recognition/value-objects/Gloss';
import type {
  RawScore,
  SignatureProfile,
} from '@domain/recognition/value-objects/RecognitionDiagnostics';
import {
  affine,
  type GruDirection,
  gruPass,
  layerNorm,
  type Matrix,
  matrix,
  meanOverTime,
  mergeDirections,
  relu,
  softmax,
} from './gru';

/**
 * The engine's own floor, and *not* the one that decides: `RecognizeSignsUseCase` applies a
 * higher one on top of it. This lower number is what keeps the two distinguishable in the
 * diagnostics panel — under it the engine says nothing at all, between the two the stabiliser
 * is what rejected the sign, and those need opposite fixes. 0.30 is also the figure the
 * continuous benchmark reports against, where 88-96% of windows never reached it before the
 * model had seen co-articulation.
 */
const MIN_CONFIDENCE = 0.3;

const MAX_CANDIDATES = 3;

/** How the abstention reads in the diagnostics panel; `__nada__` is not for human eyes. */
const ABSTENTION_TEXT = 'sin signo';

export interface VocabularyManifest {
  readonly concepts: string[];
  /**
   * The class that means "nobody is signing right now", declared by the trainer so this file
   * never has to hardcode the label. `null` when the model has no such class.
   */
  readonly abstentionConcept: string | null;
  readonly signatureLength: number;
  readonly frames: number;
  readonly hidden: number;
  readonly layers: number;
  readonly order: string[];
  readonly shapes: Record<string, number[]>;
  readonly testTop1: number;
  readonly testTop3: number;
}

/**
 * A concept the app would write into a transcript, that the model never meant as a word.
 *
 * The shipped model has carried `__NADA__` since the co-articulated retrain and nothing here
 * knew: 131 of 1,477 words on held-out signers came out as `__nada__`. Reserved concepts are
 * therefore refused unless the manifest says which one is the abstention.
 */
export class AbstentionUndeclaredError extends Error {
  constructor(concepts: readonly string[]) {
    super(
      `The model lists reserved concepts (${concepts.join(', ')}) without declaring one as ` +
        'abstentionConcept, so a class meaning "nobody is signing" would be written as a word.',
    );
    this.name = 'AbstentionUndeclaredError';
  }
}

export class SignatureLayoutMismatchError extends Error {
  constructor(expected: number, actual: number) {
    super(
      `Model expects a ${expected}-float signature but this build produces ${actual}. ` +
        'src/domain/recognition/services/vocabularySignature.ts and ' +
        'tools/train/vocabulary_features.py have drifted apart.',
    );
    this.name = 'SignatureLayoutMismatchError';
  }
}

/**
 * The trained LSE vocabulary model. Its concept list comes from the manifest, never from a
 * number written here: it was 238 concepts from SWL-LSE alone, and became 287 once LSE-Health's
 * co-articulated annotations were added — 286 signs and one trained "nobody is signing".
 *
 * Reads whole signs, so the segmenter decides when to ask it. Weights arrive as one flat
 * float32 blob described by a manifest, and are sliced up here in the order the trainer
 * wrote them.
 */
export class VocabularySignClassifier implements ISignClassifier {
  readonly id = 'lse-vocabulary';
  readonly granularity = 'window' as const;

  private manifest: VocabularyManifest | null = null;
  private tensors: Map<string, Float32Array> | null = null;
  private rawTop: readonly RawScore[] = [];
  private profile: SignatureProfile | null = null;
  private abstained = false;

  constructor(
    private readonly manifestUrl: string,
    private readonly weightsUrl: string,
  ) {}

  /** The last window's best concepts with `MIN_CONFIDENCE` not applied. See ISignClassifier. */
  get lastScores(): readonly RawScore[] {
    return this.rawTop;
  }

  /** What the last window looked like as features, per body part. */
  get lastSignatureProfile(): SignatureProfile | null {
    return this.profile;
  }

  /** Whether the last window came back empty because the model said nobody was signing. */
  get lastAbstained(): boolean {
    return this.abstained;
  }

  isReady(): boolean {
    return this.tensors !== null;
  }

  get accuracy(): { top1: number; top3: number } | null {
    return this.manifest ? { top1: this.manifest.testTop1, top3: this.manifest.testTop3 } : null;
  }

  async load(): Promise<void> {
    if (this.tensors) return;

    const [manifest, blob] = await Promise.all([
      fetch(this.manifestUrl).then((r) => r.json() as Promise<VocabularyManifest>),
      fetch(this.weightsUrl).then((r) => r.arrayBuffer()),
    ]);

    // Fail loudly here rather than predicting noise later. A signature-layout drift between
    // the app and the trainer produces confident nonsense, which is far harder to notice.
    if (manifest.signatureLength !== VOCABULARY_SIGNATURE_LENGTH) {
      throw new SignatureLayoutMismatchError(manifest.signatureLength, VOCABULARY_SIGNATURE_LENGTH);
    }

    const undeclared = manifest.concepts.filter(
      (concept) => concept.startsWith('__') && concept !== manifest.abstentionConcept,
    );
    if (undeclared.length > 0) throw new AbstentionUndeclaredError(undeclared);

    const floats = new Float32Array(blob);
    const tensors = new Map<string, Float32Array>();
    let offset = 0;
    for (const name of manifest.order) {
      const size = (manifest.shapes[name] ?? []).reduce((a, b) => a * b, 1);
      tensors.set(name, floats.subarray(offset, offset + size));
      offset += size;
    }

    this.manifest = manifest;
    this.tensors = tensors;
  }

  async classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]> {
    const manifest = this.manifest;
    const tensors = this.tensors;
    if (!manifest || !tensors || window.length === 0) return [];

    const signature = vocabularySignature(window);
    this.profile = profileSignature(signature);
    const probabilities = softmax(this.forward(signature, manifest, tensors));

    const scored = manifest.concepts
      .map((concept, i) => ({
        gloss: createGloss(concept),
        confidence: probabilities[i] ?? 0,
        source: 'vocabulary' as const,
      }))
      .sort(byConfidenceDescending);

    // Kept before the floor is applied: a window that scored 0.44 and one that was never
    // classified both leave `classify` empty, and they need opposite fixes. The abstention is
    // kept here too, because it is the only place the choice of silence is visible.
    this.rawTop = scored.slice(0, MAX_CANDIDATES).map(({ gloss, confidence }) => ({
      text: this.isAbstention(gloss.id) ? ABSTENTION_TEXT : gloss.text,
      confidence,
    }));

    // The model's own "nobody is signing" class won, so nothing is said — not even the
    // runner-up, which would contradict the answer the model just gave. Measured on held-out
    // signers, this alone takes words written into pauses from 27.5% to 10.9% of the windows
    // that land there, with continuous recall unmoved at 38.1%.
    this.abstained = scored.length > 0 && this.isAbstention(scored[0]!.gloss.id);
    if (this.abstained) return [];

    return scored
      .filter(({ gloss }) => !this.isAbstention(gloss.id))
      .slice(0, MAX_CANDIDATES)
      .filter(({ confidence }) => confidence >= MIN_CONFIDENCE);
  }

  private isAbstention(label: string): boolean {
    return this.manifest?.abstentionConcept === label;
  }

  private forward(
    signature: Float32Array,
    manifest: VocabularyManifest,
    tensors: Map<string, Float32Array>,
  ): Float32Array {
    const width = manifest.signatureLength / manifest.frames;
    const get = (name: string) => tensors.get(name)!;
    const shaped = (name: string): Matrix => {
      const [rows, cols] = manifest.shapes[name] ?? [0, 0];
      return matrix(rows ?? 0, cols ?? 0, get(name));
    };

    let sequence: Float32Array[] = [];
    for (let frame = 0; frame < manifest.frames; frame += 1) {
      const slice = signature.subarray(frame * width, (frame + 1) * width);
      sequence.push(layerNorm(slice, get('norm.weight'), get('norm.bias')));
    }

    for (let layer = 0; layer < manifest.layers; layer += 1) {
      const forward: GruDirection = {
        weightIh: shaped(`gru.weight_ih_l${layer}`),
        weightHh: shaped(`gru.weight_hh_l${layer}`),
        biasIh: get(`gru.bias_ih_l${layer}`),
        biasHh: get(`gru.bias_hh_l${layer}`),
      };
      const backward: GruDirection = {
        weightIh: shaped(`gru.weight_ih_l${layer}_reverse`),
        weightHh: shaped(`gru.weight_hh_l${layer}_reverse`),
        biasIh: get(`gru.bias_ih_l${layer}_reverse`),
        biasHh: get(`gru.bias_hh_l${layer}_reverse`),
      };
      sequence = mergeDirections(
        gruPass(sequence, forward, manifest.hidden, false),
        gruPass(sequence, backward, manifest.hidden, true),
      );
    }

    const pooled = meanOverTime(sequence);
    const hiddenLayer = relu(affine(shaped('head.0.weight'), pooled, get('head.0.bias')));
    return affine(shaped('head.3.weight'), hiddenLayer, get('head.3.bias'));
  }
}
