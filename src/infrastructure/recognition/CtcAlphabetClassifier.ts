import { DominantHandTracker } from '@domain/landmarks/services/dominantHandTracker';
import { normalizeHand } from '@domain/landmarks/services/normalizeHand';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import type { ISignClassifier } from '@domain/recognition/services/ISignClassifier';
import {
  byConfidenceDescending,
  type SignCandidate,
} from '@domain/recognition/value-objects/Gloss';
import type { RawScore } from '@domain/recognition/value-objects/RecognitionDiagnostics';
import { type GruDirection, gruStep, matrix } from './gru';

/** What `lsefs_train.py --export` writes beside the weights. */
export interface AlphabetManifest {
  readonly hidden: number;
  readonly layers: number;
  readonly inputs: number;
  readonly classes: number;
  readonly blank: number;
  readonly letters: readonly string[];
  readonly order: readonly string[];
  readonly shapes: Readonly<Record<string, readonly number[]>>;
}

const MAX_CANDIDATES = 3;

/**
 * Letters the reader should not trust, for either of two reasons.
 *
 * **Measured weak**, recall on LSE-FS-UVigo's held-out signers: Y 17%, J 32%, V 43%, Z 44%.
 * **Too rare to judge**, and so not vouched for: K appears 3 times in the test split, Ñ 8,
 * W 10 — those percentages would be one or two words either way.
 *
 * Both cases come from the same place, the training set: K appears 16 times in 18,887
 * characters, W 23, Ñ 28, Y 45. The rest of the alphabet runs 74-98%.
 *
 * This is a *measurement of the shipped weights*, not a property of the handshapes. Retrain
 * and re-derive it from block 7 of `tools/bench/alphabet.bench.ts` rather than copying it
 * forward — the previous weights had Q and X in this list and they are now 55% and 79%.
 */
export const WEAK_LETTERS = ['j', 'k', 'ñ', 'v', 'w', 'y', 'z'];

export class AlphabetLayoutMismatchError extends Error {
  constructor(declared: number, expected: number) {
    super(`The alphabet model expects ${declared} inputs per frame, the app builds ${expected}`);
    this.name = 'AlphabetLayoutMismatchError';
  }
}

/** 21 normalised points, and deliberately not the wrist's position: a letter is a letter anywhere. */
const FRAME_INPUTS = 63;

/**
 * Reads fingerspelled letters with a GRU trained on real continuous fingerspelling.
 *
 * Replaces a hand-written handshape table that, measured against LSE-FS-UVigo, wrote 17% of the
 * letters and spelled none of 456 words. Two things the table could not do and this can:
 *
 * - **Say "no letter here".** Trained with CTC, whose blank class is exactly that. A geometric
 *   table can only rank letters by similarity, so a hand mid-transition between two letters
 *   scores badly everywhere and is indistinguishable from a hand the table simply does not know.
 * - **Use what came before.** A GRU carries hidden state, so this is an automaton rather than a
 *   pure function of one frame — which is why `reset()` exists and why the caller must use it
 *   when a session starts or the hand leaves frame. State carried across a pause describes a
 *   past that is no longer there.
 *
 * The forward pass shares `gruStep` with the vocabulary engine rather than reimplementing it.
 */
export class CtcAlphabetClassifier implements ISignClassifier {
  readonly id = 'lse-alphabet-ctc';
  readonly granularity = 'frame' as const;

  #manifest: AlphabetManifest | null = null;
  #layers: GruDirection[] = [];
  #head: { weight: Float32Array; bias: Float32Array } | null = null;
  #state: Float32Array[] = [];
  readonly #hand = new DominantHandTracker();
  #scores: readonly RawScore[] = [];

  constructor(
    private readonly manifestUrl = 'models/lse-alphabet.json',
    private readonly weightsUrl = 'models/lse-alphabet.bin',
  ) {}

  isReady(): boolean {
    return this.#manifest !== null;
  }

  async load(): Promise<void> {
    if (this.#manifest) return;
    const [manifest, blob] = await Promise.all([
      fetch(this.manifestUrl).then((r) => r.json() as Promise<AlphabetManifest>),
      fetch(this.weightsUrl).then((r) => r.arrayBuffer()),
    ]);
    await this.loadFrom(manifest, blob);
  }

  /** Split out so tests and the parity fixture can drive it without a network. */
  async loadFrom(manifest: AlphabetManifest, blob: ArrayBuffer): Promise<void> {
    // Fail loudly rather than predict noise: a layout drift between trainer and app does not
    // throw on its own, it just returns confident nonsense.
    if (manifest.inputs !== FRAME_INPUTS) {
      throw new AlphabetLayoutMismatchError(manifest.inputs, FRAME_INPUTS);
    }

    const floats = new Float32Array(blob);
    const tensors = new Map<string, Float32Array>();
    let offset = 0;
    for (const name of manifest.order) {
      const size = (manifest.shapes[name] ?? []).reduce((a, b) => a * b, 1);
      tensors.set(name, floats.subarray(offset, offset + size));
      offset += size;
    }

    const gates = 3 * manifest.hidden;
    this.#layers = Array.from({ length: manifest.layers }, (_, layer) => {
      const inputs = layer === 0 ? manifest.inputs : manifest.hidden;
      return {
        weightIh: matrix(gates, inputs, tensors.get(`gru.weight_ih_l${layer}`)!),
        weightHh: matrix(gates, manifest.hidden, tensors.get(`gru.weight_hh_l${layer}`)!),
        biasIh: tensors.get(`gru.bias_ih_l${layer}`)!,
        biasHh: tensors.get(`gru.bias_hh_l${layer}`)!,
      };
    });
    this.#head = { weight: tensors.get('head.weight')!, bias: tensors.get('head.bias')! };
    this.#manifest = manifest;
    this.reset();
  }

  /** Clear the hidden state. Call on a new session, and when the hand leaves frame. */
  reset(): void {
    const hidden = this.#manifest?.hidden ?? 0;
    this.#state = this.#layers.map(() => new Float32Array(hidden));
    this.#scores = [];
    // The hand history goes too: which hand was signing before a pause says nothing about
    // which is signing after it, and a stale score would pick the wrong one for several frames.
    this.#hand.reset();
  }

  get lastScores(): readonly RawScore[] {
    return this.#scores;
  }

  async classify(window: readonly LandmarkFrame[]): Promise<readonly SignCandidate[]> {
    const manifest = this.#manifest;
    const head = this.#head;
    if (!manifest || !head) return [];

    const frame = window.at(-1);
    const hand = frame ? this.#hand.pick(frame) : null;
    if (!hand) {
      // Not a shape to classify. Stepping the GRU on zeros would advance the automaton on
      // nothing, letting a pause change what the next real frame is read as.
      this.#scores = [];
      return [];
    }

    let signal = new Float32Array(FRAME_INPUTS);
    normalizeHand(hand).forEach((point, i) => {
      signal[i * 3] = point.x;
      signal[i * 3 + 1] = point.y;
      signal[i * 3 + 2] = point.z;
    });

    this.#layers.forEach((layer, i) => {
      const next = gruStep(signal, this.#state[i]!, layer, manifest.hidden);
      this.#state[i] = next;
      signal = next;
    });

    const probabilities = softmax(affine(head.weight, signal, head.bias, manifest.classes));
    this.#scores = Array.from(probabilities, (confidence, i) => ({
      text: i === manifest.blank ? 'blank' : (manifest.letters[i - 1] ?? '?'),
      confidence,
    })).sort((a, b) => b.confidence - a.confidence);

    // Blank winning is an answer, not an absence of one: the model is saying this frame is
    // between letters. Offering the runner-up here would throw away the one thing a trained
    // abstention buys over a handshape table.
    if (this.#scores[0]?.text === 'blank') return [];

    return this.#scores
      .filter((score) => score.text !== 'blank')
      .slice(0, MAX_CANDIDATES)
      .map((score) => ({
        gloss: { id: score.text, conceptId: score.text, text: score.text },
        confidence: score.confidence,
        source: 'alphabet' as const,
      }))
      .sort(byConfidenceDescending);
  }
}

function affine(
  weight: Float32Array,
  input: Float32Array,
  bias: Float32Array,
  rows: number,
): Float32Array {
  const cols = input.length;
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r += 1) {
    let sum = bias[r] ?? 0;
    for (let c = 0; c < cols; c += 1) sum += (weight[r * cols + c] ?? 0) * (input[c] ?? 0);
    out[r] = sum;
  }
  return out;
}

function softmax(logits: Float32Array): Float32Array {
  let top = Number.NEGATIVE_INFINITY;
  for (const value of logits) if (value > top) top = value;
  const out = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i += 1) {
    out[i] = Math.exp((logits[i] ?? 0) - top);
    total += out[i]!;
  }
  for (let i = 0; i < out.length; i += 1) out[i] = (out[i] ?? 0) / total;
  return out;
}
