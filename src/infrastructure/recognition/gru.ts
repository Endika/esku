/**
 * The arithmetic of the trained vocabulary network, by hand.
 *
 * There is no inference runtime here on purpose: onnxruntime-web needs 13 MB of WASM before
 * it can run a 2.4 MB model, and on GitHub Pages it cannot even use threads because Pages
 * sends no COOP/COEP headers. The network is a fixed stack — LayerNorm, a two-layer
 * bidirectional GRU, mean over time, two linear layers — so it is cheaper to write the
 * dozen lines of algebra than to ship a general-purpose engine to run them.
 *
 * Every formula below matches PyTorch's definition exactly, including its gate ordering.
 * `VocabularySignClassifier.test.ts` checks the whole stack against logits produced by the
 * Python model for a fixed input, so a subtle mistake fails a test rather than quietly
 * degrading recognition.
 */

/** A row-major matrix, as PyTorch stores `nn.Linear` and GRU weights. */
export interface Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float32Array;
}

export function matrix(rows: number, cols: number, data: Float32Array): Matrix {
  return { rows, cols, data };
}

/** out = W · x + b, with `b` optional. */
export function affine(w: Matrix, x: Float32Array, b?: Float32Array): Float32Array {
  const out = new Float32Array(w.rows);
  for (let row = 0; row < w.rows; row += 1) {
    let sum = b ? (b[row] ?? 0) : 0;
    const base = row * w.cols;
    for (let col = 0; col < w.cols; col += 1) {
      sum += (w.data[base + col] ?? 0) * (x[col] ?? 0);
    }
    out[row] = sum;
  }
  return out;
}

export function layerNorm(
  x: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  epsilon = 1e-5,
): Float32Array {
  let mean = 0;
  for (let i = 0; i < x.length; i += 1) mean += x[i] ?? 0;
  mean /= x.length;

  let variance = 0;
  for (let i = 0; i < x.length; i += 1) {
    const delta = (x[i] ?? 0) - mean;
    variance += delta * delta;
  }
  variance /= x.length;

  const scale = 1 / Math.sqrt(variance + epsilon);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    out[i] = ((x[i] ?? 0) - mean) * scale * (weight[i] ?? 0) + (bias[i] ?? 0);
  }
  return out;
}

export function relu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) out[i] = Math.max(0, x[i] ?? 0);
  return out;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

export interface GruDirection {
  readonly weightIh: Matrix;
  readonly weightHh: Matrix;
  readonly biasIh: Float32Array;
  readonly biasHh: Float32Array;
}

/**
 * One GRU pass over a sequence, in one direction.
 *
 * PyTorch's gate layout is reset, update, new — in that order within the stacked weight —
 * and crucially the reset gate multiplies the *hidden* contribution to `n` only, after its
 * bias is added. Applying it to the summed input-plus-hidden term instead is the classic
 * way to get a network that runs, produces plausible numbers, and is wrong.
 */
export function gruPass(
  sequence: readonly Float32Array[],
  direction: GruDirection,
  hidden: number,
  reverse: boolean,
): Float32Array[] {
  const outputs: Float32Array[] = new Array(sequence.length);
  let state = new Float32Array(hidden);

  for (let step = 0; step < sequence.length; step += 1) {
    const index = reverse ? sequence.length - 1 - step : step;
    state = gruStep(sequence[index]!, state, direction, hidden);
    outputs[index] = state;
  }

  return outputs;
}

/**
 * A single GRU timestep: the loop body above, exposed so a live engine can drive it.
 *
 * The fingerspelling head reads one camera frame at a time and cannot wait for a sequence to
 * finish, so it holds the hidden state itself and steps. Sharing this function rather than
 * writing a second GRU beside it is the point: two copies of these six lines would agree on
 * the day they were written and quietly disagree afterwards.
 */
export function gruStep(
  input: Float32Array,
  state: Float32Array,
  direction: GruDirection,
  hidden: number,
): Float32Array<ArrayBuffer> {
  const gatesInput = affine(direction.weightIh, input, direction.biasIh);
  const gatesHidden = affine(direction.weightHh, state, direction.biasHh);

  const next = new Float32Array(hidden);
  for (let i = 0; i < hidden; i += 1) {
    const reset = sigmoid((gatesInput[i] ?? 0) + (gatesHidden[i] ?? 0));
    const update = sigmoid((gatesInput[hidden + i] ?? 0) + (gatesHidden[hidden + i] ?? 0));
    const candidate = Math.tanh(
      (gatesInput[2 * hidden + i] ?? 0) + reset * (gatesHidden[2 * hidden + i] ?? 0),
    );
    next[i] = (1 - update) * candidate + update * (state[i] ?? 0);
  }
  return next;
}

/** Concatenate the forward and reverse outputs per timestep, as PyTorch does. */
export function mergeDirections(
  forward: readonly Float32Array[],
  backward: readonly Float32Array[],
): Float32Array[] {
  return forward.map((step, i) => {
    const other = backward[i]!;
    const merged = new Float32Array(step.length + other.length);
    merged.set(step, 0);
    merged.set(other, step.length);
    return merged;
  });
}

export function meanOverTime(sequence: readonly Float32Array[]): Float32Array {
  const width = sequence[0]?.length ?? 0;
  const out = new Float32Array(width);
  for (const step of sequence) {
    for (let i = 0; i < width; i += 1) out[i] = (out[i] ?? 0) + (step[i] ?? 0);
  }
  for (let i = 0; i < width; i += 1) out[i] = (out[i] ?? 0) / sequence.length;
  return out;
}

/** Turn logits into probabilities, shifted by the max so `exp` cannot overflow. */
export function softmax(logits: Float32Array): Float32Array {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i += 1) max = Math.max(max, logits[i] ?? 0);

  const out = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp((logits[i] ?? 0) - max);
    out[i] = value;
    total += value;
  }
  for (let i = 0; i < out.length; i += 1) out[i] = (out[i] ?? 0) / total;
  return out;
}
