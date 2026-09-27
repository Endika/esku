/**
 * A model file that is missing, stale or cut short.
 *
 * None of those throw on their own: a 404 page parses as bytes, a truncated blob slices into
 * shorter tensors, and a tensor with no declared shape becomes a single float. Each one still
 * runs and returns confident answers, so this is raised instead.
 */
export class ModelFileError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelFileError';
  }
}

async function fetchOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new ModelFileError(`${url} answered HTTP ${response.status}`);
  return response;
}

export async function fetchModel<M>(
  manifestUrl: string,
  weightsUrl: string,
): Promise<[M, ArrayBuffer]> {
  return Promise.all([
    fetchOk(manifestUrl).then((r) => r.json() as Promise<M>),
    fetchOk(weightsUrl).then((r) => r.arrayBuffer()),
  ]);
}

/** Cuts the trainer's flat float32 blob into its named tensors, in the order it wrote them. */
export function sliceTensors(
  order: readonly string[],
  shapes: Readonly<Record<string, readonly number[]>>,
  blob: ArrayBuffer,
): Map<string, Float32Array> {
  const sizes = order.map((name) => {
    const shape = shapes[name];
    if (!shape) throw new ModelFileError(`The manifest lists ${name} but gives it no shape`);
    return shape.reduce((a, b) => a * b, 1);
  });
  const expectedBytes = sizes.reduce((a, b) => a + b, 0) * Float32Array.BYTES_PER_ELEMENT;
  if (blob.byteLength !== expectedBytes) {
    throw new ModelFileError(
      `The weights are ${blob.byteLength} bytes and the manifest describes ${expectedBytes}`,
    );
  }

  const floats = new Float32Array(blob);
  const tensors = new Map<string, Float32Array>();
  let offset = 0;
  order.forEach((name, i) => {
    const size = sizes[i]!;
    tensors.set(name, floats.subarray(offset, offset + size));
    offset += size;
  });
  return tensors;
}

/** Reads a tensor the network needs, refusing to run without it. */
export function requireTensor(
  tensors: ReadonlyMap<string, Float32Array>,
  name: string,
): Float32Array {
  const tensor = tensors.get(name);
  if (!tensor) throw new ModelFileError(`The weights have no ${name}`);
  return tensor;
}
