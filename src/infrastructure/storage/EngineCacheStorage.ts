/** Must match the `runtimeCaching` cacheName in `vite.config.ts`. */
const CACHE_NAME = 'esku-engine';

export interface EngineStorageReport {
  readonly cachedBytes: number;
  readonly entries: number;
  /**
   * Whether the WASM runtime made it into the cache, not just the model weights.
   *
   * Without it the app still works online — the browser refetches — but the offline promise
   * on the front page is false. Worth surfacing rather than leaving as a silent gap between
   * "downloaded" and "usable on a plane".
   */
  readonly hasRuntime: boolean;
}

/**
 * A minimal WebAssembly module that only validates where SIMD is supported.
 *
 * MediaPipe picks between its SIMD and no-SIMD builds with the same check. Warming the wrong
 * one would download 11 MB the browser never uses and still leave it offline-broken.
 *
 * These are the bytes from `wasm-feature-detect`, and they are worth copying exactly: a
 * hand-written version with a mis-declared body length validates as `false` everywhere,
 * including on runtimes that fully support SIMD. That failure is silent — it just caches the
 * wrong file.
 */
const SIMD_PROBE = Uint8Array.from([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15,
  253, 98, 11,
]);

export function supportsSimd(): boolean {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

/**
 * Downloads, inspects and frees the cached recognition engine.
 *
 * The engine is about 30 MB of WASM and model weights, cached so the app works offline
 * afterwards. That is a real amount of a phone's storage to take without asking, and someone
 * who only wanted to try fingerspelling once should be able to get it back.
 *
 * Deliberately does not touch taught signs: those are the user's own recordings, they live
 * in IndexedDB, and losing them to a "free up space" button would be a data-loss trap.
 */
export class EngineCacheStorage {
  constructor(private readonly base: string = '/') {}

  isSupported(): boolean {
    return typeof caches !== 'undefined';
  }

  /**
   * Every file this browser actually needs offline — the models, plus the one WASM build it
   * will choose. The other build is shipped but never fetched.
   */
  engineUrls(): string[] {
    const variant = supportsSimd() ? 'vision_wasm_internal' : 'vision_wasm_nosimd_internal';
    return [
      `${this.base}wasm/${variant}.js`,
      `${this.base}wasm/${variant}.wasm`,
      `${this.base}models/hand_landmarker.task`,
      `${this.base}models/pose_landmarker_lite.task`,
      `${this.base}models/face_landmarker.task`,
      `${this.base}models/lse-vocabulary.json`,
      `${this.base}models/lse-vocabulary.bin`,
    ];
  }

  /**
   * Fetch the engine and store it ourselves, rather than hoping to intercept MediaPipe.
   *
   * MediaPipe loads its WASM through its own machinery, and in practice the service worker
   * was not catching it: after a download the cache held the 19 MB of models and none of the
   * runtime. Online that is invisible; offline it is the difference between working and a
   * blank screen. Fetching the URLs directly puts them in the same cache the service worker
   * reads from, so MediaPipe's later request is served from it either way.
   */
  async warm(onProgress?: (done: number, total: number) => void): Promise<void> {
    if (!this.isSupported()) return;

    const cache = await caches.open(CACHE_NAME);
    const urls = this.engineUrls();
    const failed: string[] = [];
    let done = 0;

    for (const url of urls) {
      if (!(await cache.match(url))) {
        const response = await fetch(url).catch(() => null);
        if (response?.ok) {
          await cache.put(url, response.clone());
        } else {
          failed.push(url);
        }
      }
      // Counted only once the file is genuinely in the cache. Advancing on every attempt
      // would report "7 of 7" for a download where nothing was stored.
      if (!failed.includes(url)) {
        done += 1;
        onProgress?.(done, urls.length);
      }
    }

    if (failed.length > 0) {
      throw new Error(`Could not cache ${failed.length} of ${urls.length} engine files`);
    }
  }

  async report(): Promise<EngineStorageReport> {
    if (!this.isSupported()) return { cachedBytes: 0, entries: 0, hasRuntime: false };

    const cache = await caches.open(CACHE_NAME);
    const requests = await cache.keys();

    let cachedBytes = 0;
    let hasRuntime = false;
    for (const request of requests) {
      if (new URL(request.url).pathname.includes('/wasm/')) hasRuntime = true;
      const response = await cache.match(request);
      if (response) cachedBytes += await sizeOf(response);
    }

    return { cachedBytes, entries: requests.length, hasRuntime };
  }

  /** Returns false when there was nothing cached to begin with. */
  async clear(): Promise<boolean> {
    if (!this.isSupported()) return false;
    return caches.delete(CACHE_NAME);
  }
}

/**
 * The size on disk, measured from the body rather than read off a header.
 *
 * `Content-Length` looks like the cheap answer and is the wrong one: GitHub Pages gzips
 * everything, so the header carries the *transfer* size. Summing it reported 19 MB for an
 * engine that occupies 30 MB — and since the cache stores bodies decompressed, 19 MB was an
 * answer to a question this panel is not asking. It is titled "space on your device".
 *
 * Reading the blob costs a pass over ~30 MB, which is acceptable on a panel opened by hand.
 */
async function sizeOf(response: Response): Promise<number> {
  try {
    return (await response.clone().blob()).size;
  } catch {
    // Some browsers refuse to clone an opaque response; the header is better than nothing.
    const declared = Number(response.headers.get('content-length'));
    return Number.isFinite(declared) && declared > 0 ? declared : 0;
  }
}

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 MB';
  const megabytes = bytes / (1024 * 1024);
  return megabytes < 10 ? `${megabytes.toFixed(1)} MB` : `${Math.round(megabytes)} MB`;
}
