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
 * Inspects and frees the cached recognition engine.
 *
 * The engine is roughly 29 MB of WASM and model weights, cached on first use so the app
 * works offline afterwards. That is a real amount of a phone's storage to take without
 * asking, and someone who only wanted to try fingerspelling once should be able to get it
 * back.
 *
 * Deliberately does not touch taught signs: those are the user's own recordings, they live
 * in IndexedDB, and losing them to a "free up space" button would be a data-loss trap.
 */
export class EngineCacheStorage {
  isSupported(): boolean {
    return typeof caches !== 'undefined';
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
 * Content-Length first, body second.
 *
 * Reading the blob is exact but pulls the whole 11 MB WASM into memory just to measure it;
 * the header avoids that whenever the server sent one, which GitHub Pages does.
 */
async function sizeOf(response: Response): Promise<number> {
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > 0) return declared;

  try {
    return (await response.clone().blob()).size;
  } catch {
    return 0;
  }
}

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 MB';
  const megabytes = bytes / (1024 * 1024);
  return megabytes < 10 ? `${megabytes.toFixed(1)} MB` : `${Math.round(megabytes)} MB`;
}
