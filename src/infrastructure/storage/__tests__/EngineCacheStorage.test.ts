import { describe, expect, it } from 'vitest';
import { EngineCacheStorage, formatBytes, supportsSimd } from '../EngineCacheStorage';

describe('formatBytes', () => {
  it('reads a large engine in whole megabytes', () => {
    expect(formatBytes(29 * 1024 * 1024)).toBe('29 MB');
  });

  it('keeps a decimal below ten megabytes, where rounding would hide the difference', () => {
    expect(formatBytes(7.5 * 1024 * 1024)).toBe('7.5 MB');
  });

  it('shows nothing cached as zero rather than a negative or NaN', () => {
    expect(formatBytes(0)).toBe('0 MB');
    expect(formatBytes(-1)).toBe('0 MB');
  });
});

describe('supportsSimd', () => {
  it('detects SIMD where the runtime has it', () => {
    // Node and every browser this app targets support WASM SIMD. A hand-written probe with a
    // mis-declared body length validates as false *everywhere*, which silently caches the
    // wrong 11 MB build and leaves the app broken offline. This test is that guard.
    expect(supportsSimd()).toBe(true);
  });
});

describe('engineUrls', () => {
  it('lists one WASM build, not both', () => {
    // Both are shipped; a browser downloads exactly one. Warming both would waste 11 MB.
    const urls = new EngineCacheStorage('/esku/').engineUrls();
    expect(urls.filter((url) => url.includes('/wasm/'))).toHaveLength(2); // .js and .wasm
    expect(urls.filter((url) => url.includes('nosimd')).length).toBe(supportsSimd() ? 0 : 2);
  });

  it('includes every model the app needs offline', () => {
    const urls = new EngineCacheStorage('/esku/').engineUrls();
    for (const model of [
      'hand_landmarker',
      'pose_landmarker',
      'face_landmarker',
      'lse-vocabulary.bin',
    ]) {
      expect(urls.some((url) => url.includes(model))).toBe(true);
    }
  });

  it('respects the deployed base path', () => {
    // Served from /esku/ on Pages and / in dev; a leading slash would 404 on Pages.
    expect(new EngineCacheStorage('/esku/').engineUrls().every((u) => u.startsWith('/esku/'))).toBe(
      true,
    );
  });
});

describe('EngineCacheStorage', () => {
  it('reports nothing cached where the Cache API is unavailable', async () => {
    // jsdom has no CacheStorage, which is also the real situation in a non-secure context.
    const storage = new EngineCacheStorage();
    expect(storage.isSupported()).toBe(false);
    expect(await storage.report()).toEqual({ cachedBytes: 0, entries: 0, hasRuntime: false });
  });

  it('reports nothing freed rather than throwing when there is no cache to clear', async () => {
    expect(await new EngineCacheStorage().clear()).toBe(false);
  });

  it('does not claim the runtime is cached when nothing is', async () => {
    // Weights without the WASM that runs them look "downloaded" and fail offline.
    expect((await new EngineCacheStorage().report()).hasRuntime).toBe(false);
  });

  it('warming is a no-op rather than a crash where the Cache API is absent', async () => {
    await expect(new EngineCacheStorage().warm()).resolves.toBeUndefined();
  });
});
