import { describe, expect, it } from 'vitest';
import { EngineCacheStorage, formatBytes } from '../EngineCacheStorage';

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
});
