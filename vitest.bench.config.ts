import { mergeConfig } from 'vite';
import { defineConfig } from 'vitest/config';
import base from './vite.config.ts';

export default mergeConfig(
  base({ command: 'serve', mode: 'test' }),
  defineConfig({
    test: { include: ['tools/bench/**/*.bench.ts'], testTimeout: 600_000 },
  }),
);
