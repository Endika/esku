import { readFileSync } from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf-8')) as {
  version: string;
};

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/esku/' : '/',
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  resolve: {
    alias: {
      '@domain': path.resolve(import.meta.dirname, 'src/domain'),
      '@application': path.resolve(import.meta.dirname, 'src/application'),
      '@infrastructure': path.resolve(import.meta.dirname, 'src/infrastructure'),
      '@presentation': path.resolve(import.meta.dirname, 'src/presentation'),
      '@shared': path.resolve(import.meta.dirname, 'src/shared'),
      '@bootstrap': path.resolve(import.meta.dirname, 'src/bootstrap'),
      '@': path.resolve(import.meta.dirname, 'src'),
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      includeAssets: ['favicon.svg', 'favicon.ico', 'apple-touch-icon-180x180.png'],
      manifest: {
        name: 'Esku — Lengua de signos a texto',
        short_name: 'Esku',
        description:
          'Lee lengua de signos española con la cámara y la convierte en texto. Funciona sin conexión.',
        theme_color: '#7C3AED',
        background_color: '#0B0A12',
        display: 'standalone',
        orientation: 'portrait',
        lang: 'es',
        categories: ['accessibility', 'utilities', 'education'],
        icons: [
          { src: 'pwa-64x64.png', sizes: '64x64', type: 'image/png' },
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          {
            src: 'maskable-icon-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
        ],
      },
      workbox: {
        // Precache the shell only. The recognition engine is ~29 MB of WASM and weights;
        // precaching it would mean a 29 MB download before the first screen paints, and
        // most of it is the SIMD/no-SIMD pair of which any given browser uses exactly one.
        globPatterns: ['**/*.{js,css,html,svg,png,ico,woff2}'],
        globIgnores: ['wasm/**', 'models/**'],
        cleanupOutdatedCaches: true,
        runtimeCaching: [
          {
            // CacheFirst: these are content-addressed by release and never change in place,
            // so once fetched the app is fully offline without ever re-validating.
            urlPattern: ({ url }) => /\/(wasm|models)\//.test(url.pathname),
            handler: 'CacheFirst',
            options: {
              cacheName: 'esku-engine',
              expiration: { maxEntries: 12 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
}));
