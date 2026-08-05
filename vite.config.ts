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
      '@domain': path.resolve(__dirname, 'src/domain'),
      '@application': path.resolve(__dirname, 'src/application'),
      '@infrastructure': path.resolve(__dirname, 'src/infrastructure'),
      '@presentation': path.resolve(__dirname, 'src/presentation'),
      '@shared': path.resolve(__dirname, 'src/shared'),
      '@bootstrap': path.resolve(__dirname, 'src/bootstrap'),
      '@': path.resolve(__dirname, 'src'),
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
        // The MediaPipe .task bundles and the ONNX heads are several MB each; they are
        // precached deliberately so the very first offline launch can already recognise.
        globPatterns: ['**/*.{js,css,html,svg,png,ico,woff2,wasm,task,onnx,json}'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        cleanupOutdatedCaches: true,
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
