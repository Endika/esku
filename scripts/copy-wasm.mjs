import { copyFileSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Stages MediaPipe's WASM runtime into `public/wasm/` so it is served same-origin.
 *
 * Copied from `node_modules` at build time rather than committed, for two reasons: 22 MB of
 * binaries would sit in every clone forever, and a committed copy can silently drift from
 * the `@mediapipe/tasks-vision` version that loads it — a mismatch that fails at runtime,
 * inside WASM, with nothing useful in the stack trace. package-lock pins both together.
 *
 * The `_module_` variants are skipped: MediaPipe picks the SIMD build, or the no-SIMD one as
 * a fallback, and never those.
 */
const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, '..', 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const target = join(here, '..', 'public', 'wasm');

const wanted = (name) => !name.includes('_module_');

mkdirSync(target, { recursive: true });

const copied = readdirSync(source)
  .filter(wanted)
  .map((name) => {
    copyFileSync(join(source, name), join(target, name));
    return name;
  });

if (copied.length === 0) {
  throw new Error(`No MediaPipe WASM found in ${source} — is @mediapipe/tasks-vision installed?`);
}

console.log(`Staged ${copied.length} MediaPipe WASM files into public/wasm/`);
