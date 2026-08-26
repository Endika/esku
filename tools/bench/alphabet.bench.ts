import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createHandLandmarks, type Landmark } from '@domain/landmarks/value-objects/Landmark';
import type { LandmarkFrame } from '@domain/landmarks/value-objects/LandmarkFrame';
import { HandshapeAlphabetClassifier } from '@infrastructure/recognition/HandshapeAlphabetClassifier';
import { LSE_ALPHABET, UNSUPPORTED_LETTERS } from '@infrastructure/recognition/packs/lseAlphabet';
import { expect, it } from 'vitest';
import { editOps, type SpellStep, spell } from './spelling';

/**
 * What `HandshapeAlphabetClassifier` actually spells, on real continuous fingerspelling.
 *
 * It is the only engine the README advertises without a number. This runs the shipped
 * TypeScript — the table, the geometry and the stabiliser, unmodified — over LSE-FS-UVigo's
 * held-out split, because a Python reimplementation would measure the reimplementation, and
 * this repository has already been bitten twice by exactly that.
 *
 * Not part of `npm test`: it needs a corpus that is not in the repository and never will be.
 *
 *     tools/train/.venv/bin/python tools/train/lsefs_extract.py   # once
 *     npm run bench:alphabet
 */

const CACHE = path.resolve(
  import.meta.dirname,
  '..',
  'train',
  'data',
  'uvigo',
  'lsefs_test.ndjson',
);
const EXPECTED_SEQUENCES = 456;
/** Frame spacing is irrelevant here — the alphabet path reads shapes, never durations. */
const FRAME_MS = 40;

interface Sequence {
  readonly segment_id: string;
  readonly label: string;
  readonly signer: string | null;
  readonly frames: readonly { left: number[][] | null; right: number[][] | null }[];
}

function points(triples: number[][]): Landmark[] {
  return triples.map(([x, y, z]) => ({ x: x!, y: y!, z: z! }));
}

function frameOf(raw: Sequence['frames'][number], index: number): LandmarkFrame {
  const hands = [];
  // No handedness inversion: third-person footage, so MediaPipe's label is already anatomical.
  if (raw.left) hands.push(createHandLandmarks('left', points(raw.left)));
  if (raw.right) hands.push(createHandLandmarks('right', points(raw.right)));
  return { timestampMs: index * FRAME_MS, hands };
}

function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]!;
}

it('spells LSE-FS-UVigo with the shipped alphabet engine', async () => {
  const lines = readFileSync(CACHE, 'utf-8').split('\n').filter(Boolean);
  expect(lines).toHaveLength(EXPECTED_SEQUENCES);

  const supported = new Set(LSE_ALPHABET.map((template) => template.letter.toUpperCase()));
  const classifier = new HandshapeAlphabetClassifier();

  let characters = 0;
  let distance = 0;
  let substitutions = 0;
  let insertions = 0;
  let deletions = 0;
  let exact = 0;
  let spellableExact = 0;
  let spellable = 0;
  const confusions = new Map<string, number>();
  const missing = new Map<string, number>();

  let handFrames = 0;
  let starvedFrames = 0;
  const tops: number[] = [];

  for (const line of lines) {
    const sequence = JSON.parse(line) as Sequence;
    const label = sequence.label.toUpperCase();

    const steps: SpellStep[] = [];
    for (const [index, raw] of sequence.frames.entries()) {
      const frame = frameOf(raw, index);
      const candidates = await classifier.classify([frame]);
      if (frame.hands.length > 0) {
        handFrames += 1;
        const best = classifier.lastScores[0];
        if (best) tops.push(best.confidence);
        if (candidates.length === 0) starvedFrames += 1;
      }
      steps.push({ top: candidates[0] ?? null, empty: frame.hands.length === 0 });
    }

    const written = spell(steps).toUpperCase();
    const ops = editOps(label, written);

    characters += label.length;
    distance += ops.distance;
    substitutions += ops.substitutions;
    insertions += ops.insertions;
    deletions += ops.deletions;
    if (written === label) exact += 1;
    for (const [wanted, got] of ops.pairs) {
      const key = `${wanted} → ${got}`;
      confusions.set(key, (confusions.get(key) ?? 0) + 1);
    }

    const reachable = [...label].every((letter) => supported.has(letter));
    if (reachable) {
      spellable += 1;
      if (written === label) spellableExact += 1;
    }
    for (const letter of new Set(label)) {
      if (!supported.has(letter)) missing.set(letter, (missing.get(letter) ?? 0) + 1);
    }
  }

  tops.sort((a, b) => a - b);
  const pct = (value: number, total: number) => (total ? (value / total) * 100 : 0).toFixed(1);
  const say = (line: string) => process.stdout.write(`${line}\n`);

  say('');
  say('Esku deletreando LSE-FS-UVigo: lo que el motor de alfabeto escribe de verdad');
  say('');
  say(`   secuencias           : ${lines.length}`);
  say(`   caracteres anotados  : ${characters}`);
  say('');

  say('1. PALABRA');
  say(`   deletreo exacto              : ${exact}/${lines.length}  ${pct(exact, lines.length)}%`);
  say(`   distancia de edicion / letra : ${(distance / characters).toFixed(3)}`);
  say('');

  const matches = characters - substitutions - deletions;
  const written = matches + substitutions + insertions;
  say('2. ESCRIBIR Y ACERTAR  (son dos fallos distintos y el total los confunde)');
  say(`   letras escritas               : ${written}  de ${characters} anotadas`);
  say(`   de esas, correctas            : ${matches}  ${pct(matches, written)}% de lo escrito`);
  say(`   letras nunca escritas         : ${deletions}  ${pct(deletions, characters)}%`);
  say('');

  say('3. DONDE SE PIERDE  (cada uno pide un arreglo distinto)');
  say(
    `   sustituciones (la tabla)      : ${substitutions}  ` +
      `${pct(substitutions, characters)}% de las letras`,
  );
  say(`   inserciones (el estabilizador): ${insertions}  ${pct(insertions, characters)}%`);
  say(`   borrados (el umbral)          : ${deletions}  ${pct(deletions, characters)}%`);
  say('');

  say('4. CONFUSIONES  (esperada -> escrita, las 20 mas frecuentes)');
  const ranked = [...confusions.entries()].sort((a, b) => b[1] - a[1]).slice(0, 20);
  if (ranked.length === 0) say('   ninguna');
  for (const [pair, count] of ranked) say(`   ${pair.padEnd(10)} ${count}`);
  say('');

  say('5. SALUD DEL UMBRAL  (solo frames con mano visible)');
  say(`   frames con mano              : ${handFrames}`);
  say(`   sin ningun candidato         : ${starvedFrames}  ${pct(starvedFrames, handFrames)}%`);
  say(
    `   mejor puntuacion  p50 ${percentile(tops, 0.5).toFixed(3)}` +
      `  p75 ${percentile(tops, 0.75).toFixed(3)}` +
      `  p90 ${percentile(tops, 0.9).toFixed(3)}` +
      `  p99 ${percentile(tops, 0.99).toFixed(3)}`,
  );
  say('');

  say('6. COBERTURA  (lo que la tabla ni intenta)');
  say(`   letras en la tabla           : ${[...supported].sort().join(' ').toLowerCase()}`);
  say(`   declaradas no soportadas     : ${UNSUPPORTED_LETTERS.join(' ')}`);
  const ranking = [...missing.entries()].sort((a, b) => b[1] - a[1]);
  say(`   letras del test fuera de la tabla, por palabras afectadas:`);
  for (const [letter, count] of ranking) {
    say(`      ${letter}  ${count} palabras  ${pct(count, lines.length)}%`);
  }
  say('');
  say(
    `   palabras deletreables entera : ${spellable}/${lines.length}  ` +
      `${pct(spellable, lines.length)}%`,
  );
  say(
    `   exactas entre esas           : ${spellableExact}/${spellable}  ` +
      `${pct(spellableExact, spellable)}%`,
  );
  say('');
  say('El deletreo exacto del bloque 1 tiene como techo la ultima cifra del bloque 6: una');
  say('palabra con una letra que la tabla no contempla no puede salir bien nunca.');
});
