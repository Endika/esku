import type { RecognitionSource } from '@domain/recognition/value-objects/Gloss';
import { describe, expect, it } from 'vitest';
import { Transcript } from '../Transcript';

function entry(text: string, source: RecognitionSource, atMs = 0) {
  return { text, source, confidence: 0.9, atMs };
}

describe('Transcript', () => {
  it('joins consecutive fingerspelled letters into one word', () => {
    const transcript = ['c', 'a', 's', 'a'].reduce(
      (acc, letter, i) => acc.append(entry(letter, 'alphabet', i)),
      new Transcript(),
    );
    expect(transcript.toText()).toBe('Casa');
  });

  it('separates whole signs from each other', () => {
    const transcript = new Transcript()
      .append(entry('dolor', 'vocabulary'))
      .append(entry('cabeza', 'vocabulary'));
    expect(transcript.toText()).toBe('Dolor cabeza');
  });

  it('breaks a spelled word when a whole sign interrupts it', () => {
    const transcript = new Transcript()
      .append(entry('a', 'alphabet'))
      .append(entry('l', 'alphabet'))
      .append(entry('dolor', 'vocabulary'))
      .append(entry('s', 'alphabet'))
      .append(entry('i', 'alphabet'));
    expect(transcript.toText()).toBe('Al dolor si');
  });

  it('treats a taught sign as a whole word, not a letter', () => {
    const transcript = new Transcript()
      .append(entry('x', 'alphabet'))
      .append(entry('ibuprofeno', 'taught'));
    expect(transcript.toText()).toBe('X ibuprofeno');
  });

  it('does not mutate when appending', () => {
    const original = new Transcript().append(entry('a', 'alphabet'));
    original.append(entry('b', 'alphabet'));
    expect(original.entries).toHaveLength(1);
  });

  it('reports empty before anything is recognised', () => {
    expect(new Transcript().isEmpty).toBe(true);
    expect(new Transcript().toText()).toBe('');
  });

  it('drops only the last entry on removeLast', () => {
    const transcript = new Transcript()
      .append(entry('a', 'alphabet'))
      .append(entry('b', 'alphabet'))
      .removeLast();
    expect(transcript.toText()).toBe('A');
  });
});
