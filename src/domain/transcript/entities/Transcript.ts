import type { RecognitionSource } from '@domain/recognition/value-objects/Gloss';

export interface TranscriptEntry {
  readonly text: string;
  readonly source: RecognitionSource;
  readonly confidence: number;
  readonly atMs: number;
}

/**
 * The running text built from recognised signs.
 *
 * Immutable: every append returns a new transcript, so the UI can diff and the history can
 * be undone without defensive copying.
 */
export class Transcript {
  constructor(readonly entries: readonly TranscriptEntry[] = []) {}

  append(entry: TranscriptEntry): Transcript {
    return new Transcript([...this.entries, entry]);
  }

  removeLast(): Transcript {
    return new Transcript(this.entries.slice(0, -1));
  }

  clear(): Transcript {
    return new Transcript();
  }

  get isEmpty(): boolean {
    return this.entries.length === 0;
  }

  /**
   * Renders to readable Spanish. Consecutive fingerspelled letters are joined into a word
   * — the whole point of spelling is that "c","a","s","a" is meant to read as "casa" — while
   * whole-sign entries stay separate words.
   */
  toText(): string {
    const parts: string[] = [];
    let spelling: string[] = [];

    const flush = () => {
      if (spelling.length > 0) {
        parts.push(spelling.join(''));
        spelling = [];
      }
    };

    for (const entry of this.entries) {
      if (entry.source === 'alphabet') {
        spelling.push(entry.text.toLowerCase());
      } else {
        flush();
        parts.push(entry.text);
      }
    }
    flush();

    const sentence = parts.join(' ').trim();
    return sentence.charAt(0).toUpperCase() + sentence.slice(1);
  }
}
