/**
 * A recognisable unit of a sign language, in its written-gloss form.
 *
 * SWL-LSE ships 300 labels but many are variants of one concept (`ACCIDENTE`,
 * `ACCIDENTE(A)`, `ACCIDENTE2`; `AZUCAR` appears five times). `conceptId` is what the user
 * reads; `id` stays unique per trained class. Collapsing variants onto one concept both
 * reads better and trains better — fewer classes, more samples each.
 */
export interface Gloss {
  readonly id: string;
  readonly conceptId: string;
  readonly text: string;
}

export type RecognitionSource = 'vocabulary' | 'alphabet' | 'taught';

export interface SignCandidate {
  readonly gloss: Gloss;
  /** 0..1. Comparable only within a single classifier, hence `source`. */
  readonly confidence: number;
  readonly source: RecognitionSource;
}

/** Strip SWL-LSE's variant markers: `AZUCAR(M-ES)(2M)` and `AZUCAR2` both mean sugar. */
export function conceptIdOf(label: string): string {
  return label
    .replace(/\([^)]*\)/g, '')
    .replace(/\d+$/, '')
    .trim();
}

export function createGloss(label: string): Gloss {
  const conceptId = conceptIdOf(label);
  return { id: label, conceptId, text: humanize(conceptId) };
}

/**
 * Glosses are written in linguistic notation — upper case, `^` joining compounds, hyphens
 * inside multiword concepts. Turn that into something readable in a transcript.
 *
 * Left lower case on purpose: only `Transcript` knows where a sentence starts, and
 * capitalising here would spell "Dolor Cabeza" mid-sentence.
 */
function humanize(conceptId: string): string {
  return conceptId
    .split('^')
    .map((part) => part.split(/[-.]/).join(' ').toLowerCase())
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function byConfidenceDescending(a: SignCandidate, b: SignCandidate): number {
  return b.confidence - a.confidence;
}
