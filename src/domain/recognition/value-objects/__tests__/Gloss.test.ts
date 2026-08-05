import { describe, expect, it } from 'vitest';
import { conceptIdOf, createGloss } from '../Gloss';

describe('conceptIdOf', () => {
  it('collapses SWL-LSE variant markers onto one concept', () => {
    // The five sugar classes in the dataset must all mean sugar.
    const variants = ['AZUCAR', 'AZUCAR2', 'AZUCAR(2M)', 'AZUCAR(M-ES)', 'AZUCAR(M-ES)(2M)'];
    const concepts = new Set(variants.map(conceptIdOf));
    expect(concepts).toEqual(new Set(['AZUCAR']));
  });

  it('keeps genuinely different concepts apart', () => {
    expect(conceptIdOf('ESPALDA')).not.toBe(conceptIdOf('ESTOMAGO'));
  });

  it('does not strip a compound apart', () => {
    expect(conceptIdOf('CORAZON^INFARTO')).toBe('CORAZON^INFARTO');
  });
});

describe('createGloss', () => {
  it('renders a compound gloss as readable Spanish', () => {
    expect(createGloss('CORAZON^INFARTO').text).toBe('corazon infarto');
  });

  it('renders a hyphenated multiword gloss as separate words', () => {
    expect(createGloss('DOLOR-DE-CABEZA').text).toBe('dolor de cabeza');
  });

  it('leaves sentence casing to the transcript', () => {
    expect(createGloss('CABEZA').text).toBe('cabeza');
  });

  it('keeps the raw label as the trained-class id while sharing a concept', () => {
    const a = createGloss('ACCIDENTE');
    const b = createGloss('ACCIDENTE(A)');
    expect(a.id).not.toBe(b.id);
    expect(a.conceptId).toBe(b.conceptId);
  });
});
