import { describe, it, expect } from 'vitest';
import { createEncoder, type Tokenizer } from './bpe';
import tokenizerData from '../../../tools/train-tiny-transformer/tokenizer.json';
import tokenizerRef from '../../../tools/train-tiny-transformer/tokenizer_ref.json';

const tokenizer = tokenizerData as unknown as Tokenizer;
const refs = tokenizerRef as unknown as { text: string; ids: number[] }[];

describe('createEncoder', () => {
  const encoder = createEncoder(tokenizer);

  it('matches the Python reference ids exactly for each reference text', () => {
    for (const { text, ids } of refs) {
      expect(encoder.encode(text)).toEqual(ids);
    }
  });

  it('round-trips decode(encode(text)) for the reference texts', () => {
    for (const { text } of refs) {
      const ids = encoder.encode(text);
      expect(encoder.decode(ids)).toBe(text);
    }
  });

  it('skips unknown characters without throwing', () => {
    expect(() => encoder.encode('héllo 中')).not.toThrow();
    const ids = encoder.encode('héllo 中');
    const vocabSize = tokenizer.vocab.length;
    for (const id of ids) {
      expect(id).toBeGreaterThanOrEqual(0);
      expect(id).toBeLessThan(vocabSize);
    }
  });

  it('encodes the empty string to an empty array', () => {
    expect(encoder.encode('')).toEqual([]);
  });
});
