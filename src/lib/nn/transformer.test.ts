import { describe, it, expect } from 'vitest';
import { loadModel, forward, softmaxTemp, sampleNext } from './transformer';
import weightsData from '../../../tools/train-tiny-transformer/weights.json';
import refData from '../../../tools/train-tiny-transformer/ref_logits.json';

const ref = refData as unknown as { ids: number[]; last_logits_first32: number[] };

describe('loadModel + forward — Python parity', () => {
  const model = loadModel(weightsData);

  it('matches the Python reference last-position logits within 2e-3 abs (weights rounded to 5dp)', () => {
    const { logits } = forward(model, ref.ids);
    const last = logits[logits.length - 1];
    for (let i = 0; i < ref.last_logits_first32.length; i++) {
      expect(Math.abs(last[i] - ref.last_logits_first32[i])).toBeLessThan(2e-3);
    }
  });

  it('produces causal attention: zero weight beyond the query position, rows sum to ~1', () => {
    const { attention } = forward(model, ref.ids);
    const T = ref.ids.length;
    for (let l = 0; l < attention.length; l++) {
      for (let h = 0; h < attention[l].length; h++) {
        for (let t = 0; t < T; t++) {
          let sum = 0;
          for (let k = 0; k < T; k++) {
            const w = attention[l][h][t][k];
            if (k > t) {
              expect(w).toBe(0);
            }
            sum += w;
          }
          expect(sum).toBeCloseTo(1, 6);
        }
      }
    }
  });

  it('shapes: attention is [n_layer][n_head][T][T], logits is [T][vocab_size]', () => {
    const { logits, attention } = forward(model, ref.ids);
    const T = ref.ids.length;
    expect(attention.length).toBe(2);
    for (const layer of attention) {
      expect(layer.length).toBe(4);
      for (const head of layer) {
        expect(head.length).toBe(T);
        for (const row of head) {
          expect(row.length).toBe(T);
        }
      }
    }
    expect(logits.length).toBe(T);
    for (const row of logits) {
      expect(row.length).toBe(383);
    }
  });
});

describe('softmaxTemp', () => {
  it('sums to 1', () => {
    const probs = softmaxTemp([1, 2, 3, 0.5], 1.0);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it('lower temperature sharpens the distribution', () => {
    const logits = [1, 2, 3, 0.5];
    const sharp = softmaxTemp(logits, 0.5);
    const soft = softmaxTemp(logits, 1.0);
    const maxSharp = Math.max(...sharp);
    const maxSoft = Math.max(...soft);
    expect(maxSharp).toBeGreaterThan(maxSoft);
  });
});

describe('sampleNext', () => {
  const probs = [0.2, 0.5, 0.3];

  it('picks index 0 when the random draw lands in the first bucket', () => {
    expect(sampleNext(probs, () => 0.15)).toBe(0);
  });

  it('picks index 1 when the random draw lands in the second bucket', () => {
    expect(sampleNext(probs, () => 0.3)).toBe(1);
  });

  it('picks index 2 when the random draw lands in the third bucket', () => {
    expect(sampleNext(probs, () => 0.95)).toBe(2);
  });
});
