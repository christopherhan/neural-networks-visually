import { describe, it, expect } from 'vitest';
import { linearScale } from './scale';

describe('linearScale', () => {
  it('maps domain endpoints to range endpoints', () => {
    const s = linearScale([-1, 1], [0, 100]);
    expect(s(-1)).toBe(0);
    expect(s(1)).toBe(100);
    expect(s(0)).toBe(50);
  });

  it('supports inverted ranges (SVG y-axis)', () => {
    const s = linearScale([-1, 1], [200, 0]);
    expect(s(-1)).toBe(200);
    expect(s(1)).toBe(0);
    expect(s(0)).toBe(100);
  });

  it('invert round-trips', () => {
    const s = linearScale([-1.2, 1.2], [10, 330]);
    for (const v of [-1.2, -0.3, 0, 0.77, 1.2]) {
      expect(s.invert(s(v))).toBeCloseTo(v);
    }
  });
});
