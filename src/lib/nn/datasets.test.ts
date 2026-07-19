import { describe, it, expect } from 'vitest';
import { twoClouds, xorQuadrants } from './datasets';

describe('twoClouds', () => {
  it('is deterministic and returns 2n points, n per class', () => {
    const a = twoClouds(15, 1);
    const b = twoClouds(15, 1);
    expect(a).toEqual(b);
    expect(a).toHaveLength(30);
    expect(a.filter((p) => p.label === 1)).toHaveLength(15);
    expect(a.filter((p) => p.label === -1)).toHaveLength(15);
  });

  it('keeps points inside the plot domain', () => {
    for (const p of twoClouds(50, 2)) {
      expect(Math.abs(p.x)).toBeLessThanOrEqual(1.1);
      expect(Math.abs(p.y)).toBeLessThanOrEqual(1.1);
    }
  });

  it('is linearly separable by the diagonal for default params', () => {
    // Clouds centered (±0.55, ±0.55) with tight spread: x + y sign splits them.
    const pts = twoClouds(30, 1);
    const correct = pts.filter((p) => Math.sign(p.x + p.y) === p.label).length;
    expect(correct / pts.length).toBeGreaterThan(0.95);
  });
});

describe('xorQuadrants', () => {
  it('is deterministic with labels = sign(x * y)', () => {
    const a = xorQuadrants(40, 2);
    expect(a).toEqual(xorQuadrants(40, 2));
    for (const p of a) {
      expect(p.label).toBe(p.x * p.y > 0 ? 1 : -1);
    }
  });

  it('avoids the axis band so the pattern reads clearly', () => {
    for (const p of xorQuadrants(60, 3)) {
      expect(Math.abs(p.x)).toBeGreaterThanOrEqual(0.15);
      expect(Math.abs(p.y)).toBeGreaterThanOrEqual(0.15);
      expect(Math.abs(p.x)).toBeLessThanOrEqual(1.1);
      expect(Math.abs(p.y)).toBeLessThanOrEqual(1.1);
    }
  });

  it('touches all four quadrants', () => {
    const pts = xorQuadrants(60, 4);
    const quadrant = (p: { x: number; y: number }) =>
      `${Math.sign(p.x)}${Math.sign(p.y)}`;
    expect(new Set(pts.map(quadrant)).size).toBe(4);
  });

  it('default seed gives a visually balanced spread (every quadrant ≥ n/8)', () => {
    const pts = xorQuadrants(40);
    const counts = new Map<string, number>();
    for (const p of pts) {
      const q = `${Math.sign(p.x)},${Math.sign(p.y)}`;
      counts.set(q, (counts.get(q) ?? 0) + 1);
    }
    expect(counts.size).toBe(4);
    for (const c of counts.values()) {
      expect(c).toBeGreaterThanOrEqual(5);
    }
  });
});
