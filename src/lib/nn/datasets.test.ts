import { describe, it, expect } from 'vitest';
import { twoClouds, xorQuadrants, circles, moons, spiral, PLAYGROUND_DATASETS } from './datasets';

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

describe('circles', () => {
  it('is deterministic with half the points per class', () => {
    const a = circles(60, 5);
    expect(a).toEqual(circles(60, 5));
    expect(a).toHaveLength(60);
    expect(a.filter((p) => p.label === 1)).toHaveLength(30);
  });

  it('separates inner disk from outer ring by radius', () => {
    for (const p of circles(60, 5)) {
      const r = Math.hypot(p.x, p.y);
      if (p.label === 1) expect(r).toBeLessThanOrEqual(0.36);
      else expect(r).toBeGreaterThanOrEqual(0.6);
    }
  });
});

describe('moons', () => {
  it('is deterministic, in-domain, with half the points per class', () => {
    const a = moons(60, 6);
    expect(a).toEqual(moons(60, 6));
    expect(a.filter((p) => p.label === 1)).toHaveLength(30);
    for (const p of a) {
      expect(Math.abs(p.x)).toBeLessThanOrEqual(1.1);
      expect(Math.abs(p.y)).toBeLessThanOrEqual(1.1);
    }
  });

  it('is NOT linearly separable by any axis-aligned split (interleaved)', () => {
    // The moons interleave: each class spans both sides of x=0.
    const pts = moons(60, 6);
    for (const label of [1, -1] as const) {
      const xs = pts.filter((p) => p.label === label).map((p) => p.x);
      expect(Math.min(...xs)).toBeLessThan(0);
      expect(Math.max(...xs)).toBeGreaterThan(0);
    }
  });
});

describe('spiral', () => {
  it('is deterministic, in-domain, with half the points per class', () => {
    const a = spiral(80, 7);
    expect(a).toEqual(spiral(80, 7));
    expect(a.filter((p) => p.label === 1)).toHaveLength(40);
    for (const p of a) {
      expect(Math.abs(p.x)).toBeLessThanOrEqual(1.1);
      expect(Math.abs(p.y)).toBeLessThanOrEqual(1.1);
    }
  });

  it('arms grow outward — radius increases along each arm', () => {
    const arm = spiral(80, 7).filter((p) => p.label === 1);
    const rFirst = Math.hypot(arm[0].x, arm[0].y);
    const rLast = Math.hypot(arm[arm.length - 1].x, arm[arm.length - 1].y);
    expect(rLast).toBeGreaterThan(rFirst + 0.4);
  });
});

describe('PLAYGROUND_DATASETS', () => {
  it('exposes the five datasets, each deterministic and labeled ±1', () => {
    const keys = Object.keys(PLAYGROUND_DATASETS);
    expect(keys).toEqual(['clouds', 'xor', 'circles', 'moons', 'spiral']);
    for (const key of keys as Array<keyof typeof PLAYGROUND_DATASETS>) {
      const entry = PLAYGROUND_DATASETS[key];
      expect(entry.label.length).toBeGreaterThan(0);
      const a = entry.generate();
      expect(a).toEqual(entry.generate());
      expect(a.length).toBeGreaterThanOrEqual(30);
      for (const p of a) expect([1, -1]).toContain(p.label);
    }
  });
});
