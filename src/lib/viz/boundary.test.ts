import { describe, it, expect } from 'vitest';
import { boundaryGrid, BOUNDARY_DOMAIN } from './boundary';

describe('boundaryGrid', () => {
  it('returns n*n samples', () => {
    expect(boundaryGrid(() => 0, 8)).toHaveLength(64);
  });

  it('maps columns to x: left negative, right positive, antisymmetric', () => {
    const g = boundaryGrid((x) => x, 4);
    // row 0: cells at x = -0.9, -0.3, 0.3, 0.9 (for domain 1.2, n=4)
    expect(g[0]).toBeCloseTo(-0.9);
    expect(g[3]).toBeCloseTo(0.9);
    expect(g[1]).toBeCloseTo(-g[2]);
  });

  it('maps rows to y: top positive, bottom negative (screen orientation)', () => {
    const g = boundaryGrid((_x, y) => y, 4);
    expect(g[0]).toBeCloseTo(0.9);       // top row
    expect(g[12]).toBeCloseTo(-0.9);     // bottom row
  });

  it('spans the shared domain constant', () => {
    const g = boundaryGrid((x) => x, 2);
    expect(g[1]).toBeCloseTo(BOUNDARY_DOMAIN / 2);
  });
});
