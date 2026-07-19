import { gaussian, mulberry32 } from './rng';

export interface Point2 {
  x: number;
  y: number;
  label: 1 | -1;
}

const clamp = (v: number, lim = 1.1) => Math.max(-lim, Math.min(lim, v));

/** Two gaussian blobs on the diagonal — linearly separable. n points per class. */
export function twoClouds(n = 15, seed = 1): Point2[] {
  const rng = mulberry32(seed);
  const points: Point2[] = [];
  for (const label of [1, -1] as const) {
    const cx = 0.55 * label;
    const cy = 0.55 * label;
    for (let i = 0; i < n; i++) {
      points.push({
        x: clamp(cx + gaussian(rng) * 0.18),
        y: clamp(cy + gaussian(rng) * 0.18),
        label,
      });
    }
  }
  return points;
}

/** The XOR arrangement: label = sign(x*y). Uniform in the four quadrants,
 *  keeping a clear band around the axes. No single line separates it. */
export function xorQuadrants(n = 40, seed = 2): Point2[] {
  const rng = mulberry32(seed);
  const points: Point2[] = [];
  const sample = () => {
    // magnitude in [0.15, 1.05], sign uniform
    const mag = 0.15 + rng() * 0.9;
    return rng() < 0.5 ? -mag : mag;
  };
  for (let i = 0; i < n; i++) {
    const x = sample();
    const y = sample();
    points.push({ x, y, label: x * y > 0 ? 1 : -1 });
  }
  return points;
}
