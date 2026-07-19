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
export function xorQuadrants(n = 40, seed = 3): Point2[] {
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

/** Concentric classes: inner disk (label 1) vs surrounding ring (label −1).
 *  Not linearly separable — needs a curved boundary. */
export function circles(n = 60, seed = 5): Point2[] {
  const rng = mulberry32(seed);
  const points: Point2[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const r = 0.35 * Math.sqrt(rng());
    const a = rng() * 2 * Math.PI;
    points.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: 1 });
  }
  for (let i = 0; i < n - half; i++) {
    const r = 0.65 + rng() * 0.3;
    const a = rng() * 2 * Math.PI;
    points.push({ x: clamp(r * Math.cos(a)), y: clamp(r * Math.sin(a)), label: -1 });
  }
  return points;
}

/** Two interleaved half-moons — the classic "needs a wiggly boundary" set. */
export function moons(n = 60, seed = 6): Point2[] {
  const rng = mulberry32(seed);
  const points: Point2[] = [];
  const half = Math.floor(n / 2);
  const noise = 0.06;
  for (let i = 0; i < half; i++) {
    const t = rng() * Math.PI;
    points.push({
      x: clamp((Math.cos(t) - 0.5) * 0.75 + gaussian(rng) * noise),
      y: clamp((Math.sin(t) - 0.25) * 0.75 + gaussian(rng) * noise),
      label: 1,
    });
  }
  for (let i = 0; i < n - half; i++) {
    const t = rng() * Math.PI;
    points.push({
      x: clamp((1 - Math.cos(t) - 0.5) * 0.75 + gaussian(rng) * noise),
      y: clamp((0.5 - Math.sin(t) - 0.25) * 0.75 + gaussian(rng) * noise),
      label: -1,
    });
  }
  return points;
}

/** Two interlocking spiral arms — the hardest of the toy sets. */
export function spiral(n = 80, seed = 7): Point2[] {
  const rng = mulberry32(seed);
  const points: Point2[] = [];
  const half = Math.floor(n / 2);
  for (const label of [1, -1] as const) {
    const offset = label === 1 ? 0 : Math.PI;
    const count = label === 1 ? half : n - half;
    for (let i = 0; i < count; i++) {
      const t = (i + 0.5) / count;
      const r = 0.12 + 0.88 * t;
      const a = 3 * Math.PI * t + offset;
      points.push({
        x: clamp(r * Math.cos(a) + gaussian(rng) * 0.025),
        y: clamp(r * Math.sin(a) + gaussian(rng) * 0.025),
        label,
      });
    }
  }
  return points;
}

export interface PlaygroundDataset {
  label: string;
  generate: () => Point2[];
}

/** The Training Playground's menu. Thunks bake in counts and seeds so every
 *  entry is deterministic and the per-dataset n-semantics stay internal. */
export const PLAYGROUND_DATASETS = {
  clouds: { label: 'Two clouds', generate: () => twoClouds(20, 1) },
  xor: { label: 'XOR', generate: () => xorQuadrants(60) },
  circles: { label: 'Circles', generate: () => circles(60) },
  moons: { label: 'Moons', generate: () => moons(60) },
  spiral: { label: 'Spiral', generate: () => spiral(80) },
} satisfies Record<string, PlaygroundDataset>;

export type PlaygroundDatasetKey = keyof typeof PLAYGROUND_DATASETS;
