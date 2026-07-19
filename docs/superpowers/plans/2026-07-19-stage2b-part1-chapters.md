# Stage 2b: Part I Chapters 2–5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chapters 2–5 of Part I, each with its interactive figure — Activation Explorer, Gradient Descent (roll the ball, scrub the learning rate), Backprop Stepper (step the chain rule through a real computation graph), and the Training Playground flagship (train a live MLP, watch the decision boundary form) — plus the datasets and viz helpers they need.

**Architecture:** Figures follow the Stage 2a pattern (Svelte 5 islands, `client:visible`, `FigureShell`, deterministic seeds, SSR fallback). New infra: three more datasets + a playground registry, a rAF `createTicker` with injectable timing for tests, and a pure `boundaryGrid` sampler for canvas heatmaps. The playground uses the real `MLP`/`trainStep` engine at interactive rates (~1.3 ms/step measured in 2a).

**Tech Stack:** Existing — TypeScript, Svelte 5 runes, Vitest, Astro/MDX. No new dependencies.

**Governing docs:** spec `docs/superpowers/specs/2026-07-19-neural-networks-site-design.md`; Stage 2a addendum house rules in `docs/superpowers/plans/2026-07-19-stage2a-engine-and-first-figure.md` (SSR fallback, deterministic seeds, near-miss defaults, tooltip/a11y rules, RNG-sequence contract).

**File map:**

```
src/lib/nn/datasets.ts                 +circles, +moons, +spiral, +PLAYGROUND_DATASETS   (Task 1)
src/lib/viz/ticker.ts                  createTicker (rAF, injectable)                    (Task 2)
src/lib/viz/boundary.ts                boundaryGrid + BOUNDARY_DOMAIN                    (Task 2)
src/components/figures/activation/ActivationExplorer.svelte                              (Task 3)
src/content/chapters/activation-functions.mdx                                            (Task 3)
src/components/figures/gradient/GradientDescentFigure.svelte                             (Task 4)
src/content/chapters/loss-and-gradient-descent.mdx                                       (Task 4)
src/components/figures/backprop/BackpropStepper.svelte                                   (Task 5)
src/content/chapters/backpropagation.mdx                                                 (Task 5)
src/components/figures/playground/TrainingPlayground.svelte                              (Task 6)
src/content/chapters/training-playground.mdx                                             (Task 6)
```

**Frontmatter contract (build fails on mismatch — curriculum.ts is the source of truth):**

| file | number | title |
|---|---|---|
| activation-functions.mdx | 2 | Activation Functions |
| loss-and-gradient-descent.mdx | 3 | Loss & Gradient Descent |
| backpropagation.mdx | 4 | Backpropagation |
| training-playground.mdx | 5 | The Training Playground |

---

### Task 1: Datasets — circles, moons, spiral + playground registry (TDD)

**Files:**
- Modify: `src/lib/nn/datasets.ts` (append after `xorQuadrants`)
- Modify: `src/lib/nn/datasets.test.ts` (append new describes)

- [ ] **Step 1: Append these tests to `src/lib/nn/datasets.test.ts`** (add `circles, moons, spiral, PLAYGROUND_DATASETS` to the existing import from `./datasets`):

```ts
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
```

- [ ] **Step 2: Run `npm test`** — expected FAIL: `circles` etc. not exported.

- [ ] **Step 3: Append to `src/lib/nn/datasets.ts`:**

```ts
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
```

- [ ] **Step 4: Run `npm test`** — expected PASS (59 + 8 new = 67). If a radius/interleave assertion fails for a pinned seed, adjust only the seed constant (in BOTH the function default and the test if the test uses the default) to one that satisfies the property, and note it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/lib/nn/datasets.ts src/lib/nn/datasets.test.ts
git commit -m "feat: circles/moons/spiral datasets and playground registry"
```

---

### Task 2: Viz — createTicker + boundaryGrid (TDD)

**Files:**
- Create: `src/lib/viz/ticker.ts`, `src/lib/viz/boundary.ts`
- Test: `src/lib/viz/ticker.test.ts`, `src/lib/viz/boundary.test.ts`

- [ ] **Step 1: Write `src/lib/viz/ticker.test.ts`:**

```ts
import { describe, it, expect } from 'vitest';
import { createTicker } from './ticker';

/** Fake rAF: caller pumps frames with explicit timestamps. */
function fakeRaf() {
  let nextId = 1;
  const pending = new Map<number, (t: number) => void>();
  return {
    raf: (cb: (t: number) => void) => {
      const id = nextId++;
      pending.set(id, cb);
      return id;
    },
    caf: (id: number) => {
      pending.delete(id);
    },
    pump(t: number) {
      const cbs = [...pending.values()];
      pending.clear();
      for (const cb of cbs) cb(t);
    },
    get scheduled() {
      return pending.size;
    },
  };
}

describe('createTicker', () => {
  it('fires at most once per interval', () => {
    const frames = fakeRaf();
    let ticks = 0;
    const t = createTicker(() => ticks++, { fps: 10, raf: frames.raf, caf: frames.caf });
    t.start();
    frames.pump(0);      // first frame always ticks
    frames.pump(50);     // 50ms < 100ms interval — no tick
    frames.pump(100);    // 100ms since last tick — ticks
    frames.pump(140);    // no
    frames.pump(210);    // yes
    expect(ticks).toBe(3);
  });

  it('stop cancels the pending frame; start/stop are idempotent', () => {
    const frames = fakeRaf();
    let ticks = 0;
    const t = createTicker(() => ticks++, { fps: 60, raf: frames.raf, caf: frames.caf });
    t.start();
    t.start();
    expect(frames.scheduled).toBe(1);
    t.stop();
    t.stop();
    expect(frames.scheduled).toBe(0);
    frames.pump(1000);
    expect(ticks).toBe(0);
    expect(t.running).toBe(false);
  });

  it('reports running state', () => {
    const frames = fakeRaf();
    const t = createTicker(() => {}, { raf: frames.raf, caf: frames.caf });
    expect(t.running).toBe(false);
    t.start();
    expect(t.running).toBe(true);
    t.stop();
    expect(t.running).toBe(false);
  });
});
```

- [ ] **Step 2: Write `src/lib/viz/boundary.test.ts`:**

```ts
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
```

- [ ] **Step 3: Run `npm test`** — expected FAIL: modules not found.

- [ ] **Step 4: Write `src/lib/viz/ticker.ts`:**

```ts
/** requestAnimationFrame loop throttled to a target fps.
 *  raf/caf are injectable so tests can drive time explicitly. */
export interface Ticker {
  start(): void;
  stop(): void;
  readonly running: boolean;
}

interface TickerOptions {
  fps?: number;
  raf?: (cb: (t: number) => void) => number;
  caf?: (id: number) => void;
}

export function createTicker(tick: () => void, options: TickerOptions = {}): Ticker {
  const fps = options.fps ?? 30;
  const raf = options.raf ?? ((cb: (t: number) => void) => globalThis.requestAnimationFrame(cb));
  const caf = options.caf ?? ((id: number) => globalThis.cancelAnimationFrame(id));
  const interval = 1000 / fps;
  let id: number | null = null;
  let last = -Infinity;

  const loop = (t: number) => {
    id = raf(loop);
    if (t - last >= interval) {
      last = t;
      tick();
    }
  };

  return {
    start() {
      if (id === null) {
        last = -Infinity;
        id = raf(loop);
      }
    },
    stop() {
      if (id !== null) {
        caf(id);
        id = null;
      }
    },
    get running() {
      return id !== null;
    },
  };
}
```

- [ ] **Step 5: Write `src/lib/viz/boundary.ts`:**

```ts
/** The figures' shared plot domain half-width: plots span [-1.2, 1.2]². */
export const BOUNDARY_DOMAIN = 1.2;

/** Sample `predict` over an n×n grid of cell centers covering the domain.
 *  Row-major from the top-left: y DECREASES with row index, matching screen
 *  coordinates, so the result can be written straight into ImageData. */
export function boundaryGrid(
  predict: (x: number, y: number) => number,
  n: number
): Float32Array {
  const D = BOUNDARY_DOMAIN;
  const out = new Float32Array(n * n);
  const cell = (2 * D) / n;
  for (let row = 0; row < n; row++) {
    const y = D - (row + 0.5) * cell;
    for (let col = 0; col < n; col++) {
      const x = -D + (col + 0.5) * cell;
      out[row * n + col] = predict(x, y);
    }
  }
  return out;
}
```

- [ ] **Step 6: Run `npm test`** — expected PASS (67 + 7 = 74).

- [ ] **Step 7: Commit**

```bash
git add src/lib/viz/ticker.ts src/lib/viz/ticker.test.ts src/lib/viz/boundary.ts src/lib/viz/boundary.test.ts
git commit -m "feat: rAF ticker with injectable timing; boundary grid sampler"
```

---

### Task 3: Chapter 2 — Activation Explorer + MDX

**Files:**
- Create: `src/components/figures/activation/ActivationExplorer.svelte`
- Create: `src/content/chapters/activation-functions.mdx`

- [ ] **Step 1: Write `src/components/figures/activation/ActivationExplorer.svelte`:**

```svelte
<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { linearScale } from '../../../lib/viz/scale';

  type FnKey = 'step' | 'sigmoid' | 'tanh' | 'relu';

  const FNS: Record<FnKey, { label: string; f: (x: number) => number; df: (x: number) => number }> = {
    step: { label: 'Step', f: (x) => (x > 0 ? 1 : 0), df: () => 0 },
    sigmoid: {
      label: 'Sigmoid',
      f: (x) => 1 / (1 + Math.exp(-x)),
      df: (x) => {
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
      },
    },
    tanh: { label: 'Tanh', f: Math.tanh, df: (x) => 1 - Math.tanh(x) ** 2 },
    relu: { label: 'ReLU', f: (x) => Math.max(0, x), df: (x) => (x > 0 ? 1 : 0) },
  };

  let fnKey: FnKey = $state('tanh');
  let x = $state(0.8);
  let showDerivative = $state(false);

  const sx = linearScale([-3, 3], [12, 328]);
  const sy = linearScale([-1.5, 1.5], [208, 12]);
  const clampY = (v: number) => Math.max(-1.5, Math.min(1.5, v));
  const fmt = (v: number) => v.toFixed(2);

  const fn = $derived(FNS[fnKey]);

  function sample(g: (x: number) => number): string {
    const pts: string[] = [];
    for (let i = 0; i <= 120; i++) {
      const xi = -3 + (6 * i) / 120;
      pts.push(`${sx(xi)},${sy(clampY(g(xi)))}`);
    }
    return pts.join(' ');
  }

  const curve = $derived(sample(fn.f));
  const dCurve = $derived(sample(fn.df));
  const deadStep = $derived(fnKey === 'step' && showDerivative);
</script>

<FigureShell
  title="The activation zoo"
  caption="Pick a function, slide the input. The derivative (toggle) is what chapter 3 will climb along."
>
  {#snippet children()}
    <svg viewBox="0 0 340 220" aria-label="Graph of the selected activation function with a movable input marker">
      <line x1={sx(-3)} y1={sy(0)} x2={sx(3)} y2={sy(0)} stroke="var(--line-soft)" />
      <line x1={sx(0)} y1={sy(-1.5)} x2={sx(0)} y2={sy(1.5)} stroke="var(--line-soft)" />
      <polyline points={curve} fill="none" stroke="var(--accent-blue)" stroke-width="2.5" />
      {#if showDerivative}
        <polyline points={dCurve} fill="none" stroke="var(--accent-gold)" stroke-width="2" stroke-dasharray="5 4" />
      {/if}
      <line x1={sx(x)} y1={sy(-1.5)} x2={sx(x)} y2={sy(clampY(fn.f(x)))} stroke="var(--faint)" stroke-dasharray="3 3" />
      <circle cx={sx(x)} cy={sy(clampY(fn.f(x)))} r="6" fill="var(--accent-red)" />
    </svg>

    <p class="readout">
      f({fmt(x)}) = <strong>{fmt(fn.f(x))}</strong>
      {#if showDerivative}
        &nbsp;·&nbsp; slope f′({fmt(x)}) = <strong class="gold">{fmt(fn.df(x))}</strong>
      {/if}
    </p>

    {#if deadStep}
      <p class="dead">
        The step function's slope is <strong>zero everywhere</strong> — it never
        whispers "a little more" or "a little less." Nothing downstream can
        learn from it. This single fact is why the next two chapters exist.
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    {#each Object.entries(FNS) as [key, def]}
      <button
        type="button"
        class:active={fnKey === key}
        onclick={() => (fnKey = key as FnKey)}
      >
        {def.label}
      </button>
    {/each}
    <label>input x <input type="range" min="-3" max="3" step="0.05" bind:value={x} /> {fmt(x)}</label>
    <label><input type="checkbox" bind:checked={showDerivative} /> show slope</label>
  {/snippet}
</FigureShell>

<style>
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.5rem 0 0;
  }
  .readout .gold {
    color: var(--accent-gold-ink);
  }
  .dead {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    background: var(--paper);
    border-left: 3px solid var(--accent-gold);
    padding: 0.6rem 0.9rem;
    margin: 0.75rem 0 0;
  }
  button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
</style>
```

- [ ] **Step 2: Write `src/content/chapters/activation-functions.mdx`:**

````mdx
---
number: 2
title: Activation Functions
summary: Why networks need to bend — and what happens when they can't.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import ActivationExplorer from '../../components/figures/activation/ActivationExplorer.svelte';

Chapter 1 left us stuck at a wall: a neuron's decision is a straight line,
and XOR laughs at straight lines. Your first instinct might be to throw more
neurons at the problem — feed the outputs of two neurons into a third. Layers!
Surely stacking decisions makes them smarter.

Here's the uncomfortable truth: **stacking linear things gives you another
linear thing.** A weighted sum of weighted sums is... a weighted sum. You can
pile up a hundred layers of pure `w·x + b` and the whole tower collapses,
mathematically, into one straight line. Same wall, more paperwork.

Something in the pipeline has to *bend*. That something is the **activation
function** — the σ wrapped around every neuron's sum. Chapter 1's perceptron
used the bluntest bend there is: a hard step from 0 to 1. Meet the rest of
the family:

<ActivationExplorer client:visible />

Each has a personality:

- **Step** — the 1958 original. Decisive, binary… and, as you may have
  discovered with the "show slope" toggle, *unteachable*. Its slope is zero
  everywhere, so it offers no hint about which direction would improve
  things. A yes/no oracle that never explains itself.
- **Sigmoid** — a step that went to finishing school. Smoothly squashes
  anything into (0, 1), and its slope is alive near the middle. The classic
  choice for "probability-flavored" outputs.
- **Tanh** — sigmoid's centered sibling, squashing into (−1, 1). Outputs
  hover around zero, which keeps layered networks better behaved. It's what
  our engine uses by default.
- **ReLU** — embarrassingly simple: negative in, zero out; positive in,
  unchanged. The workhorse of deep learning, mostly because it's cheap and
  its slope doesn't fade at large inputs the way the squashers' do.

The pattern to notice: every *useful* activation has a meaningful slope
somewhere. That slope is the entire secret of learning. It answers the only
question that matters: *"if I nudge the input a little, which way does the
output move?"* A function that answers honestly can be taught. A function
that shrugs — the step — cannot.

<MathAside>

Composition of affine maps is affine: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x +
(W_2b_1 + b_2)$ — one matrix, one bias, one line. Nonlinearity between
layers breaks the collapse. The derivatives that matter later:
$\sigma'(x) = \sigma(x)(1-\sigma(x))$ for sigmoid,
$1 - \tanh^2(x)$ for tanh, and $\mathbb{1}[x > 0]$ for ReLU.

</MathAside>

So a bendy function lets layers genuinely add power, and a live slope tells
us *which way to nudge*. Nudge toward what, exactly? We need a way to score
how wrong the network is — and then a way to roll downhill on that score.
That's the next chapter.

<EngineerFooter>

An activation is an elementwise nonlinearity — `map(σ)` over a layer's
output vector, a few nanoseconds per element. ReLU is literally
`max(0, x)`: one compare, no transcendentals, which is half the reason it
won. The "linear layers collapse" fact is worth internalizing: without
activations, a billion-parameter network has exactly the expressive power
of one matrix multiply.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build`, all green; then `grep -c "astro-island" dist/chapters/activation-functions/index.html` ≥ 1, and `grep -c "activation-functions" dist/chapters/the-neuron/index.html` ≥ 1 (chapter 1's "next" link now chains).

- [ ] **Step 4: Commit**

```bash
git add src/components/figures/activation/ src/content/chapters/activation-functions.mdx
git commit -m "feat: chapter 2 — activation functions with explorer figure"
```

---

### Task 4: Chapter 3 — Gradient Descent figure + MDX

**Files:**
- Create: `src/components/figures/gradient/GradientDescentFigure.svelte`
- Create: `src/content/chapters/loss-and-gradient-descent.mdx`

- [ ] **Step 1: Write `src/components/figures/gradient/GradientDescentFigure.svelte`:**

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { linearScale } from '../../../lib/viz/scale';
  import { mulberry32, gaussian } from '../../../lib/nn/rng';
  import { createTicker } from '../../../lib/viz/ticker';

  // One-parameter regression: find w minimizing L(w) = Σ (w·xᵢ − yᵢ)² over 12
  // fixed points drawn from y = 0.8x + noise. SUM (not mean) keeps the bowl
  // steep: learning rates beyond ~0.25 visibly oscillate, then diverge.
  const rng = mulberry32(9);
  const data = Array.from({ length: 12 }, () => {
    const px = rng() * 2 - 1;
    return { x: px, y: 0.8 * px + gaussian(rng) * 0.15 };
  });

  const loss = (w: number) => data.reduce((s, p) => s + (w * p.x - p.y) ** 2, 0);
  const dLoss = (w: number) => data.reduce((s, p) => s + 2 * (w * p.x - p.y) * p.x, 0);

  const W0 = 3.2;
  const W_MIN = -2;
  const W_MAX = 3.6;

  let w = $state(W0);
  let lr = $state(0.04);
  let steps = $state(0);
  let trail: number[] = $state([]);
  let playing = $state(false);
  let diverged = $state(false);

  const sx = linearScale([W_MIN, W_MAX], [12, 328]);
  const maxL = Math.max(loss(W_MIN), loss(W_MAX)) * 1.05;
  const sy = linearScale([0, maxL], [228, 12]);

  const curve = (() => {
    const pts: string[] = [];
    for (let i = 0; i <= 140; i++) {
      const wi = W_MIN + ((W_MAX - W_MIN) * i) / 140;
      pts.push(`${sx(wi)},${sy(loss(wi))}`);
    }
    return pts.join(' ');
  })();

  const ballX = $derived(sx(Math.max(W_MIN, Math.min(W_MAX, w))));
  const ballY = $derived(sy(Math.min(loss(w), maxL)));

  function step() {
    if (diverged) return;
    trail = [...trail.slice(-14), w];
    w = w - lr * dLoss(w);
    steps += 1;
    if (!Number.isFinite(w) || Math.abs(w) > 8) {
      diverged = true;
      stopPlay();
    }
  }

  const ticker = createTicker(step, { fps: 4 });
  onDestroy(() => ticker.stop());

  function stopPlay() {
    playing = false;
    ticker.stop();
  }

  function togglePlay() {
    if (diverged) return;
    playing = !playing;
    if (playing) ticker.start();
    else ticker.stop();
  }

  function reset() {
    stopPlay();
    diverged = false;
    w = W0;
    steps = 0;
    trail = [];
  }

  const fmt = (v: number) => v.toFixed(3);
</script>

<FigureShell
  title="Roll downhill"
  caption="The curve is the loss for every possible weight. The ball is where the weight is now. Each step follows the slope — scaled by the learning rate."
>
  {#snippet children()}
    <svg viewBox="0 0 340 240" aria-label="Loss curve over the weight, with a ball descending by gradient steps">
      <polyline points={curve} fill="none" stroke="var(--ink-strong)" stroke-width="2" />
      {#each trail as tw, i}
        <circle
          cx={sx(Math.max(W_MIN, Math.min(W_MAX, tw)))}
          cy={sy(Math.min(loss(tw), maxL))}
          r="4"
          fill="var(--accent-red)"
          opacity={(0.12 + (0.5 * (i + 1)) / trail.length).toFixed(2)}
        />
      {/each}
      <circle cx={ballX} cy={ballY} r="7" fill="var(--accent-red)" />
    </svg>

    <p class="readout">
      step {steps} &nbsp;·&nbsp; w = {fmt(w)} &nbsp;·&nbsp; loss = {fmt(Math.min(loss(w), 9999))}
    </p>

    {#if diverged}
      <p class="boom">
        💥 <strong>Diverged.</strong> Each step overshot the valley by more than
        the last — the ball bounced out entirely. This isn't a bug; it's what a
        too-large learning rate does. Reset and try a smaller one.
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>learning rate <input type="range" min="0.005" max="0.4" step="0.005" bind:value={lr} /> {lr.toFixed(3)}</label>
    <button type="button" onclick={step} disabled={diverged}>Step</button>
    <button type="button" onclick={togglePlay} disabled={diverged}>{playing ? 'Pause' : 'Play'}</button>
    <button type="button" onclick={reset}>Reset</button>
  {/snippet}
</FigureShell>

<style>
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.5rem 0 0;
  }
  .boom {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    background: var(--paper);
    border-left: 3px solid var(--accent-red);
    padding: 0.6rem 0.9rem;
    margin: 0.75rem 0 0;
  }
</style>
```

- [ ] **Step 2: Write `src/content/chapters/loss-and-gradient-descent.mdx`:**

````mdx
---
number: 3
title: Loss & Gradient Descent
summary: Roll downhill on the error surface — and learn why step size is everything.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import GradientDescentFigure from '../../components/figures/gradient/GradientDescentFigure.svelte';

In chapter 1 you tuned weights by hand, eyeballing which slider made things
better. To automate that, the machine needs what you had: a sense of
*better*. That sense is a single number called the **loss** — a score of how
wrong the network currently is. Zero means perfect; bigger means worse. One
honest number that the machine can try to shrink.

The classic choice is squared error: for each example, take the difference
between prediction and truth, square it (so misses in either direction count,
and big misses count *a lot*), and add them up.

Now the beautiful move. Imagine computing the loss not just for the current
weights, but for *every possible* setting of a weight. Plot it: loss on the
vertical, the weight on the horizontal. You get a landscape — here a simple
valley. Somewhere at the bottom is the weight that makes the loss smallest.
Finding it is no longer a learning problem; it's a *hiking* problem: you're
standing on a hillside in fog, and you want the valley floor.

You can't see the bottom. But you can feel the slope under your feet — and
walk downhill:

<GradientDescentFigure client:visible />

Two things to try before reading on. First, run it with the default step
size and watch the ball settle into the valley — that's **gradient descent**,
the algorithm that trains every neural network you've ever heard of. Second,
crank the learning rate up past ~0.3 and watch the ball overshoot the valley,
ping-pong between the walls, and finally fly out of the plot. 💥

That explosion is the most instructive failure in machine learning. The
**learning rate** scales each step. Too small: you inch along, wasting time.
Too large: each hop overshoots farther than the last, and the loss climbs
instead of falling. Every practitioner has watched a training run print
`loss: NaN` because of exactly this. Now you've *seen* it happen, in one
dimension, in slow motion.

Real networks have millions of weights, so the landscape isn't a curve —
it's a surface in millions of dimensions, full of gullies and plateaus. But
the algorithm doesn't change: feel the slope in every direction at once,
step downhill, repeat.

Which leaves one question — the big one. Feeling the slope means knowing, for
*each* weight, how the loss would change if you nudged it. Millions of
weights, one loss. How can you possibly compute all those slopes without
re-running the network millions of times? That trick has a name:
backpropagation. Next chapter.

<MathAside>

Squared-error loss over a dataset: $L(w) = \sum_i (f_w(x_i) - y_i)^2$. The
gradient $\nabla L$ is the vector of partial derivatives $\partial L /
\partial w_j$ — it points uphill, so learning steps go the other way:
$w \leftarrow w - \eta \, \nabla L(w)$, where $\eta$ is the learning rate.
For a quadratic valley, steps converge when $\eta < 2/L''$ and oscillate or
diverge beyond it — the figure's blow-up is that inequality, live.

</MathAside>

<EngineerFooter>

The entire optimizer is three lines: `zero the grads`, `compute grads`,
`w -= lr * grad`. That loop — not anything exotic — is what burns the GPU
hours. Modern optimizers (Adam, momentum) are refinements that adapt the
step per-weight, but they're seasoning on the same dish. The `loss: NaN`
you'll someday see in production is the figure's 💥 wearing a stack trace.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build` green; `grep -c "astro-island" dist/chapters/loss-and-gradient-descent/index.html` ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add src/components/figures/gradient/ src/content/chapters/loss-and-gradient-descent.mdx
git commit -m "feat: chapter 3 — loss and gradient descent with rolling-ball figure"
```

---

### Task 5: Chapter 4 — Backprop Stepper + MDX

**Files:**
- Create: `src/components/figures/backprop/BackpropStepper.svelte`
- Create: `src/content/chapters/backpropagation.mdx`

- [ ] **Step 1: Write `src/components/figures/backprop/BackpropStepper.svelte`:**

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { Value } from '../../../lib/nn/value';
  import { createTicker } from '../../../lib/viz/ticker';

  // One neuron, one training example, computed by the REAL engine:
  // L = (tanh(w1*x1 + w2*x2 + b) - t)^2
  const w1 = new Value(0.9);
  const x1 = new Value(0.5);
  const w2 = new Value(-0.4);
  const x2 = new Value(-1.0);
  const b = new Value(0.1);
  const t = 1.0;
  const m1 = w1.mul(x1);
  const m2 = w2.mul(x2);
  const s = m1.add(m2);
  const z = s.add(b);
  const y = z.tanh();
  const d = y.sub(t);
  const L = d.pow(2);
  L.backward();

  interface GraphNode {
    id: string;
    label: string;
    v: Value;
    x: number;
    y: number;
  }

  const NODES: GraphNode[] = [
    { id: 'w1', label: 'w₁', v: w1, x: 52, y: 42 },
    { id: 'x1', label: 'x₁', v: x1, x: 52, y: 106 },
    { id: 'w2', label: 'w₂', v: w2, x: 52, y: 170 },
    { id: 'x2', label: 'x₂', v: x2, x: 52, y: 234 },
    { id: 'b', label: 'b', v: b, x: 52, y: 292 },
    { id: 'm1', label: 'w₁·x₁', v: m1, x: 180, y: 74 },
    { id: 'm2', label: 'w₂·x₂', v: m2, x: 180, y: 202 },
    { id: 's', label: 'sum', v: s, x: 300, y: 138 },
    { id: 'z', label: '+ b', v: z, x: 404, y: 202 },
    { id: 'y', label: 'tanh', v: y, x: 500, y: 138 },
    { id: 'd', label: '− t', v: d, x: 592, y: 202 },
    { id: 'L', label: '(·)²', v: L, x: 648, y: 106 },
  ];
  const EDGES: Array<[string, string]> = [
    ['w1', 'm1'], ['x1', 'm1'], ['w2', 'm2'], ['x2', 'm2'],
    ['m1', 's'], ['m2', 's'], ['s', 'z'], ['b', 'z'],
    ['z', 'y'], ['y', 'd'], ['d', 'L'],
  ];
  const byId = new Map(NODES.map((n) => [n.id, n]));

  const f2 = (v: number) => v.toFixed(2);
  const f3 = (v: number) => v.toFixed(3);

  const FORWARD: Array<{ id: string; text: string }> = [
    { id: 'm1', text: `w₁·x₁ = ${f2(w1.data)} × ${f2(x1.data)} = ${f3(m1.data)}` },
    { id: 'm2', text: `w₂·x₂ = ${f2(w2.data)} × ${f2(x2.data)} = ${f3(m2.data)}` },
    { id: 's', text: `sum = ${f3(m1.data)} + ${f3(m2.data)} = ${f3(s.data)}` },
    { id: 'z', text: `add the bias: ${f3(s.data)} + ${f2(b.data)} = ${f3(z.data)}` },
    { id: 'y', text: `squash: tanh(${f3(z.data)}) = ${f3(y.data)}` },
    { id: 'd', text: `miss: prediction ${f3(y.data)} − target ${f2(t)} = ${f3(d.data)}` },
    { id: 'L', text: `loss: (${f3(d.data)})² = ${f3(L.data)}` },
  ];

  const BACKWARD: Array<{ id: string; text: string }> = [
    { id: 'L', text: '∂L/∂L = 1 — the blame starts, whole, at the loss' },
    { id: 'd', text: `∂L/∂d = 2·d = ${f3(d.grad)} (square doubles the blame)` },
    { id: 'y', text: `∂L/∂y = ${f3(y.grad)} (subtraction passes blame straight through)` },
    { id: 'z', text: `∂L/∂z = ∂L/∂y × (1 − tanh²z) = ${f3(z.grad)} (squasher damps it)` },
    { id: 's', text: `∂L/∂s = ${f3(s.grad)} (addition copies blame to each input)` },
    { id: 'b', text: `∂L/∂b = ${f3(b.grad)} — the bias's share, ready for a nudge` },
    { id: 'm1', text: `∂L/∂m₁ = ${f3(m1.grad)}` },
    { id: 'm2', text: `∂L/∂m₂ = ${f3(m2.grad)}` },
    { id: 'w1', text: `∂L/∂w₁ = ∂L/∂m₁ × x₁ = ${f3(w1.grad)} (multiply swaps in the partner)` },
    { id: 'w2', text: `∂L/∂w₂ = ∂L/∂m₂ × x₂ = ${f3(w2.grad)}` },
  ];

  const TOTAL = FORWARD.length + BACKWARD.length;

  let stepIndex = $state(0);
  let playing = $state(false);

  const phase = $derived(
    stepIndex === 0 ? 'ready' : stepIndex <= FORWARD.length ? 'forward' : 'backward'
  );
  const current = $derived(
    stepIndex === 0
      ? null
      : stepIndex <= FORWARD.length
        ? FORWARD[stepIndex - 1]
        : BACKWARD[stepIndex - FORWARD.length - 1]
  );
  const revealedData = $derived(
    new Set([
      'w1', 'x1', 'w2', 'x2', 'b',
      ...FORWARD.slice(0, Math.min(stepIndex, FORWARD.length)).map((f) => f.id),
    ])
  );
  const revealedGrad = $derived(
    new Set(
      stepIndex > FORWARD.length
        ? BACKWARD.slice(0, stepIndex - FORWARD.length).map((s2) => s2.id)
        : []
    )
  );

  function advance() {
    if (stepIndex < TOTAL) stepIndex += 1;
    if (stepIndex >= TOTAL) stopPlay();
  }

  const ticker = createTicker(advance, { fps: 1.1 });
  onDestroy(() => ticker.stop());

  function stopPlay() {
    playing = false;
    ticker.stop();
  }

  function togglePlay() {
    playing = !playing;
    if (playing) {
      if (stepIndex >= TOTAL) stepIndex = 0;
      ticker.start();
    } else {
      ticker.stop();
    }
  }

  function reset() {
    stopPlay();
    stepIndex = 0;
  }
</script>

<FigureShell
  title="The chain rule, one step at a time"
  caption="Forward: compute left to right. Backward: blame flows right to left. Every number here comes from the site's own autograd engine."
>
  {#snippet children()}
    <svg viewBox="0 0 700 330" aria-label="Computation graph of one neuron; stepping reveals forward values then backward gradients">
      {#each EDGES as [from, to]}
        {@const a = byId.get(from)!}
        {@const c = byId.get(to)!}
        <line x1={a.x + 34} y1={a.y} x2={c.x - 34} y2={c.y} stroke="var(--line)" stroke-width="1.5" />
      {/each}
      {#each NODES as n}
        {@const isCurrent = current?.id === n.id}
        <g>
          <rect
            x={n.x - 34}
            y={n.y - 26}
            width="68"
            height="52"
            rx="10"
            fill="var(--paper-raised)"
            stroke={isCurrent ? 'var(--accent-red)' : 'var(--line)'}
            stroke-width={isCurrent ? 3 : 1.5}
          />
          <text x={n.x} y={n.y - 10} text-anchor="middle" class="n-label">{n.label}</text>
          {#if revealedData.has(n.id)}
            <text x={n.x} y={n.y + 6} text-anchor="middle" class="n-data">{f3(n.v.data)}</text>
          {/if}
          {#if revealedGrad.has(n.id)}
            <text x={n.x} y={n.y + 20} text-anchor="middle" class="n-grad">∂ {f3(n.v.grad)}</text>
          {/if}
        </g>
      {/each}
    </svg>

    <p class="explain" data-phase={phase}>
      {#if current}
        <span class="phase label">{phase}</span> {current.text}
      {:else}
        <span class="phase label">ready</span> Press Step (or Play) to run the forward pass, then watch the blame flow back.
      {/if}
    </p>
  {/snippet}

  {#snippet controls()}
    <button type="button" onclick={advance} disabled={stepIndex >= TOTAL}>Step</button>
    <button type="button" onclick={togglePlay}>{playing ? 'Pause' : 'Play'}</button>
    <button type="button" onclick={reset}>Reset</button>
    <span class="progress">{stepIndex} / {TOTAL}</span>
  {/snippet}
</FigureShell>

<style>
  .n-label {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 700;
    fill: var(--ink-strong);
  }
  .n-data {
    font-family: var(--font-code);
    font-size: 12px;
    fill: var(--link);
  }
  .n-grad {
    font-family: var(--font-code);
    font-size: 12px;
    fill: var(--accent-gold-ink);
  }
  .explain {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    color: var(--ink);
    background: var(--paper);
    border-left: 3px solid var(--line);
    padding: 0.6rem 0.9rem;
    margin: 0.75rem 0 0;
    min-height: 2.4em;
  }
  .explain[data-phase='backward'] {
    border-left-color: var(--accent-gold);
  }
  .phase {
    margin-right: 0.5rem;
  }
  .progress {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--faint);
  }
</style>
```

- [ ] **Step 2: Write `src/content/chapters/backpropagation.mdx`:**

````mdx
---
number: 4
title: Backpropagation
summary: Blame flows backward - how every weight learns its share of the error.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import BackpropStepper from '../../components/figures/backprop/BackpropStepper.svelte';

Last chapter ended on a cliffhanger: gradient descent needs the slope of the
loss with respect to *every* weight — millions of them — and re-running the
network once per weight would take geologic time. The answer is an algorithm
so central that everything else in deep learning is a footnote to it.

The insight: a network isn't a monolith. It's a **graph of tiny operations**
— multiply here, add there, squash at the end — and each operation is so
simple that it knows its own local slope by heart. A multiply knows that
nudging one input scales through by the *other* input. An add passes nudges
straight through. A tanh damps them by `1 − tanh²`.

**Backpropagation** is bookkeeping over those local facts. Run the graph
forward to get the loss. Then walk *backward*, and at each node apply one
rule — the chain rule: *blame arriving at a node's output, times the node's
local slope, is the blame for its input.* By the time you reach the weights,
each one holds exactly ∂loss/∂weight. One forward pass, one backward pass,
every slope. Step through it yourself:

<BackpropStepper client:visible />

Watch the backward phase closely and you'll see the personalities: the
square *doubles* blame, the add *copies* it to both parents, the multiply
*swaps in the partner* (w₁'s blame gets scaled by x₁ — how much w₁ mattered
depends on what it was multiplied by), and the tanh *damps* blame by its
slope. When its input is extreme, tanh's slope is nearly zero — blame dies
there. That's the "vanishing gradient," and it will haunt us in chapter 6.

Here's the part this site is built to show you. The figure isn't an
illustration *of* the algorithm — it *ran* the algorithm. Every number came
from the same ~120-line engine that trains networks throughout this site,
and its whole backward pass fits on a napkin:

```ts
// value.ts — the heart of it. Each op records how to route blame:
mul(other) {
  const out = new Value(this.data * other.data, [this, other]);
  out._backward = () => {
    this.grad += other.data * out.grad;   // partner swaps in
    other.grad += this.data * out.grad;
  };
  return out;
}

backward() {
  // topologically sort the graph, then apply the chain rule in reverse
  this.grad = 1;
  for (const node of reverseTopoOrder(this)) node._backward();
}
```

That's not pseudocode — it's lightly trimmed from
[`src/lib/nn/value.ts`](https://github.com/christopherhan/neural_networks/blob/main/src/lib/nn/value.ts)
in this site's repository. PyTorch and JAX are this exact idea, plus tensors,
plus a decade of engineering.

<MathAside>

For a composition $L = f(g(w))$, the chain rule gives $\frac{\partial
L}{\partial w} = \frac{\partial L}{\partial g}\cdot\frac{\partial
g}{\partial w}$. Backprop applies this at every node in reverse topological
order, accumulating with $+{=}$ when a value feeds multiple consumers.
Cost: one backward pass is a small constant times the forward pass,
regardless of parameter count — that constant-factor miracle is why
million-parameter training is possible at all.

</MathAside>

Now we hold all the pieces: neurons that bend (ch. 2), a score to shrink
(ch. 3), and — as of this chapter — every slope, cheaply. There is nothing
left to learn before *learning itself*. Next: put it all together and watch
a network train, live.

<EngineerFooter>

Backprop is reverse-mode automatic differentiation: a reverse topological
traversal where each op applies its local Jacobian to the incoming
gradient. It's `O(forward cost)` per backward pass — independent of
parameter count, which is the whole ballgame. The `+=` accumulation (a
value used twice gets blame from both uses) and the "zero grads between
steps" contract are the two details that bite people implementing it from
scratch. You've now read a complete implementation; nothing in
`torch.autograd` will ever feel mystical again.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build` green; `grep -c "astro-island" dist/chapters/backpropagation/index.html` ≥ 1; `grep -c "katex" dist/chapters/backpropagation/index.html` ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add src/components/figures/backprop/ src/content/chapters/backpropagation.mdx
git commit -m "feat: chapter 4 — backpropagation with engine-powered stepper"
```

---

### Task 6: Chapter 5 — Training Playground flagship + MDX

**Files:**
- Create: `src/components/figures/playground/TrainingPlayground.svelte`
- Create: `src/content/chapters/training-playground.mdx`

- [ ] **Step 1: Write `src/components/figures/playground/TrainingPlayground.svelte`:**

```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { MLP, trainStep } from '../../../lib/nn/network';
  import { mulberry32 } from '../../../lib/nn/rng';
  import {
    PLAYGROUND_DATASETS,
    type PlaygroundDatasetKey,
    type Point2,
  } from '../../../lib/nn/datasets';
  import { boundaryGrid, BOUNDARY_DOMAIN } from '../../../lib/viz/boundary';
  import { createTicker } from '../../../lib/viz/ticker';

  const ARCHS = {
    small: { label: '8 neurons', sizes: [2, 8, 1] },
    deep: { label: '8 × 8 deep', sizes: [2, 8, 8, 1] },
  } as const;
  type ArchKey = keyof typeof ARCHS;

  const GRID = 60;
  const SIZE = 340;
  const EPOCHS_PER_TICK = 5;

  let datasetKey: PlaygroundDatasetKey = $state('xor');
  let archKey: ArchKey = $state('small');
  let lr = $state(0.1);
  let playing = $state(false);
  let diverged = $state(false);
  let epoch = $state(0);
  let lossNow = $state(NaN);
  let lossHistory: number[] = $state([]);

  // Training state is deliberately non-reactive: the canvas redraw is manual.
  let points: Point2[] = PLAYGROUND_DATASETS[datasetKey].generate();
  let inputs = points.map((p) => [p.x, p.y]);
  let targets = points.map((p) => [p.label as number]);
  let mlp = new MLP(ARCHS[archKey].sizes, mulberry32(7));

  let canvas: HTMLCanvasElement;
  let off: HTMLCanvasElement | null = null;

  function rebuild() {
    stopPlay();
    points = PLAYGROUND_DATASETS[datasetKey].generate();
    inputs = points.map((p) => [p.x, p.y]);
    targets = points.map((p) => [p.label as number]);
    mlp = new MLP(ARCHS[archKey].sizes, mulberry32(7));
    epoch = 0;
    lossNow = NaN;
    lossHistory = [];
    diverged = false;
    draw();
  }

  function selectDataset(k: PlaygroundDatasetKey) {
    datasetKey = k;
    rebuild();
  }

  function selectArch(k: ArchKey) {
    archKey = k;
    rebuild();
  }

  function epochStep(count: number) {
    for (let i = 0; i < count; i++) {
      lossNow = trainStep(mlp, inputs, targets, lr);
      epoch += 1;
    }
    lossHistory = [...lossHistory.slice(-199), lossNow];
    if (!Number.isFinite(lossNow)) {
      diverged = true;
      stopPlay();
    }
    draw();
  }

  function draw() {
    if (!canvas) return;
    if (!off) {
      off = document.createElement('canvas');
      off.width = GRID;
      off.height = GRID;
    }
    const grid = boundaryGrid((x, y) => mlp.predict([x, y])[0], GRID);
    const octx = off.getContext('2d')!;
    const img = octx.createImageData(GRID, GRID);
    for (let i = 0; i < grid.length; i++) {
      const v = Math.max(-1, Math.min(1, grid[i]));
      const o = i * 4;
      if (v >= 0) {
        img.data[o] = 38;
        img.data[o + 1] = 139;
        img.data[o + 2] = 210; // --accent-blue
      } else {
        img.data[o] = 220;
        img.data[o + 1] = 50;
        img.data[o + 2] = 47; // --accent-red
      }
      img.data[o + 3] = Math.round(Math.abs(v) * 140) + 20;
    }
    octx.putImageData(img, 0, 0);

    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, SIZE, SIZE);

    const px = (v: number) => ((v + BOUNDARY_DOMAIN) / (2 * BOUNDARY_DOMAIN)) * SIZE;
    const py = (v: number) => ((BOUNDARY_DOMAIN - v) / (2 * BOUNDARY_DOMAIN)) * SIZE;
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = p.label === 1 ? '#268bd2' : '#dc322f';
      ctx.fill();
      ctx.strokeStyle = '#fffdf5';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  const ticker = createTicker(() => epochStep(EPOCHS_PER_TICK), { fps: 20 });
  onDestroy(() => ticker.stop());

  function stopPlay() {
    playing = false;
    ticker.stop();
  }

  function togglePlay() {
    if (diverged) return;
    playing = !playing;
    if (playing) ticker.start();
    else ticker.stop();
  }

  onMount(() => {
    draw();
  });

  const sparkline = $derived.by(() => {
    if (lossHistory.length < 2) return '';
    const max = Math.max(...lossHistory, 0.001);
    const last = lossHistory.length - 1;
    return lossHistory
      .map((l, i) => `${((i / last) * 196 + 2).toFixed(1)},${(34 - Math.min(l / max, 1) * 30).toFixed(1)}`)
      .join(' ');
  });
  const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(4) : '—');
</script>

<FigureShell
  title="The training playground"
  caption="A real network, training in your browser tab. Blue region: the network says +1. Red: −1. The dots are the truth it's learning from."
>
  {#snippet children()}
    <canvas bind:this={canvas} width={SIZE} height={SIZE} aria-label="Decision boundary heatmap with training points; colors update live as the network trains"></canvas>

    <div class="stats">
      <span>epoch {epoch}</span>
      <span>loss {fmt(lossNow)}</span>
      {#if lossHistory.length >= 2}
        <svg viewBox="0 0 200 36" class="spark" aria-label="Loss over recent epochs">
          <polyline points={sparkline} fill="none" stroke="var(--accent-green)" stroke-width="1.5" />
        </svg>
      {/if}
    </div>

    {#if diverged}
      <p class="boom">
        💥 <strong>Loss exploded</strong> — chapter 3's too-hot learning rate,
        now in the wild. Reset, lower the rate, and try again.
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    {#each Object.entries(PLAYGROUND_DATASETS) as [key, def]}
      <button
        type="button"
        class:active={datasetKey === key}
        onclick={() => selectDataset(key as PlaygroundDatasetKey)}
      >
        {def.label}
      </button>
    {/each}
    {#each Object.entries(ARCHS) as [key, def]}
      <button
        type="button"
        class:active={archKey === key}
        onclick={() => selectArch(key as ArchKey)}
      >
        {def.label}
      </button>
    {/each}
    <label>learning rate <input type="range" min="0.01" max="1" step="0.01" bind:value={lr} /> {lr.toFixed(2)}</label>
    <button type="button" onclick={togglePlay} disabled={diverged}>{playing ? 'Pause' : 'Train'}</button>
    <button type="button" onclick={() => epochStep(1)} disabled={diverged}>Step</button>
    <button type="button" onclick={rebuild}>Reset</button>
  {/snippet}
</FigureShell>

<style>
  canvas {
    width: 100%;
    max-width: 340px;
    display: block;
    margin: 0 auto;
    border: 1.5px solid var(--line-soft);
    border-radius: 10px;
    background: var(--paper);
  }
  .stats {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.25rem;
    font-family: var(--font-ui);
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.5rem;
  }
  .spark {
    width: 120px;
    height: 22px;
  }
  .boom {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    background: var(--paper);
    border-left: 3px solid var(--accent-red);
    padding: 0.6rem 0.9rem;
    margin: 0.75rem 0 0;
  }
  button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
</style>
```

- [ ] **Step 2: Write `src/content/chapters/training-playground.mdx`:**

````mdx
---
number: 5
title: The Training Playground
summary: Everything so far, live - train a real network and watch it carve up the plane.
---

import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import TrainingPlayground from '../../components/figures/playground/TrainingPlayground.svelte';

Four chapters of parts. Time to turn the key.

Below is a complete neural network — real weights, real tanh neurons, the
real autograd engine from last chapter — training **live in your browser
tab**, right now, on whatever you throw at it. The colored regions are the
network's current opinion about every point in the plane; the dots are the
truth it's being graded against. Press **Train** and watch opinion bend
toward truth.

<TrainingPlayground client:visible />

A guided tour of things worth trying:

1. **XOR, at last.** It starts on the dataset that humiliated the
   perceptron. Press Train. Within a few hundred epochs, the boundary folds
   itself into the four-quadrant checkerboard no single line could draw.
   Chapter 1's wall, demolished in about two seconds — this is what "bendy
   functions plus layers" buys.
2. **Circles.** A ring around a disk. Watch the network discover it needs a
   *closed curve* — early on the boundary is a clumsy blob, then it tightens
   around the inner cluster.
3. **Spiral, with the small network.** Here's a healthy failure: 8 neurons
   genuinely struggle to wrap two interlocking arms. Watch it try — then
   switch to **8 × 8 deep** and watch capacity matter. This intuition (some
   shapes need more network) is half of chapter 6.
4. **Blow it up.** Set the learning rate to 1.0 on any dataset. The loss
   sparkline goes jagged, then 💥. Chapter 3's rolling ball, now with a few
   dozen dimensions doing the rolling. Reset, drop the rate, feel the
   difference.
5. **Watch the loss curve, not just the picture.** The sparkline is what
   practitioners actually stare at all day. Smooth descent: learning.
   Plateaus: stuck. Jagged sawtooth: rate too hot.

Every idea from Part I is on this screen at once. Neurons computing weighted
sums (ch. 1), tanh bending them (ch. 2), a loss being driven downhill
(ch. 3), and backprop supplying every gradient for every step (ch. 4). There
is no other machinery. What you're watching is, at toy scale, exactly what
happens on a GPU cluster when someone trains the models in Part III — same
loop, more zeros.

Part I is complete. Part II asks the natural next questions: what happens
when you stack *many* layers (and why gradients vanish on the way down),
and what happens when you shape the network to match the data — grids for
images, loops for sequences.

<EngineerFooter>

What separates this toy from production training: batching (we do
full-batch; real training samples minibatches), tensors (we push scalars
through a graph; frameworks push matrices through fused GPU kernels), an
optimizer with state (Adam keeps running moments per weight), and
regularization. What's identical: the loop. `forward → loss → backward →
step`, millions of times. If you can read this page's source — and you can,
it's ~400 lines all-in — you already understand the skeleton of every
training run on earth.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build` green; `grep -c "astro-island" dist/chapters/training-playground/index.html` ≥ 1.

- [ ] **Step 4: Manual smoke via preview** — start the preview server (`.claude/launch.json` config `preview` serves the built dist) or `npm run dev`, load `/chapters/training-playground/`, press Train on XOR, confirm the boundary visibly changes and epoch counts climb; switch datasets; hit lr=1.0 and confirm the divergence message instead of a crash. Report observations (this step is observational — UI drive-through, not a test assertion).

- [ ] **Step 5: Commit**

```bash
git add src/components/figures/playground/ src/content/chapters/training-playground.mdx
git commit -m "feat: chapter 5 — the training playground flagship"
```

---

### Task 7: Integration verification

**Files:** none created — verification only (fixes allowed if issues found, committed separately with clear messages).

- [ ] **Step 1: Full suite** — `npm test && npm run check && npm run build` → 74 tests, 0 errors, Complete.

- [ ] **Step 2: Link integrity incl. new chapter chain** —

```bash
npx linkinator ./dist --recurse --silent --skip "^https?://(?!localhost)"
grep -c 'chapters/activation-functions/' dist/chapters/the-neuron/index.html        # ≥1 (ch1 → ch2 next-link)
grep -c 'chapters/training-playground/' dist/chapters/backpropagation/index.html     # ≥1 (ch4 → ch5)
grep -c 'Next: coming soon' dist/chapters/training-playground/index.html             # ≥1 (ch5 next is unwritten ch6)
grep -o '<a href="/chapters/[^"]*"' dist/index.html | sort -u | wc -l               # 5 (trail map: five solid stops)
```

- [ ] **Step 3: Base-path build guard** —

```bash
SITE_URL=https://example.github.io BASE_PATH=/neural_networks npm run build
if grep -rlE '(href|src|component-url|renderer-url)="/(chapters/|_astro/)' dist --include='*.html'; then echo FAIL; else echo OK; fi
npm run build   # restore root-base dist
```

- [ ] **Step 4: Commit any fixes** made during verification with descriptive messages; if none, no commit.

---

## Out of scope for Stage 2b

- Part II chapters (6–8), ScrollStage scrollytelling, MLA/attention anything.
- Optimizations (param caching, batch predict) unless the playground measurably janks during Task 6's manual smoke — in which case reduce `GRID` to 48 and/or `EPOCHS_PER_TICK` to 3 and note it.
