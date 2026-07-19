# Stage 2a: Micro-ML Engine + First Interactive Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The hand-rolled scalar autograd engine (`src/lib/nn`), seeded toy datasets, viz scale helpers, the reusable Svelte figure shell, and Chapter 1's real interactive figure (perceptron playground with the XOR wall) — replacing the Chapter 1 stub with full content.

**Architecture:** Micrograd-style scalar autograd: a `Value` class holding `data`/`grad` with operator methods that record a backward closure; `backward()` topologically sorts and propagates. `MLP` composes `Neuron`/`Layer` over `Value`. All randomness flows through an injected seeded PRNG (mulberry32) so tests and figures are deterministic. Figures are Svelte 5 islands (`client:visible`) that server-render their initial state as the no-JS fallback; `FigureShell.svelte` provides the shared sketch-frame chrome.

**Tech Stack:** TypeScript (pure, no deps) for `lib/nn` + `lib/viz`; Svelte 5 (runes) for figures; Vitest for tests; existing Astro/MDX pipeline.

**Spec:** `docs/superpowers/specs/2026-07-19-neural-networks-site-design.md`. Stage 2b (chapters 2–5, Training Playground flagship) follows in a separate plan and builds directly on these modules.

**File map:**

```
src/lib/nn/rng.ts            seeded PRNG + gaussian        (Task 1)
src/lib/nn/value.ts          scalar autograd               (Task 2)
src/lib/nn/network.ts        Neuron/Layer/MLP/loss/SGD     (Task 3)
src/lib/nn/datasets.ts       twoClouds, xorQuadrants       (Task 4)
src/lib/viz/scale.ts         linearScale with invert       (Task 5)
src/components/figures/FigureShell.svelte                  (Task 6)
src/components/figures/perceptron/PerceptronPlayground.svelte (Task 6)
src/content/chapters/the-neuron.mdx  (rewritten, full)     (Task 6)
```

---

### Task 1: Seeded RNG (TDD)

**Files:**
- Create: `src/lib/nn/rng.ts`
- Test: `src/lib/nn/rng.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/nn/rng.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { mulberry32, gaussian } from './rng';

describe('mulberry32', () => {
  it('is deterministic for a given seed', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const seqA = Array.from({ length: 5 }, () => a());
    const seqB = Array.from({ length: 5 }, () => b());
    expect(seqA).toEqual(seqB);
  });

  it('differs across seeds', () => {
    expect(mulberry32(1)()).not.toBe(mulberry32(2)());
  });

  it('emits values in [0, 1)', () => {
    const rng = mulberry32(7);
    for (let i = 0; i < 1000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe('gaussian', () => {
  it('is deterministic and roughly standard-normal', () => {
    const rng = mulberry32(42);
    const xs = Array.from({ length: 5000 }, () => gaussian(rng));
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(Math.abs(variance - 1)).toBeLessThan(0.15);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./rng`.

- [ ] **Step 3: Write `src/lib/nn/rng.ts`**

```ts
/** Deterministic PRNG. All randomness in lib/nn flows through an injected Rng
 *  so tests and figures are reproducible. */
export type Rng = () => number;

/** mulberry32 — tiny, fast, good-enough 32-bit PRNG. Returns values in [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal sample via Box–Muller. */
export function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS (13 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/lib/nn/rng.ts src/lib/nn/rng.test.ts
git commit -m "feat: seeded PRNG (mulberry32) and gaussian sampler for lib/nn"
```

---

### Task 2: Scalar autograd `Value` (TDD with numerical gradient checks)

The heart of the engine. Gradients are verified against numerical differentiation — the canonical way to test autograd.

**Files:**
- Create: `src/lib/nn/value.ts`
- Test: `src/lib/nn/value.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/nn/value.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { Value } from './value';

/** Numerical derivative of f at x via central difference. */
function numericGrad(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

describe('Value forward pass', () => {
  it('computes basic ops', () => {
    const a = new Value(2);
    const b = new Value(-3);
    expect(a.add(b).data).toBe(-1);
    expect(a.mul(b).data).toBe(-6);
    expect(a.sub(b).data).toBe(5);
    expect(a.div(new Value(4)).data).toBe(0.5);
    expect(a.pow(3).data).toBe(8);
    expect(a.neg().data).toBe(-2);
    expect(new Value(0).exp().data).toBe(1);
    expect(new Value(0).tanh().data).toBe(0);
    expect(new Value(-1).relu().data).toBe(0);
    expect(new Value(2).relu().data).toBe(2);
    expect(new Value(0).sigmoid().data).toBeCloseTo(0.5);
  });

  it('accepts plain numbers as operands', () => {
    const a = new Value(2);
    expect(a.add(1).data).toBe(3);
    expect(a.mul(3).data).toBe(6);
    expect(a.sub(1).data).toBe(1);
    expect(a.div(2).data).toBe(1);
  });
});

describe('Value backward pass — gradient checks vs numerical differentiation', () => {
  const cases: Array<{
    name: string;
    fv: (x: Value) => Value;
    fn: (x: number) => number;
    at: number[];
  }> = [
    { name: 'add', fv: (x) => x.add(3), fn: (x) => x + 3, at: [2, -1.5] },
    { name: 'mul', fv: (x) => x.mul(-4), fn: (x) => x * -4, at: [2, 0.3] },
    { name: 'pow', fv: (x) => x.pow(3), fn: (x) => x ** 3, at: [2, -1.2] },
    { name: 'div', fv: (x) => new Value(1).div(x), fn: (x) => 1 / x, at: [2, -3] },
    { name: 'exp', fv: (x) => x.exp(), fn: (x) => Math.exp(x), at: [0.5, -1] },
    { name: 'tanh', fv: (x) => x.tanh(), fn: (x) => Math.tanh(x), at: [0.5, -2] },
    { name: 'relu+', fv: (x) => x.relu(), fn: (x) => Math.max(0, x), at: [1.5] },
    { name: 'relu-', fv: (x) => x.relu(), fn: (x) => Math.max(0, x), at: [-1.5] },
    {
      name: 'sigmoid',
      fv: (x) => x.sigmoid(),
      fn: (x) => 1 / (1 + Math.exp(-x)),
      at: [0.7, -0.7],
    },
    {
      name: 'composite tanh(x*3 + x^2)',
      fv: (x) => x.mul(3).add(x.pow(2)).tanh(),
      fn: (x) => Math.tanh(x * 3 + x ** 2),
      at: [0.4, -0.6],
    },
  ];

  for (const { name, fv, fn, at } of cases) {
    for (const x0 of at) {
      it(`${name} at x=${x0}`, () => {
        const x = new Value(x0);
        const y = fv(x);
        y.backward();
        expect(x.grad).toBeCloseTo(numericGrad(fn, x0), 4);
      });
    }
  }

  it('accumulates gradients when a value is used twice', () => {
    // y = x*x + x  =>  dy/dx = 2x + 1
    const x = new Value(3);
    const y = x.mul(x).add(x);
    y.backward();
    expect(x.grad).toBeCloseTo(7);
  });

  it('propagates through multi-variable expressions', () => {
    // z = (a*b + a).tanh(); check both partials numerically
    const f = (a: number, b: number) => Math.tanh(a * b + a);
    const a = new Value(0.5);
    const b = new Value(-1.3);
    const z = a.mul(b).add(a).tanh();
    z.backward();
    expect(a.grad).toBeCloseTo(numericGrad((x) => f(x, -1.3), 0.5), 4);
    expect(b.grad).toBeCloseTo(numericGrad((x) => f(0.5, x), -1.3), 4);
  });

  it('backward on a diamond graph visits each node once', () => {
    // y = (x+x) * (x+x) = 4x^2  =>  dy/dx = 8x
    const x = new Value(2);
    const s = x.add(x);
    const y = s.mul(s);
    y.backward();
    expect(x.grad).toBeCloseTo(16);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./value`.

- [ ] **Step 3: Write `src/lib/nn/value.ts`**

```ts
/**
 * Scalar autograd, micrograd-style. Every arithmetic op returns a new Value
 * that remembers its inputs and how to route gradients back to them.
 * `backward()` topologically sorts the graph and applies the chain rule.
 *
 * This file is deliberately small and readable: Chapter 4 shows it as
 * teaching material.
 */
export class Value {
  data: number;
  grad = 0;
  private _backward: () => void = () => {};
  private readonly _prev: readonly Value[];

  constructor(data: number, prev: readonly Value[] = []) {
    this.data = data;
    this._prev = prev;
  }

  private static of(x: number | Value): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(other: number | Value): Value {
    const o = Value.of(other);
    const out = new Value(this.data + o.data, [this, o]);
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: number | Value): Value {
    const o = Value.of(other);
    const out = new Value(this.data * o.data, [this, o]);
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  /** Raise to a constant power (exponent is not differentiated). */
  pow(n: number): Value {
    const out = new Value(this.data ** n, [this]);
    out._backward = () => {
      this.grad += n * this.data ** (n - 1) * out.grad;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: number | Value): Value {
    return this.add(Value.of(other).neg());
  }

  div(other: number | Value): Value {
    return this.mul(Value.of(other).pow(-1));
  }

  exp(): Value {
    const out = new Value(Math.exp(this.data), [this]);
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this]);
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this]);
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  /** Backpropagate: seed d(this)/d(this) = 1 and apply the chain rule in
   *  reverse topological order. */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS — all gradient checks green.

- [ ] **Step 5: Commit**

```bash
git add src/lib/nn/value.ts src/lib/nn/value.test.ts
git commit -m "feat: scalar autograd Value with numerically-verified gradients"
```

---

### Task 3: Neuron/Layer/MLP + loss + SGD (TDD, trains XOR)

**Files:**
- Create: `src/lib/nn/network.ts`
- Test: `src/lib/nn/network.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/nn/network.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { mulberry32 } from './rng';
import { MLP, mseLoss, trainStep, zeroGrad } from './network';
import { Value } from './value';

describe('MLP structure', () => {
  it('has the right parameter count', () => {
    // [2,4,1]: layer1 = 4*(2+1) = 12, layer2 = 1*(4+1) = 5 → 17
    const mlp = new MLP([2, 4, 1], mulberry32(1));
    expect(mlp.parameters()).toHaveLength(17);
  });

  it('forward returns one Value per output and predict returns numbers', () => {
    const mlp = new MLP([2, 3, 2], mulberry32(1));
    const out = mlp.forward([0.5, -0.5]);
    expect(out).toHaveLength(2);
    expect(out[0]).toBeInstanceOf(Value);
    const nums = mlp.predict([0.5, -0.5]);
    expect(nums).toHaveLength(2);
    expect(typeof nums[0]).toBe('number');
  });

  it('tanh outputs are bounded in (-1, 1)', () => {
    const mlp = new MLP([2, 4, 1], mulberry32(3));
    for (const p of [[1, 1], [-1, 1], [0.3, -0.9]]) {
      const [y] = mlp.predict(p);
      expect(Math.abs(y)).toBeLessThan(1);
    }
  });

  it('is deterministic for a given seed', () => {
    const a = new MLP([2, 4, 1], mulberry32(42)).predict([0.2, 0.8]);
    const b = new MLP([2, 4, 1], mulberry32(42)).predict([0.2, 0.8]);
    expect(a).toEqual(b);
  });
});

describe('loss and training', () => {
  it('mseLoss is zero for perfect predictions and positive otherwise', () => {
    expect(mseLoss([new Value(1), new Value(-1)], [1, -1]).data).toBe(0);
    expect(mseLoss([new Value(0.5)], [1]).data).toBeCloseTo(0.25);
  });

  it('zeroGrad clears gradients', () => {
    const mlp = new MLP([2, 2, 1], mulberry32(1));
    const loss = mseLoss(mlp.forward([1, -1]), [1]);
    loss.backward();
    const params = mlp.parameters();
    expect(params.some((p) => p.grad !== 0)).toBe(true);
    zeroGrad(params);
    expect(params.every((p) => p.grad === 0)).toBe(true);
  });

  it('a trainStep reduces loss on a simple batch', () => {
    const mlp = new MLP([2, 4, 1], mulberry32(7));
    const inputs = [[0.5, 0.5], [-0.5, -0.5]];
    const targets = [[1], [-1]];
    const first = trainStep(mlp, inputs, targets, 0.1);
    let last = first;
    for (let i = 0; i < 20; i++) last = trainStep(mlp, inputs, targets, 0.1);
    expect(last).toBeLessThan(first);
  });

  it('learns XOR — the integration test for the whole engine', () => {
    // If the seed proves unlucky for convergence, try small integer seeds and
    // pin the first that converges — determinism is the point, not seed 42.
    const mlp = new MLP([2, 8, 1], mulberry32(42));
    const inputs = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    const targets = [[-1], [1], [1], [-1]];
    let loss = Infinity;
    for (let epoch = 0; epoch < 800; epoch++) {
      loss = trainStep(mlp, inputs, targets, 0.3);
    }
    expect(loss).toBeLessThan(0.05);
    for (let i = 0; i < inputs.length; i++) {
      const [y] = mlp.predict(inputs[i]);
      expect(Math.sign(y)).toBe(Math.sign(targets[i][0]));
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./network`.

- [ ] **Step 3: Write `src/lib/nn/network.ts`**

```ts
import { Value } from './value';
import type { Rng } from './rng';

export type Activation = 'tanh' | 'relu' | 'linear';

export class Neuron {
  readonly w: Value[];
  readonly b: Value;
  readonly activation: Activation;

  constructor(nin: number, activation: Activation, rng: Rng) {
    this.w = Array.from({ length: nin }, () => new Value(rng() * 2 - 1));
    this.b = new Value(0);
    this.activation = activation;
  }

  forward(x: Value[]): Value {
    let act: Value = this.b;
    for (let i = 0; i < this.w.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    if (this.activation === 'tanh') return act.tanh();
    if (this.activation === 'relu') return act.relu();
    return act;
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  readonly neurons: Neuron[];

  constructor(nin: number, nout: number, activation: Activation, rng: Rng) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, activation, rng));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

export class MLP {
  readonly layers: Layer[];

  /** sizes e.g. [2, 8, 1]: 2 inputs, one hidden layer of 8, 1 output.
   *  Hidden layers use `hidden` activation; the last layer uses `output`. */
  constructor(
    sizes: number[],
    rng: Rng,
    hidden: Activation = 'tanh',
    output: Activation = 'tanh'
  ) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const isLast = i === sizes.length - 2;
      this.layers.push(new Layer(sizes[i], sizes[i + 1], isLast ? output : hidden, rng));
    }
  }

  forward(xs: number[]): Value[] {
    let v = xs.map((x) => new Value(x));
    for (const layer of this.layers) v = layer.forward(v);
    return v;
  }

  predict(xs: number[]): number[] {
    return this.forward(xs).map((v) => v.data);
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

export function mseLoss(preds: Value[], targets: number[]): Value {
  let sum = new Value(0);
  for (let i = 0; i < preds.length; i++) {
    sum = sum.add(preds[i].sub(targets[i]).pow(2));
  }
  return sum.div(preds.length);
}

export function zeroGrad(params: Value[]): void {
  for (const p of params) p.grad = 0;
}

export function sgdStep(params: Value[], lr: number): void {
  for (const p of params) p.data -= lr * p.grad;
}

/** One full-batch gradient step. Returns the batch loss before the update. */
export function trainStep(
  mlp: MLP,
  inputs: number[][],
  targets: number[][],
  lr: number
): number {
  const params = mlp.parameters();
  zeroGrad(params);
  let total = new Value(0);
  for (let i = 0; i < inputs.length; i++) {
    total = total.add(mseLoss(mlp.forward(inputs[i]), targets[i]));
  }
  const loss = total.div(inputs.length);
  loss.backward();
  sgdStep(params, lr);
  return loss.data;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS including the XOR convergence test. If XOR fails to converge with seed 42, change only the seed in the test (try 1, 2, 3…) and note it in the commit message — do not weaken the assertions.

- [ ] **Step 5: Commit**

```bash
git add src/lib/nn/network.ts src/lib/nn/network.test.ts
git commit -m "feat: MLP with SGD training loop; learns XOR deterministically"
```

---

### Task 4: Toy datasets (TDD)

Only the two datasets Chapter 1 needs. Circles/moons/spiral arrive in Stage 2b with the Training Playground.

**Files:**
- Create: `src/lib/nn/datasets.ts`
- Test: `src/lib/nn/datasets.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/nn/datasets.test.ts`**

```ts
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
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./datasets`.

- [ ] **Step 3: Write `src/lib/nn/datasets.ts`**

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS. (If the separability or four-quadrant assertions fail for the pinned seeds, adjust only the seed constants in the tests to seeds that satisfy them — the properties, not the seeds, are the contract.)

- [ ] **Step 5: Commit**

```bash
git add src/lib/nn/datasets.ts src/lib/nn/datasets.test.ts
git commit -m "feat: deterministic toy datasets (two clouds, XOR quadrants)"
```

---

### Task 5: Viz scale helper (TDD)

**Files:**
- Create: `src/lib/viz/scale.ts`
- Test: `src/lib/viz/scale.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/viz/scale.test.ts`**

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./scale`.

- [ ] **Step 3: Write `src/lib/viz/scale.ts`**

```ts
/** Linear domain→range mapping with inversion, for SVG/canvas coordinates. */
export interface Scale {
  (value: number): number;
  invert(px: number): number;
}

export function linearScale(
  [d0, d1]: [number, number],
  [r0, r1]: [number, number]
): Scale {
  const scale = ((value: number) =>
    r0 + ((value - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  scale.invert = (px: number) => d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
  return scale;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/viz/scale.ts src/lib/viz/scale.test.ts
git commit -m "feat: linearScale viz helper with inversion"
```

---

### Task 6: FigureShell + PerceptronPlayground + full Chapter 1

The first real figure, establishing the pattern every later figure follows: a Svelte 5 island, server-rendered initial state as the no-JS fallback, hydrated `client:visible`, wrapped in the shared `FigureShell`.

**Files:**
- Create: `src/components/figures/FigureShell.svelte`
- Create: `src/components/figures/perceptron/PerceptronPlayground.svelte`
- Modify: `src/content/chapters/the-neuron.mdx` (full rewrite below)

- [ ] **Step 1: Write `src/components/figures/FigureShell.svelte`**

```svelte
<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    caption?: string;
    children: Snippet;
    controls?: Snippet;
  }

  let { title, caption = '', children, controls }: Props = $props();
</script>

<figure class="fig sketch">
  <figcaption class="fig-title label">{title}</figcaption>
  <div class="fig-body">{@render children()}</div>
  {#if controls}
    <div class="fig-controls">{@render controls()}</div>
  {/if}
  {#if caption}
    <p class="fig-caption">{caption}</p>
  {/if}
</figure>

<style>
  .fig {
    margin: 2rem 0;
    padding: 1.25rem;
  }
  .fig-title {
    margin-bottom: 0.75rem;
  }
  .fig-body :global(svg) {
    width: 100%;
    height: auto;
    display: block;
  }
  .fig-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem 1.25rem;
    margin-top: 1rem;
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--muted);
  }
  .fig-controls :global(label) {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .fig-controls :global(input[type='range']) {
    accent-color: var(--accent-blue);
    width: 110px;
  }
  .fig-controls :global(button) {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    padding: 0.3rem 0.8rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .fig-controls :global(button:hover) {
    border-color: var(--ink-strong);
  }
  .fig-caption {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0.75rem 0 0;
    text-align: center;
  }
</style>
```

- [ ] **Step 2: Write `src/components/figures/perceptron/PerceptronPlayground.svelte`**

Behavior contract: two datasets (clouds = separable, xor = impossible); three sliders (w₁, w₂, b); live decision line; misclassified points ringed; accuracy readout; hover a point to see its weighted sum; XOR mode shows the "wall" message once the user is near the best achievable accuracy; reset button.

```svelte
<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { twoClouds, xorQuadrants, type Point2 } from '../../../lib/nn/datasets';
  import { linearScale } from '../../../lib/viz/scale';

  const DEFAULTS = { w1: 0.4, w2: -0.6, b: 0.1 };

  let w1 = $state(DEFAULTS.w1);
  let w2 = $state(DEFAULTS.w2);
  let b = $state(DEFAULTS.b);
  let mode: 'clouds' | 'xor' = $state('clouds');

  const clouds = twoClouds(15, 1);
  const xor = xorQuadrants(40); // default seed — pinned balanced by datasets tests
  const points = $derived(mode === 'clouds' ? clouds : xor);

  const sx = linearScale([-1.2, 1.2], [12, 328]);
  const sy = linearScale([-1.2, 1.2], [328, 12]);

  const predict = (p: Point2) => (w1 * p.x + w2 * p.y + b > 0 ? 1 : -1);
  const sum = (p: Point2) => w1 * p.x + w2 * p.y + b;

  const correct = $derived(points.filter((p) => predict(p) === p.label).length);
  const accuracy = $derived(correct / points.length);

  // Decision line w1*x + w2*y + b = 0, clipped to the domain box.
  const line = $derived.by(() => {
    if (Math.abs(w1) < 1e-9 && Math.abs(w2) < 1e-9) return null;
    if (Math.abs(w2) >= Math.abs(w1)) {
      const yAt = (x: number) => -(w1 * x + b) / w2;
      return { x1: -1.2, y1: yAt(-1.2), x2: 1.2, y2: yAt(1.2) };
    }
    const xAt = (y: number) => -(w2 * y + b) / w1;
    return { x1: xAt(-1.2), y1: -1.2, x2: xAt(1.2), y2: 1.2 };
  });

  // XOR's best achievable single-line accuracy hovers around 50–75%.
  const showWall = $derived(mode === 'xor' && accuracy >= 0.6);

  function reset() {
    w1 = DEFAULTS.w1;
    w2 = DEFAULTS.w2;
    b = DEFAULTS.b;
  }

  function setMode(m: 'clouds' | 'xor') {
    mode = m;
    reset();
  }

  const fmt = (v: number) => v.toFixed(2);
</script>

<FigureShell
  title="Build a perceptron"
  caption="Drag the sliders to move the decision line. Score: how many points end up on their correct side."
>
  {#snippet children()}
    <svg viewBox="0 0 340 340" aria-label="Scatter plot of labeled points with an adjustable decision line">
      <line x1={sx(-1.2)} y1={sy(0)} x2={sx(1.2)} y2={sy(0)} stroke="var(--line-soft)" stroke-width="1" />
      <line x1={sx(0)} y1={sy(-1.2)} x2={sx(0)} y2={sy(1.2)} stroke="var(--line-soft)" stroke-width="1" />

      {#if line}
        <line
          x1={sx(line.x1)}
          y1={sy(line.y1)}
          x2={sx(line.x2)}
          y2={sy(line.y2)}
          stroke="var(--ink-strong)"
          stroke-width="2.5"
          stroke-linecap="round"
        />
      {/if}

      {#each points as p}
        {@const wrong = predict(p) !== p.label}
        <g>
          <title>
            {`w₁·x + w₂·y + b = ${fmt(sum(p))} → predicts ${sum(p) > 0 ? '+1' : '−1'}, actually ${p.label > 0 ? '+1' : '−1'}`}
          </title>
          <circle
            cx={sx(p.x)}
            cy={sy(p.y)}
            r="6"
            fill={p.label === 1 ? 'var(--accent-blue)' : 'var(--accent-red)'}
            opacity="0.85"
          />
          {#if wrong}
            <circle
              cx={sx(p.x)}
              cy={sy(p.y)}
              r="10"
              fill="none"
              stroke="var(--accent-gold)"
              stroke-width="2"
              stroke-dasharray="3 2"
            />
          {/if}
        </g>
      {/each}
    </svg>

    <p class="score" class:perfect={correct === points.length}>
      {correct} / {points.length} correct
      {#if correct === points.length}— separated!{/if}
    </p>

    {#if showWall}
      <p class="wall">
        Stuck? That's the point. No single line can split all four quadrants —
        this is the famous <strong>XOR wall</strong> that stalled neural
        networks for a decade. Breaking through it takes bendier functions
        (next chapter) and stacked layers (chapter 6).
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>w₁ <input type="range" min="-2" max="2" step="0.05" bind:value={w1} /> {fmt(w1)}</label>
    <label>w₂ <input type="range" min="-2" max="2" step="0.05" bind:value={w2} /> {fmt(w2)}</label>
    <label>b <input type="range" min="-1.5" max="1.5" step="0.05" bind:value={b} /> {fmt(b)}</label>
    <button type="button" onclick={reset}>Reset</button>
    <button
      type="button"
      class:active={mode === 'xor'}
      onclick={() => setMode(mode === 'xor' ? 'clouds' : 'xor')}
    >
      {mode === 'xor' ? '← Back to clouds' : 'Try XOR'}
    </button>
  {/snippet}
</FigureShell>

<style>
  .score {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.5rem 0 0;
  }
  .score.perfect {
    color: var(--accent-green);
    font-weight: 700;
  }
  .wall {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    background: var(--paper);
    border-left: 3px solid var(--accent-red);
    padding: 0.6rem 0.9rem;
    margin: 0.75rem 0 0;
    color: var(--ink);
  }
  button.active {
    border-color: var(--accent-red);
    color: var(--accent-red);
  }
</style>
```

- [ ] **Step 3: Rewrite `src/content/chapters/the-neuron.mdx` in full**

````mdx
---
number: 1
title: The Neuron
summary: From if statements to weighted sums — the smallest unit of learning.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import PerceptronPlayground from '../../components/figures/perceptron/PerceptronPlayground.svelte';

You already write neurons. Every time you code a threshold —

```ts
if (0.6 * temperature + 0.4 * humidity > 75) turnOnFan();
```

— you've built one: a weighted sum of inputs compared against a threshold.
That is the perceptron, the 1958 original, and it is the atom this entire
site builds from.

Look at the shape of that `if` statement. Each input gets a **weight** — a
number saying how much it matters. Temperature matters more than humidity
here (0.6 vs 0.4), so it pulls harder on the decision. Sum up the pulls,
compare against a threshold, act. That's the whole machine.

Rearrange it slightly and the threshold becomes a **bias** term `b` on the
left-hand side, and the comparison is always against zero:

```ts
if (w1 * x1 + w2 * x2 + b > 0) fire();
```

Why rearrange? Because now the decision has a *geometry*. The equation
`w1·x1 + w2·x2 + b = 0` is a line. Everything on one side fires, everything
on the other side doesn't. Choosing weights **is** choosing where that line
sits — and you can try it yourself:

<PerceptronPlayground client:visible />

With the two clouds, a few slider nudges find a line that separates them
perfectly. Congratulations — you just did, by hand, what "learning" will
soon do automatically: adjust weights until the decisions come out right.

Now click **Try XOR**.

The pattern is simple — same-sign coordinates are one class, opposite-sign
the other — but no line you can draw gets them all. You can prove it to
yourself in about thirty seconds of frustrated slider-dragging. This is the
**XOR wall**, and it is not a toy problem: in 1969 Minsky and Papert used
exactly this limitation to argue perceptrons were a dead end, and neural
network research froze for a decade.

The escape route has two parts, and they are the next two ideas in this
course: activation functions that *bend* (so decisions aren't stuck being
straight lines), and layers that *stack* (so simple decisions compose into
complicated ones). A single neuron can only draw a line. A network of them
can draw anything.

<MathAside>

A neuron computes $y = \sigma(w \cdot x + b)$, where $w$ is the weight
vector, $b$ is the bias, and $\sigma$ is an activation function. The
perceptron's $\sigma$ is a hard step: output $1$ if $w \cdot x + b > 0$,
else $-1$. The decision boundary $w \cdot x + b = 0$ is a hyperplane —
a line in 2D, a plane in 3D — and $w$ is its normal vector: it points
perpendicular to the boundary, toward the positive class.

</MathAside>

<EngineerFooter>

A neuron is `dot(w, x) + b` piped through a nonlinearity — a fused
multiply-accumulate with a squashing function. A layer is a matrix-vector
product; a network is a pipeline of them. That's the whole trick. The rest
of this site is about choosing `w` automatically (chapters 3–5) and wiring
billions of these together (chapters 6 onward). The figure above is powered
by the same ~150-line autograd engine you'll meet in chapter 4 — view
source, it's all there.

</EngineerFooter>
````

- [ ] **Step 4: Verify build, tests, hydration**

Run: `npm test && npm run check && npm run build`
Expected: all tests pass; 0 check errors; build `Complete!`.

Then verify the island actually hydrates and falls back:
```bash
grep -c "astro-island" dist/chapters/the-neuron/index.html   # ≥ 1 — island registered
grep -c "Build a perceptron" dist/chapters/the-neuron/index.html  # ≥ 1 — server-rendered fallback markup present
ls dist/_astro/ | grep -i -c "svelte"                        # ≥ 1 — svelte runtime emitted
```

- [ ] **Step 5: Verify links still good**

Run: `npx linkinator ./dist --recurse --silent --skip "^https?://(?!localhost)"`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add src/components/figures/ src/content/chapters/the-neuron.mdx
git commit -m "feat: perceptron playground figure and full chapter 1 content"
```

---

## Out of scope for Stage 2a

- Chapters 2–5, the Training Playground, circles/moons/spiral datasets, decision-boundary heatmap rendering, ScrollStage scrollytelling — all Stage 2b.
- Any change to the deploy pipeline (unchanged from Stage 1).

## Addendum: review-driven deviations (recorded post-execution)

Canonical for later stages:

1. **RNG sequence is a public contract.** `rng.test.ts` pins golden values for `mulberry32(42)`. Changing the algorithm — or the *number of draws* an init scheme consumes (e.g. switching weight init from uniform to gaussian) — breaks downstream seed-pinned tests (XOR convergence, dataset shapes) and requires re-pinning seeds deliberately.
2. **XOR test seed is 1** (42 doesn't converge with [2,8,1]/lr 0.3/800 epochs); `xorQuadrants` default seed is **3** (seed 2 is visually lopsided: 18 points in one quadrant). A quadrant-balance test guards the default.
3. **`backward()` docs the accumulation contract** — callers must `zeroGrad` between independent passes.
4. **Chapter 1 prose**: the EngineerFooter must not claim the perceptron figure runs on the autograd engine (it computes sums inline); the honest forward-reference wording is in the committed MDX.
5. **Figure defaults open as a near-miss, not broken**: playground DEFAULTS {w1:0.5, w2:0.5, b:-0.7} = 60% on clouds, verified empirically. New figures should open 50–70% "solvable", never below chance.

### House rules for all Stage 2b–5 figures (decided once, here)

- **SVG data-point tooltips** (`<title>`) are hover-only progressive enhancement — acceptable ONLY when the information is redundantly available (color + rings + text readout). If a tooltip carries non-redundant data, the element must be focusable with an accessible name.
- **Class/category encoding**: color + at least one redundant channel (position, ring, or text readout). Prefer adding a shape distinction when points overlap heavily.
- **Range inputs**: wrap in `<label>` (implicit association). If the visible value readout creates a noisy accessible name, prefer static `aria-label` + `aria-valuetext`.
- **Islands**: `client:visible`, SSR initial state as the no-JS fallback, wrapped in `FigureShell` (title/caption/children/controls snippets). Deterministic seeds only.
- **Stage 2b perf note**: `trainStep` allocates the whole graph per step (~1.3ms for [2,8,1]×100 — fine). If slider-scrub jank appears, cache `parameters()` and consider batch predict; don't optimize preemptively.
