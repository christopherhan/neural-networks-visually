# Stage 3: Part II Chapters 6–8 Implementation Plan

> **For agentic workers:** Execution mode for this stage (per user direction): implementer subagent per task, coordinator verifies inline between tasks — no separate reviewer subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Part II — chapter 6 (Going Deep: MLPs, with a real-engine gradient-flow figure showing vanishing gradients), chapter 7 (Convolutional Networks, with a step-through kernel playground), chapter 8 (RNNs & LSTMs, with a recurrence memory-decay figure).

**Architecture:** Same figure pattern as Stages 2a/2b (Svelte 5 islands, `client:visible`, `FigureShell`, deterministic, SSR fallback, house rules from the 2a addendum). Chapter 6's figure runs the real `MLP`/`Value` engine per interaction to measure per-layer gradient magnitudes. Chapters 7–8 use closed-form local computation (convolution loop, recurrence loop) — no engine needed.

**Tech Stack:** Existing only — TypeScript, Svelte 5 runes, Astro/MDX, Vitest. No new dependencies, no new lib modules (all figure logic is component-local and deterministic).

**Frontmatter contract (build fails on mismatch):**

| file | number | title |
|---|---|---|
| going-deep-mlps.mdx | 6 | Going Deep: MLPs |
| convolutional-networks.mdx | 7 | Convolutional Networks |
| rnns-and-lstms.mdx | 8 | RNNs & LSTMs |

**File map:**

```
src/components/figures/depth/GradientFlowFigure.svelte      (Task 1)
src/content/chapters/going-deep-mlps.mdx                    (Task 1)
src/components/figures/conv/KernelPlayground.svelte         (Task 2)
src/content/chapters/convolutional-networks.mdx             (Task 2)
src/components/figures/rnn/MemoryDecayFigure.svelte         (Task 3)
src/content/chapters/rnns-and-lstms.mdx                     (Task 3)
```

---

### Task 1: Chapter 6 — Gradient Flow figure + MDX

**Files:**
- Create: `src/components/figures/depth/GradientFlowFigure.svelte`
- Create: `src/content/chapters/going-deep-mlps.mdx`

- [ ] **Step 1: Write `src/components/figures/depth/GradientFlowFigure.svelte`:**

```svelte
<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { MLP, mseLoss } from '../../../lib/nn/network';
  import type { Activation } from '../../../lib/nn/network';
  import { mulberry32 } from '../../../lib/nn/rng';

  // Build a narrow-but-deep net, push one example through, backprop once,
  // and measure the mean |gradient| of each layer's weights. The real engine
  // does the measuring — nothing here is illustrative fakery.
  let depth = $state(5); // number of hidden layers, 1..8
  let activation: Activation = $state('tanh');
  let seedOffset = $state(0);

  const WIDTH = 4;
  const FLOOR = 1e-6;

  function measure(d: number, act: Activation, seed: number): number[] {
    const sizes = [1, ...Array(d).fill(WIDTH), 1];
    const mlp = new MLP(sizes, mulberry32(seed), act, 'tanh');
    const loss = mseLoss(mlp.forward([0.7]), [1]);
    loss.backward();
    return mlp.layers.map((layer) => {
      const params = layer.parameters();
      const mean =
        params.reduce((s, p) => s + Math.abs(p.grad), 0) / params.length;
      return Math.max(mean, FLOOR);
    });
  }

  const grads = $derived(measure(depth, activation, 11 + seedOffset));
  const ratio = $derived(grads[grads.length - 1] / grads[0]);

  // Log-scale bars: FLOOR (1e-6) → 0px tall, 1 → full height.
  const H = 150;
  const barH = (g: number) => {
    const t = (Math.log10(g) + 6) / 6; // 1e-6..1 → 0..1
    return Math.max(2, Math.min(1, Math.max(0, t)) * H);
  };
  const barW = $derived(Math.min(44, 280 / grads.length));

  const fmtExp = (v: number) => v.toExponential(1).replace('e-', 'e−');
</script>

<FigureShell
  title="Where gradients go to die"
  caption="Mean |gradient| of each layer's weights after one backward pass, log scale. Layer 1 is the input side — the farthest from the loss."
>
  {#snippet children()}
    <svg viewBox="0 0 340 200" aria-label="Bar chart of gradient magnitude per layer; earlier layers receive smaller gradients with tanh">
      {#each grads as g, i}
        {@const x = 30 + i * (barW + 8)}
        <rect
          {x}
          y={170 - barH(g)}
          width={barW}
          height={barH(g)}
          rx="3"
          fill={i === 0 ? 'var(--accent-red)' : 'var(--accent-blue)'}
          opacity="0.85"
        />
        <text x={x + barW / 2} y={185} text-anchor="middle" class="axis">
          {i + 1}
        </text>
      {/each}
      <text x="30" y="14" class="axis">|grad| (log scale)</text>
    </svg>

    <p class="readout">
      output layer {fmtExp(grads[grads.length - 1])} &nbsp;→&nbsp; layer 1
      <strong>{fmtExp(grads[0])}</strong>
      {#if ratio > 30}
        <span class="shrunk">
          — the far layer's signal is <strong>{Math.round(ratio).toLocaleString()}×</strong> weaker
        </span>
      {/if}
    </p>

    {#if activation === 'tanh' && depth >= 6 && ratio > 100}
      <p class="note">
        Every tanh the blame passes through multiplies it by a slope ≤ 1 —
        stack enough of them and the early layers barely hear about their
        mistakes. Now click <strong>ReLU</strong> and watch the same
        architecture breathe again.
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>hidden layers <input type="range" min="1" max="8" step="1" bind:value={depth} /> {depth}</label>
    <button
      type="button"
      class:active={activation === 'tanh'}
      aria-pressed={activation === 'tanh'}
      onclick={() => (activation = 'tanh')}
    >
      Tanh
    </button>
    <button
      type="button"
      class:active={activation === 'relu'}
      aria-pressed={activation === 'relu'}
      onclick={() => (activation = 'relu')}
    >
      ReLU
    </button>
    <button type="button" onclick={() => (seedOffset = (seedOffset + 1) % 7)}>
      Re-roll weights
    </button>
  {/snippet}
</FigureShell>

<style>
  .axis {
    font-family: var(--font-ui);
    font-size: 10px;
    fill: var(--faint);
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.5rem 0 0;
  }
  .shrunk {
    color: var(--accent-gold-ink);
  }
  .note {
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

- [ ] **Step 2: Write `src/content/chapters/going-deep-mlps.mdx`:**

````mdx
---
number: 6
title: "Going Deep: MLPs"
summary: Stacking layers buys expressive power - and quietly starves the layers that need feedback most.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import GradientFlowFigure from '../../components/figures/depth/GradientFlowFigure.svelte';

Welcome to Part II. Part I built the machine; this part is about *shaping*
it. The Training Playground already whispered the first lesson: the spiral
needed the deep network. Time to say it out loud — and then meet the tax
that depth collects.

Why does depth help at all? A one-hidden-layer network can, in principle,
fit anything — but it does so the dumb way, by brute-force memorizing bumps.
Stacked layers compose instead: layer one learns edges of the problem,
layer two combines edges into motifs, layer three combines motifs into the
answer. Reusable parts, combined hierarchically — the same reason you write
functions instead of one enormous main(). Depth is *composition*; width is
*vocabulary*. Most problems want both, but composition is where the magic
lives.

So: stack twenty layers and profit? Not so fast. Remember how blame travels
(chapter 4): backward through every operation, *multiplied* by each local
slope along the way. A tanh's slope is at most 1 — and usually much less.
Blame that crosses eight tanh layers gets multiplied by eight small numbers.
Watch what that does:

<GradientFlowFigure client:visible />

Drag the depth up with tanh selected. The bars are per-layer gradient
strength — the learning signal each layer receives — on a log scale. By
layer eight the early layers are getting a signal thousands of times weaker
than the output layer. They're effectively frozen: the layers closest to
the *input*, the ones that must learn the most basic features, hear the
faintest whisper of what went wrong. This is the **vanishing gradient
problem**, the great villain of 1990s neural networks — the same wall the
tanh damping hinted at in chapter 4.

Now click ReLU. Its slope is exactly 1 wherever it's active — blame passes
through undamped. The bars level out (mostly — dead ReLUs claim their own
small tax). One activation swap, and depth becomes affordable. This single
trick, plus better initialization and (later) skip connections, is a large
part of why "deep" learning became possible at all.

Hit "Re-roll weights" a few times — the pattern survives every draw. It's
not bad luck; it's arithmetic.

<MathAside>

The gradient reaching layer $k$ of an $n$-layer net is a product of
Jacobians: $\frac{\partial L}{\partial h_k} = \frac{\partial L}{\partial
h_n}\prod_{j=k}^{n-1} W_{j+1}^\top \,\mathrm{diag}(\sigma'(z_j))$. With
$|\sigma'| \le 1$ (tanh) the product shrinks geometrically in depth;
with ReLU, $\sigma' \in \{0, 1\}$, so active paths pass gradients
unchanged. Exploding gradients are the mirror case, when the weight terms
dominate and the product grows.

</MathAside>

Convolutional networks — next chapter — take composition seriously in a
second way: instead of just stacking layers, they *shape* each layer to
match the structure of the data itself.

<EngineerFooter>

A deep net's backward pass is a chain of matrix products; vanishing/
exploding gradients are just the condition number of that chain. The
modern fixes are engineering, not magic: ReLU-family activations
(slope 1), careful init (Xavier/He scale weights so products stay near 1),
normalization layers, and residual connections — which add an identity
"bypass lane" so gradients can skip the multiplications entirely. You'll
meet that last idea again inside every transformer block in Part III.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build` green; `grep -c "astro-island" dist/chapters/going-deep-mlps/index.html` ≥ 1; `grep -c "going-deep-mlps" dist/chapters/training-playground/index.html` ≥ 1 (ch5's next-link chains).

- [ ] **Step 4: Commit**

```bash
git add src/components/figures/depth/ src/content/chapters/going-deep-mlps.mdx
git commit -m "feat: chapter 6 — going deep, with real-engine gradient-flow figure"
```

---

### Task 2: Chapter 7 — Kernel Playground + MDX

**Files:**
- Create: `src/components/figures/conv/KernelPlayground.svelte`
- Create: `src/content/chapters/convolutional-networks.mdx`

- [ ] **Step 1: Write `src/components/figures/conv/KernelPlayground.svelte`:**

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { createTicker } from '../../../lib/viz/ticker';

  // A 14x14 grayscale "window frame" image: strong vertical and horizontal
  // strokes so edge kernels have something honest to find.
  const N = 14;
  const IMG: number[][] = Array.from({ length: N }, () => Array(N).fill(0.05));
  for (let i = 3; i <= 10; i++) {
    IMG[3][i] = 1;
    IMG[10][i] = 1;
    IMG[i][3] = 1;
    IMG[i][10] = 1;
  }
  IMG[6][6] = IMG[6][7] = IMG[7][6] = IMG[7][7] = 0.8; // center pane

  type KernelKey = 'vertical' | 'horizontal' | 'sharpen' | 'blur';
  const KERNELS: Record<KernelKey, { label: string; k: number[][] }> = {
    vertical: { label: 'Vertical edges', k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
    horizontal: { label: 'Horizontal edges', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
    sharpen: { label: 'Sharpen', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    blur: { label: 'Blur', k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
  };

  const OUT = N - 2; // valid convolution: 12x12

  let kernelKey: KernelKey = $state('vertical');
  let revealed = $state(0); // number of output cells computed so far, 0..144
  let playing = $state(false);

  function convolveAt(row: number, col: number, k: number[][]): number {
    let sum = 0;
    for (let dr = 0; dr < 3; dr++) {
      for (let dc = 0; dc < 3; dc++) {
        sum += k[dr][dc] * IMG[row + dr][col + dc];
      }
    }
    return sum;
  }

  const fullOutput = $derived.by(() => {
    const k = KERNELS[kernelKey].k;
    const out: number[] = [];
    for (let r = 0; r < OUT; r++) {
      for (let c = 0; c < OUT; c++) out.push(convolveAt(r, c, k));
    }
    return out;
  });
  const maxAbs = $derived(Math.max(...fullOutput.map(Math.abs), 1e-9));

  const curRow = $derived(Math.min(Math.floor(revealed / OUT), OUT - 1));
  const curCol = $derived(revealed % OUT);
  const done = $derived(revealed >= OUT * OUT);

  function advance() {
    if (revealed < OUT * OUT) revealed += 1;
    if (revealed >= OUT * OUT) stopPlay();
  }
  const ticker = createTicker(advance, { fps: 30 });
  onDestroy(() => ticker.stop());

  function stopPlay() {
    playing = false;
    ticker.stop();
  }
  function togglePlay() {
    if (done) revealed = 0;
    playing = !playing;
    if (playing) ticker.start();
    else ticker.stop();
  }
  function finish() {
    stopPlay();
    revealed = OUT * OUT;
  }
  function reset() {
    stopPlay();
    revealed = 0;
  }
  function pick(k: KernelKey) {
    stopPlay();
    kernelKey = k;
    revealed = 0;
  }

  const CELL = 15;
  const gray = (v: number) => {
    const g = Math.round(235 - Math.max(0, Math.min(1, v)) * 190);
    return `rgb(${g},${g - 4},${g - 14})`;
  };
  const diverging = (v: number) => {
    const t = Math.max(-1, Math.min(1, v / maxAbs));
    return t >= 0
      ? `rgba(38, 139, 210, ${0.12 + 0.88 * t})`
      : `rgba(220, 50, 47, ${0.12 - 0.88 * t})`;
  };
  const fmt = (v: number) => v.toFixed(2);
</script>

<FigureShell
  title="Slide a kernel, light a feature map"
  caption="Left: the input image. Right: the feature map — one cell per kernel position. Blue = positive response, red = negative."
>
  {#snippet children()}
    <div class="panes">
      <svg
        viewBox="0 0 {N * CELL} {N * CELL}"
        class="pane"
        aria-label="14 by 14 input image with the kernel's current 3 by 3 position outlined"
      >
        {#each IMG as rowVals, r}
          {#each rowVals as v, c}
            <rect x={c * CELL} y={r * CELL} width={CELL - 1} height={CELL - 1} fill={gray(v)} />
          {/each}
        {/each}
        {#if !done && revealed > 0}
          <rect
            x={curCol * CELL - 1}
            y={curRow * CELL - 1}
            width={CELL * 3 + 1}
            height={CELL * 3 + 1}
            fill="none"
            stroke="var(--accent-gold)"
            stroke-width="2.5"
          />
        {/if}
      </svg>

      <svg
        viewBox="0 0 {OUT * CELL} {OUT * CELL}"
        class="pane"
        aria-label="12 by 12 feature map filling in as the kernel slides"
      >
        {#each fullOutput as v, i}
          {#if i < revealed}
            <rect
              x={(i % OUT) * CELL}
              y={Math.floor(i / OUT) * CELL}
              width={CELL - 1}
              height={CELL - 1}
              fill={diverging(v)}
            />
          {/if}
        {/each}
      </svg>
    </div>

    <p class="readout">
      {#if revealed === 0}
        Press Play — the kernel visits all {OUT * OUT} positions.
      {:else if !done}
        position ({curRow + 1}, {curCol + 1}): kernel ⊙ patch =
        <strong>{fmt(fullOutput[revealed - 1])}</strong>
      {:else}
        Done — {OUT * OUT} weighted sums, all sharing the same 9 weights.
      {/if}
    </p>
  {/snippet}

  {#snippet controls()}
    {#each Object.entries(KERNELS) as [key, def]}
      <button
        type="button"
        class:active={kernelKey === key}
        aria-pressed={kernelKey === key}
        onclick={() => pick(key as KernelKey)}
      >
        {def.label}
      </button>
    {/each}
    <button type="button" onclick={advance} disabled={done}>Step</button>
    <button type="button" onclick={togglePlay}>{playing ? 'Pause' : 'Play'}</button>
    <button type="button" onclick={finish} disabled={done}>Finish</button>
    <button type="button" onclick={reset}>Reset</button>
  {/snippet}
</FigureShell>

<style>
  .panes {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
  }
  .pane {
    width: min(240px, 44%);
    height: auto;
    background: var(--paper);
    border-radius: 8px;
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.75rem 0 0;
    min-height: 1.4em;
  }
  button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
</style>
```

- [ ] **Step 2: Write `src/content/chapters/convolutional-networks.mdx`:**

````mdx
---
number: 7
title: Convolutional Networks
summary: Stop wiring every pixel to every neuron - slide one tiny neuron everywhere instead.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import KernelPlayground from '../../components/figures/conv/KernelPlayground.svelte';

Feed a photo to the networks we've built so far and you hit an absurdity.
A megapixel image has a million inputs; wiring each to even a modest hidden
layer costs *billions* of weights — nearly all of which would relearn the
same lesson. Because here's what a dense layer doesn't know: an edge in the
top-left corner is the same thing as an edge in the bottom-right. Images
have structure — nearby pixels matter to each other, and patterns repeat
everywhere. A layer that ignores that structure pays for its ignorance in
parameters.

The convolutional insight: build one *tiny* neuron — a 3×3 grid of weights
called a **kernel** — and *slide it* across the whole image, computing its
little weighted sum at every position. Same nine weights everywhere. The
grid of responses it leaves behind is a **feature map**: a picture of
*where the pattern lives*. Watch it happen:

<KernelPlayground client:visible />

Play the vertical-edge kernel across the window frame. Look at its weights:
negative column, zeros, positive column — it computes "right side minus
left side," so it fires exactly where brightness changes left-to-right: the
frame's vertical strokes. Switch to horizontal edges and the response flips
to the other strokes. Same image, different kernel, different question
asked at every location. (Sharpen and blur are the same machinery wearing
photo-editing clothes — your phone's filters are convolutions.)

Now the numbers. That kernel asked its question at 144 positions using
**9 weights total**. A dense layer answering the same 144 questions would
have needed 144 separate neurons × 196 inputs ≈ 28,000 weights. Weight
sharing is a ~3000× discount — and it comes with a bonus insight baked in:
the pattern detector works *everywhere by construction*, so the network
doesn't have to see a cat in every corner of the frame to recognize
corner cats.

A real convolutional network stacks this: dozens of kernels per layer (each
learning its own pattern — and yes, *learning*: kernels are trained by the
same backprop as everything else, not hand-designed like our presets), with
each layer's feature maps becoming the next layer's input. Layer one finds
edges; layer two combines edges into corners and textures; deeper layers
respond to eyes, wheels, faces. It's chapter 6's composition story, but
with geometry built in.

<MathAside>

A convolution computes $(I * K)_{r,c} = \sum_{i,j} K_{i,j}\, I_{r+i,\,c+j}$
— a dot product between the kernel and each patch. Our figure uses "valid"
padding ($14{-}2 = 12$ outputs per side) and stride 1. Parameter count for
a conv layer: $k^2 \cdot C_{in} \cdot C_{out}$, independent of image size —
that independence is the whole economic argument.

</MathAside>

One more architecture question remains for Part II. Images repeat in
*space* — but what about data that unfolds in *time*, where the past has to
inform the present? That needs a network with memory.

<EngineerFooter>

Convolution is a matmul with tied weights and a sliding access pattern —
which makes it embarrassingly parallel and cache-friendly, i.e. exactly
what GPUs eat. Frameworks lower it to im2col + GEMM or use implicit
kernels (cuDNN). The parameters-vs-activations asymmetry is the operational
gotcha: conv layers are cheap to *store* but expensive to *compute*, the
reverse of dense layers. Pooling/stride trades resolution for receptive
field — by the deep layers, each cell "sees" most of the image.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build` green; `grep -c "astro-island" dist/chapters/convolutional-networks/index.html` ≥ 1.

- [ ] **Step 4: Commit**

```bash
git add src/components/figures/conv/ src/content/chapters/convolutional-networks.mdx
git commit -m "feat: chapter 7 — convolutional networks with kernel playground"
```

---

### Task 3: Chapter 8 — Memory Decay figure + MDX

**Files:**
- Create: `src/components/figures/rnn/MemoryDecayFigure.svelte`
- Create: `src/content/chapters/rnns-and-lstms.mdx`

- [ ] **Step 1: Write `src/components/figures/rnn/MemoryDecayFigure.svelte`:**

```svelte
<script lang="ts">
  import FigureShell from '../FigureShell.svelte';

  // A single recurrent unit: h[t+1] = tanh(w * h[t] + x[t]).
  // Click input cells to place spikes; drag w to control how much of
  // yesterday survives into today.
  const T = 20;

  let w = $state(0.65);
  let spikes: boolean[] = $state(
    Array.from({ length: T }, (_, i) => i === 2)
  );

  const hs = $derived.by(() => {
    const out: number[] = [];
    let h = 0;
    for (let t = 0; t < T; t++) {
      h = Math.tanh(w * h + (spikes[t] ? 1 : 0));
      out.push(h);
    }
    return out;
  });

  function toggle(i: number) {
    spikes = spikes.map((s, j) => (j === i ? !s : s));
  }

  function preset(kind: 'early' | 'refresh' | 'clear') {
    if (kind === 'early') {
      spikes = Array.from({ length: T }, (_, i) => i === 2);
    } else if (kind === 'refresh') {
      spikes = Array.from({ length: T }, (_, i) => i % 6 === 2);
    } else {
      spikes = Array(T).fill(false);
    }
  }

  const regime = $derived(
    w < 0.6 ? 'fades' : w <= 1.0 ? 'lingers' : 'latches'
  );

  const CELL = 15;
  const H = 120;
  const fmt = (v: number) => v.toFixed(2);
</script>

<FigureShell
  title="A neuron that remembers (briefly)"
  caption="Bottom row: the input over 20 time steps — click cells to add or remove spikes. Bars: the hidden state h, the network's running memory."
>
  {#snippet children()}
    <svg viewBox="0 0 {T * CELL + 40} {H + 20}" aria-label="Hidden state magnitude over 20 time steps, responding to input spikes">
      <line x1="20" y1={H} x2={T * CELL + 20} y2={H} stroke="var(--line-soft)" />
      {#each hs as h, t}
        <rect
          x={22 + t * CELL}
          y={H - Math.abs(h) * (H - 10)}
          width={CELL - 4}
          height={Math.max(2, Math.abs(h) * (H - 10))}
          rx="2"
          fill="var(--accent-blue)"
          opacity={0.35 + 0.65 * Math.abs(h)}
        />
      {/each}
    </svg>

    <div class="input-row" role="group" aria-label="Input spikes per time step">
      {#each spikes as s, i}
        <button
          type="button"
          class="spike"
          class:on={s}
          aria-pressed={s}
          aria-label={`input at step ${i + 1}`}
          onclick={() => toggle(i)}
        >
          {s ? '▲' : '·'}
        </button>
      {/each}
    </div>

    <p class="readout">
      w = {fmt(w)} · memory <strong>{regime}</strong>
      {#if regime === 'fades'}
        — each step keeps only {Math.round(w * 100)}% of the last; the spike is gone in a few ticks
      {:else if regime === 'lingers'}
        — echoes survive several steps, but old news still loses to new input
      {:else}
        — above w ≈ 1 the loop feeds itself: the memory self-sustains and won't let go
      {/if}
    </p>
  {/snippet}

  {#snippet controls()}
    <label>recurrence w <input type="range" min="0.1" max="1.5" step="0.05" bind:value={w} /> {fmt(w)}</label>
    <button type="button" onclick={() => preset('early')}>One early spike</button>
    <button type="button" onclick={() => preset('refresh')}>Refreshed spikes</button>
    <button type="button" onclick={() => preset('clear')}>Clear inputs</button>
  {/snippet}
</FigureShell>

<style>
  .input-row {
    display: flex;
    gap: 1px;
    justify-content: center;
    margin-top: 0.25rem;
  }
  .spike {
    width: 14px;
    height: 20px;
    padding: 0;
    font-size: 10px;
    line-height: 1;
    background: var(--paper);
    border: 1px solid var(--line-soft);
    border-radius: 3px;
    color: var(--faint);
    cursor: pointer;
  }
  .spike.on {
    color: var(--accent-red);
    border-color: var(--accent-red);
    background: var(--paper-raised);
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.75rem 0 0;
  }
</style>
```

- [ ] **Step 2: Write `src/content/chapters/rnns-and-lstms.mdx`:**

````mdx
---
number: 8
title: RNNs & LSTMs
summary: Give the network a loop and it can remember - until the loop itself becomes the problem.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';
import MemoryDecayFigure from '../../components/figures/rnn/MemoryDecayFigure.svelte';

Everything so far looks at its input all at once. But language, audio,
sensor logs, keystrokes — sequences — arrive one piece at a time, and the
meaning of *now* depends on *before*. "The trophy didn't fit in the
suitcase because **it** was too big" — resolving that "it" requires
remembering two nouns from earlier in the stream.

The **recurrent** answer is beautifully economical: give the network a
loop. One set of weights processes each time step, and alongside the input
it receives its *own previous output* — a hidden state `h`, the network's
running summary of everything so far. Memory in one extra arrow. Play with
the smallest possible version:

<MemoryDecayFigure client:visible />

That single slider `w` decides the fate of a memory. Drop a spike at step 2
and watch: at w = 0.65, each tick keeps only 65% of the last — the echo
dies within five steps, drowned long before step 20. Slide toward 1.0 and
the echo stretches. Push past 1.0 and something strange happens: the loop
feeds itself faster than tanh can squash it, and the memory *latches* —
self-sustaining, refusing to fade.

Now connect this to training. Unroll the loop across 20 steps and you get —
squint — a 20-layer network in which **every layer shares the same
weight**. Chapter 6's vanishing gradient returns, concentrated: blame
flowing back from step 20 to step 2 crosses eighteen copies of the same
multiplication. Below 1, it vanishes exponentially; above 1, it explodes.
The knife-edge in the figure *is* the training problem: plain RNNs can't
hold memories across long gaps because gradients can't survive the trip
back to deposit them.

The 1997 fix is one of the great engineering moves in the field. The
**LSTM** (Long Short-Term Memory) stops hoping the loop behaves and takes
*control* of it: alongside `h` it keeps a protected cell state — a conveyor
belt — governed by learned **gates**. A *forget gate* decides what fraction
of the old memory survives (a learned, per-moment version of our `w`
slider). An *input gate* decides what new information gets written. An
*output gate* decides what's revealed. The gates are tiny neurons
themselves — so the network *learns when to remember and when to forget*,
instead of decaying at a fixed rate. On the conveyor belt, gradients travel
back through addition, not repeated multiplication — the long-gap memory
problem, largely solved.

For fifteen years, LSTMs ran the show: translation, speech recognition,
your phone's autocomplete. But note the shape of what they do: all of
history, squeezed through one fixed-size hidden state, step by step, in
order. What if — Part III asks — instead of *carrying* a compressed memory
forward, the network could simply *look back* at any part of the sequence,
directly, whenever it needs to?

<MathAside>

Unrolled recurrence: $h_t = \tanh(W_h h_{t-1} + W_x x_t)$, so
$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} W_h^\top\,
\mathrm{diag}(\tanh'(z_t))$ — the same Jacobian product as chapter 6, but
with the *same* $W_h$ repeated, which is why the spectral radius of $W_h$
so sharply separates vanish from explode. The LSTM cell update $c_t = f_t
\odot c_{t-1} + i_t \odot \tilde{c}_t$ is additive, giving gradients a
multiplication-free path backward.

</MathAside>

<EngineerFooter>

An RNN is a fold over a sequence with learned state; an LSTM is that fold
with an engineered highway for the carry. The operational pain is that the
fold is *sequential* — step t needs step t−1, so you can't parallelize
across time on a GPU the way you can across an image. That serialization,
as much as the memory problem, is what Part III's attention architecture
kills: it trades the O(1)-memory loop for parallel random access over the
whole sequence. Part II complete — the classical era ends here.

</EngineerFooter>
````

- [ ] **Step 3: Verify** — `npm test && npm run check && npm run build` green; `grep -c "astro-island" dist/chapters/rnns-and-lstms/index.html` ≥ 1; `grep -c "Next: coming soon" dist/chapters/rnns-and-lstms/index.html` ≥ 1 (ch9 unwritten).

- [ ] **Step 4: Commit**

```bash
git add src/components/figures/rnn/ src/content/chapters/rnns-and-lstms.mdx
git commit -m "feat: chapter 8 — RNNs and LSTMs with memory-decay figure"
```

---

### Task 4: Integration verification (coordinator, inline)

- [ ] **Step 1:** `npm test && npm run check && npm run build` → 74 tests, 0 errors, Complete.
- [ ] **Step 2:** `npx linkinator ./dist --recurse --silent --skip "^https?://(?!localhost)"` → exit 0; trail map now has **8** unique chapter links; ch5→ch6, ch6→ch7, ch7→ch8 next-links present.
- [ ] **Step 3:** Base-path build + guard grep (same commands as Stage 2b Task 7) → OK.
- [ ] **Step 4:** Merge to main, push, watch CI, curl the three new chapter URLs → 200.

---

## Out of scope for Stage 3

- Part III anything (tokenizers, attention, the Python trainer).
- Pooling/stride interactives, multi-channel conv, actual LSTM gate simulation (prose + static description suffice at this depth per spec).
