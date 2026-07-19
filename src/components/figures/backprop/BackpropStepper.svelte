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
