<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { twoClouds, xorQuadrants, type Point2 } from '../../../lib/nn/datasets';
  import { linearScale } from '../../../lib/viz/scale';

  const DEFAULTS = { w1: 0.5, w2: 0.5, b: -0.7 };

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

  // The wall message should reward a genuine attempt, not greet the reader on
  // arrival: defaults score 63% on XOR, and the best single line reaches ~78%,
  // so trigger only once the user has pushed close to that ceiling.
  const showWall = $derived(mode === 'xor' && accuracy >= 0.7);

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
  title="Build a neuron"
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
