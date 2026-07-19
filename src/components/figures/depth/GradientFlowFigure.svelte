<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { MLP, mseLoss } from '../../../lib/nn/network';
  import { mulberry32 } from '../../../lib/nn/rng';

  // Deep nets multiply blame through every layer on the way back. With the
  // classic textbook advice — "initialize with small random weights" — every
  // factor in that product is < 1, so the signal reaching early layers
  // shrinks exponentially with depth. Fan-in-scaled (Xavier) init sizes the
  // factors to sit near 1 instead. Every number here comes from the real
  // engine: build a net, backprop once, measure per-layer mean |grad|.
  let depth = $state(6); // hidden layers, 1..8
  let scaled = $state(false);
  let seedIndex = $state(0);

  const WIDTH = 8;
  const SEEDS = [11, 12, 13, 14, 15, 16, 17];
  const FLOOR = 1e-9;
  // Engine init is U(-1,1) (std ≈ 0.577). Multiplying by these factors gives:
  // classic small-random init (std ≈ 0.14) vs Xavier for fan-in 8 (std ≈ 0.35).
  // IMPLEMENTER NOTE: verify both factors empirically per the task's
  // verification step before committing; committed constants = verified ones.
  const FACTOR_SMALL = 0.25;
  const FACTOR_XAVIER = 0.61;

  function measure(d: number, factor: number, seed: number): number[] {
    // Linear output keeps the loss gradient healthy at the top, so the bars
    // show purely how the hidden stack treats blame on the way down.
    const mlp = new MLP([1, ...Array(d).fill(WIDTH), 1], mulberry32(seed), 'tanh', 'linear');
    for (const p of mlp.parameters()) p.data *= factor;
    const loss = mseLoss(mlp.forward([0.7]), [1]);
    loss.backward();
    return mlp.layers.map((layer) => {
      const params = layer.parameters();
      const mean = params.reduce((s, p) => s + Math.abs(p.grad), 0) / params.length;
      return Math.max(mean, FLOOR);
    });
  }

  const grads = $derived(
    measure(depth, scaled ? FACTOR_XAVIER : FACTOR_SMALL, SEEDS[seedIndex])
  );
  const ratio = $derived(grads[grads.length - 1] / grads[0]);

  // Log-scale bars spanning 1e-9 .. 1e1.
  const H = 150;
  const barH = (g: number) => {
    const t = (Math.log10(g) + 9) / 10;
    return Math.max(2, Math.min(1, Math.max(0, t)) * H);
  };
  const barW = $derived(Math.min(40, 280 / grads.length));

  const fmtRatio = (r: number) =>
    r >= 1000 ? `${Math.round(r / 100) / 10}k` : `${Math.round(r)}`;
</script>

<FigureShell
  title="The deep-layer whisper"
  caption="Mean |gradient| of each layer's weights after one backward pass, log scale. Layer 1 is the input side — farthest from the loss. Re-roll draws fresh random weights."
>
  {#snippet children()}
    <svg viewBox="0 0 340 200" aria-label="Bar chart of gradient magnitude per layer; small init starves early layers, scaled init keeps them fed">
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
      {#if ratio > 3}
        layer 1 hears the loss at
        <strong>1/{fmtRatio(ratio)}</strong> the volume of the output layer
      {:else}
        every layer hears the loss at roughly the same volume — balanced
      {/if}
    </p>

    {#if !scaled && depth >= 6 && ratio > 100}
      <p class="note">
        Each layer multiplies the passing blame by weights smaller than 1 and
        a tanh slope of at most 1. {depth} hidden layers deep, the product has
        collapsed — the layers that must learn the most basic features are
        effectively frozen. Now flip to <strong>fan-in scaled</strong> and
        re-roll: same architecture, same randomness, factors sized to 1.
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>hidden layers <input type="range" min="1" max="8" step="1" bind:value={depth} /> {depth}</label>
    <button
      type="button"
      class:active={!scaled}
      aria-pressed={!scaled}
      onclick={() => (scaled = false)}
    >
      Small random (classic)
    </button>
    <button
      type="button"
      class:active={scaled}
      aria-pressed={scaled}
      onclick={() => (scaled = true)}
    >
      Fan-in scaled (Xavier)
    </button>
    <button type="button" onclick={() => (seedIndex = (seedIndex + 1) % SEEDS.length)}>
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
