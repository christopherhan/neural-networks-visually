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
