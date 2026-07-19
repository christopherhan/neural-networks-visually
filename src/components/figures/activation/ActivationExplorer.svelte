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
  const fmt = (v: number) => v.toFixed(2);

  const fn = $derived(FNS[fnKey]);

  function sample(g: (x: number) => number): string {
    const pts: string[] = [];
    for (let i = 0; i <= 120; i++) {
      const xi = -3 + (6 * i) / 120;
      pts.push(`${sx(xi)},${sy(g(xi))}`);
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
      <line x1={sx(x)} y1={sy(-1.5)} x2={sx(x)} y2={sy(fn.f(x))} stroke="var(--faint)" stroke-dasharray="3 3" />
      <circle cx={sx(x)} cy={sy(fn.f(x))} r="6" fill="var(--accent-red)" />
    </svg>

    <p class="readout">
      f({fmt(x)}) = <strong>{fmt(fn.f(x))}</strong>
      {#if fn.f(x) > 1.5} <span class="off">(off the chart ↑)</span>{/if}
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
        aria-pressed={fnKey === key}
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
  .readout .off {
    color: var(--faint);
    font-style: italic;
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
