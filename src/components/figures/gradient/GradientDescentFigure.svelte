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
