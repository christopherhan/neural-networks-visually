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
