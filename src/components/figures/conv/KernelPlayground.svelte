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
