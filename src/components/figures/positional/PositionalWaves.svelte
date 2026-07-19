<script lang="ts">
  import FigureShell from '../FigureShell.svelte';

  // Classic sinusoidal positional encoding (Vaswani et al. 2017), computed
  // closed-form — no model, no training. d_model = 64, positions 0..63,
  // matching the site's toy transformer's context length.
  const D = 64;
  const POS = 64;

  function computePE(): number[][] {
    const pe: number[][] = Array.from({ length: POS }, () => Array(D).fill(0));
    for (let pos = 0; pos < POS; pos++) {
      for (let i = 0; i < D / 2; i++) {
        const freq = 1 / Math.pow(10000, (2 * i) / D);
        pe[pos][2 * i] = Math.sin(pos * freq);
        pe[pos][2 * i + 1] = Math.cos(pos * freq);
      }
    }
    return pe;
  }

  const PE = computePE();

  function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  function norm(a: number[]): number {
    return Math.sqrt(dot(a, a));
  }
  function cosineSim(a: number[], b: number[]): number {
    const denom = norm(a) * norm(b);
    return denom < 1e-9 ? 0 : dot(a, b) / denom;
  }

  let posA = $state(8);
  let posB = $state(12);

  const rowA = $derived(PE[posA]);
  const rowB = $derived(PE[posB]);
  const similarity = $derived(cosineSim(rowA, rowB));

  const CELL = 5;
  const GRID = D * CELL; // 320

  // Values are already bounded to [-1, 1] by construction (sin/cos), so the
  // color scale needs no data-dependent max — unlike KernelPlayground's
  // convolution outputs, this magnitude ceiling is exact.
  const diverging = (v: number) => {
    const t = Math.max(-1, Math.min(1, v));
    return t >= 0
      ? `rgba(38, 139, 210, ${0.12 + 0.88 * t})`
      : `rgba(220, 50, 47, ${0.12 - 0.88 * t})`;
  };

  const CURVE_H = 90;
  const curvePoints = (row: number[]) =>
    row
      .map((v, i) => {
        const x = (i / (D - 1)) * GRID;
        const y = CURVE_H / 2 - v * (CURVE_H / 2 - 4);
        return `${x},${y}`;
      })
      .join(' ');

  const fmt2 = (v: number) => v.toFixed(2);
</script>

<FigureShell
  title="A wave-shaped ruler for position"
  caption="64 positions x 64 dimensions. Blue = positive, red = negative. Low dimensions oscillate fast (fine-grained position); high dimensions oscillate slow (coarse position) — a continuous analog of binary counting."
>
  {#snippet children()}
    <svg
      viewBox="0 0 {GRID} {GRID}"
      class="heatmap"
      aria-label="64 by 64 heatmap of the sinusoidal positional encoding; blue cells are positive, red cells are negative, and the two selected positions are outlined"
    >
      {#each PE as row, pos}
        {#each row as v, dim}
          <rect x={dim * CELL} y={pos * CELL} width={CELL} height={CELL} fill={diverging(v)} />
        {/each}
      {/each}
      <rect
        x="0"
        y={posA * CELL}
        width={GRID}
        height={CELL}
        fill="none"
        stroke="var(--accent-blue)"
        stroke-width="1.5"
      />
      <rect
        x="0"
        y={posB * CELL}
        width={GRID}
        height={CELL}
        fill="none"
        stroke="var(--accent-gold)"
        stroke-width="1.5"
      />
    </svg>

    <svg
      viewBox="0 0 {GRID} {CURVE_H}"
      class="curves"
      aria-label="Line curves of position A and position B's encoding values across all 64 dimensions"
    >
      <line x1="0" y1={CURVE_H / 2} x2={GRID} y2={CURVE_H / 2} stroke="var(--line-soft)" stroke-width="1" />
      <polyline points={curvePoints(rowA)} fill="none" stroke="var(--accent-blue)" stroke-width="1.75" />
      <polyline points={curvePoints(rowB)} fill="none" stroke="var(--accent-gold)" stroke-width="1.75" />
    </svg>

    <p class="readout">
      cosine similarity between position <strong>{posA}</strong> (blue) and position
      <strong>{posB}</strong> (gold): <strong>{fmt2(similarity)}</strong> — nearby
      positions look alike; distant ones don't.
    </p>
  {/snippet}

  {#snippet controls()}
    <label>position A <input type="range" min="0" max="63" step="1" bind:value={posA} /> {posA}</label>
    <label>position B <input type="range" min="0" max="63" step="1" bind:value={posB} /> {posB}</label>
  {/snippet}
</FigureShell>

<style>
  .heatmap {
    display: block;
    width: 100%;
    max-width: 420px;
    height: auto;
    margin: 0 auto;
    background: var(--paper);
    border-radius: 8px;
  }
  .curves {
    display: block;
    width: 100%;
    max-width: 420px;
    height: auto;
    margin: 0.75rem auto 0;
    background: var(--paper);
    border-radius: 8px;
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.75rem 0 0;
  }
</style>
