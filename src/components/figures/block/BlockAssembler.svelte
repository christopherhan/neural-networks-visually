<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { createTicker } from '../../../lib/viz/ticker';

  // No model here — this figure is a labeled diagram of the transformer
  // block's data flow, revealed one operation at a time. Reuses the
  // step/play/reset interaction pattern from BackpropStepper exactly.

  interface StepDef {
    id: string;
    label: string;
    explain: string;
  }

  const STEPS: StepDef[] = [
    {
      id: 'tokens',
      label: 'tokens',
      explain:
        'A sequence of token vectors arrives — one column of numbers per token (chapter 9).',
    },
    {
      id: 'positions',
      label: '+ positions',
      explain:
        'Position fingerprints are added in (chapter 12 tells that story) — same tensor, now order-aware.',
    },
    {
      id: 'ln1',
      label: 'LayerNorm',
      explain:
        "Normalize each vector: mean 0, spread 1. Keeps eight of these blocks stacked without chapter 6's product problems.",
    },
    {
      id: 'qkv',
      label: 'Q·K·V',
      explain:
        'Three learned projections of the SAME input: what I seek (Q), what I advertise (K), what I offer (V).',
    },
    {
      id: 'mha',
      label: 'multi-head attention',
      explain:
        'Four attention heads run in parallel on 16-dim slices — four independent consultation habits, concatenated back together.',
    },
    {
      id: 'res1',
      label: '+ residual',
      explain:
        "Add the block's input straight back in. The bypass lane from chapter 6 — gradients and information can skip the whole operation.",
    },
    {
      id: 'ln2',
      label: 'LayerNorm',
      explain: 'Normalize again before the second half.',
    },
    {
      id: 'mlp',
      label: 'MLP',
      explain:
        "A little two-layer network (chapter 5's species!) applied to each token independently — where per-token computation happens between the mixing steps.",
    },
    {
      id: 'res2',
      label: '+ residual',
      explain:
        'Second bypass lane. Attention mixed across tokens; the MLP thought per token; the input rode along untouched.',
    },
    {
      id: 'output',
      label: 'output',
      explain:
        'Out comes the same shape that came in — which is the whole trick: stack this block N times. GPT is this diagram, repeated.',
    },
  ];
  const TOTAL = STEPS.length;

  let stepIndex = $state(0);
  let playing = $state(false);

  const current = $derived(stepIndex === 0 ? null : STEPS[stepIndex - 1]);

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

  // ---- Diagram geometry (static; rows map 1:1 onto STEPS by index) ----
  const CX = 330;
  const MAIN_W = 220;
  const LEFT = CX - MAIN_W / 2; // 220
  const RIGHT = CX + MAIN_W / 2; // 440

  interface Row {
    y: number;
    h: number;
  }
  const ROWS: Row[] = [
    { y: 16, h: 26 }, // 0 tokens
    { y: 48, h: 26 }, // 1 + positions
    { y: 80, h: 26 }, // 2 LayerNorm
    { y: 112, h: 26 }, // 3 Q·K·V
    { y: 144, h: 26 }, // 4 multi-head attention
    { y: 176, h: 26 }, // 5 + residual
    { y: 208, h: 26 }, // 6 LayerNorm
    { y: 240, h: 26 }, // 7 MLP
    { y: 272, h: 26 }, // 8 + residual
    { y: 304, h: 26 }, // 9 output
  ];
  const VIEW_H = 350;

  const topY = (i: number) => ROWS[i].y;
  const botY = (i: number) => ROWS[i].y + ROWS[i].h;
  const midY = (i: number) => ROWS[i].y + ROWS[i].h / 2;

  // Q/K/V sub-boxes sit in row 3, splitting the main column into three.
  const QKV_W = 64;
  const QKV_GAP = 14;
  const qkvBoxes = [0, 1, 2].map((k) => ({
    x: LEFT + k * (QKV_W + QKV_GAP),
    label: ['Q', 'K', 'V'][k],
  }));

  // Simple one-in-one-out vertical connectors (row i -> row i+1).
  const SIMPLE_LINKS = [0, 1, 4, 5, 6, 7, 8];

  // Curved bypass paths, looped out to the right of the main column.
  // Bypass 1: output of "+ positions" (row 1) skips LN1/QKV/MHA, rejoins at
  // "+ residual" (row 5). Bypass 2: output of that same "+ residual" (row 5)
  // skips LN2/MLP, rejoins at the second "+ residual" (row 8) — nested one
  // loop further out so the two paths stay visually distinct.
  const bypass1 = `M ${RIGHT} ${midY(1)} C ${RIGHT + 100} ${midY(1)}, ${RIGHT + 100} ${midY(5)}, ${RIGHT} ${midY(5)}`;
  const bypass2 = `M ${RIGHT} ${midY(5)} C ${RIGHT + 140} ${midY(5)}, ${RIGHT + 140} ${midY(8)}, ${RIGHT} ${midY(8)}`;

  const isRevealed = (rowIndex: number) => stepIndex >= rowIndex + 1;
  const isCurrent = (rowIndex: number) => stepIndex === rowIndex + 1;
</script>

<FigureShell
  title="One transformer block, assembled"
  caption="Every block in every GPT-style model is this same shape: attention mixes across tokens, the MLP thinks per token, residuals and LayerNorm are the plumbing that lets many of these stack."
>
  {#snippet children()}
    <svg
      viewBox={`0 0 700 ${VIEW_H}`}
      aria-label="Flow diagram of one transformer block, assembled step by step from token embeddings through attention and MLP sublayers to the block's output"
    >
      <defs>
        <marker
          id="block-arrow-ink"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="var(--line)" />
        </marker>
        <marker
          id="block-arrow-green"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent-green)" />
        </marker>
      </defs>

      <!-- main vertical flow: simple one-to-one links -->
      {#each SIMPLE_LINKS as i}
        <line
          x1={CX}
          y1={botY(i)}
          x2={CX}
          y2={topY(i + 1)}
          stroke="var(--line)"
          stroke-width="1.5"
          marker-end="url(#block-arrow-ink)"
        />
      {/each}

      <!-- LayerNorm (row 2) fans out into Q, K, V (row 3) -->
      {#each qkvBoxes as qb}
        <line
          x1={CX}
          y1={botY(2)}
          x2={qb.x + QKV_W / 2}
          y2={topY(3)}
          stroke="var(--line)"
          stroke-width="1.5"
          marker-end="url(#block-arrow-ink)"
        />
      {/each}

      <!-- Q, K, V (row 3) converge into multi-head attention (row 4) -->
      {#each qkvBoxes as qb}
        <line
          x1={qb.x + QKV_W / 2}
          y1={botY(3)}
          x2={CX}
          y2={topY(4)}
          stroke="var(--line)"
          stroke-width="1.5"
          marker-end="url(#block-arrow-ink)"
        />
      {/each}

      <!-- residual bypass loops -->
      <path
        d={bypass1}
        fill="none"
        stroke="var(--accent-green)"
        stroke-width="2"
        stroke-dasharray="6 4"
        marker-end="url(#block-arrow-green)"
      />
      <path
        d={bypass2}
        fill="none"
        stroke="var(--accent-green)"
        stroke-width="2"
        stroke-dasharray="6 4"
        marker-end="url(#block-arrow-green)"
      />

      <!-- boxes -->
      {#each ROWS as row, i}
        {#if i === 3}
          {#each qkvBoxes as qb}
            <g>
              <rect
                x={qb.x}
                y={row.y}
                width={QKV_W}
                height={row.h}
                rx="6"
                fill={isRevealed(i) ? 'var(--paper-raised)' : 'none'}
                stroke={isCurrent(i) ? 'var(--accent-red)' : 'var(--line)'}
                stroke-width={isCurrent(i) ? 3 : 1.5}
                stroke-dasharray={isRevealed(i) ? 'none' : '4 3'}
              />
              {#if isRevealed(i)}
                <text x={qb.x + QKV_W / 2} y={row.y + row.h / 2 + 5} text-anchor="middle" class="qkv-label">
                  {qb.label}
                </text>
              {/if}
            </g>
          {/each}
        {:else}
          {@const label = STEPS[i].label}
          <g>
            <rect
              x={LEFT}
              y={row.y}
              width={MAIN_W}
              height={row.h}
              rx="8"
              fill={isRevealed(i) ? 'var(--paper-raised)' : 'none'}
              stroke={isCurrent(i) ? 'var(--accent-red)' : 'var(--line)'}
              stroke-width={isCurrent(i) ? 3 : 1.5}
              stroke-dasharray={isRevealed(i) ? 'none' : '4 3'}
            />
            {#if isRevealed(i)}
              <text x={CX} y={row.y + row.h / 2 + 5} text-anchor="middle" class="box-label">{label}</text>
            {/if}
          </g>
        {/if}
      {/each}
    </svg>

    <p class="explain">
      {#if current}
        <span class="phase label">{current.label}</span> {current.explain}
      {:else}
        <span class="phase label">ready</span> Press Step (or Play) to assemble the block, one layer at
        a time.
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
  .box-label {
    font-family: var(--font-ui);
    font-size: 11px;
    font-weight: 700;
    fill: var(--ink-strong);
  }
  .qkv-label {
    font-family: var(--font-ui);
    font-size: 14px;
    font-weight: 700;
    fill: var(--ink-strong);
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
  .phase {
    margin-right: 0.5rem;
  }
  .progress {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--faint);
  }
</style>
