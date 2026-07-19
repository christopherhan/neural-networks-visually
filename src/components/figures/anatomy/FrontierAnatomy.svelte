<script lang="ts">
  import FigureShell from '../FigureShell.svelte';

  // No model here — this figure is a labeled architecture diagram, revealed
  // through real buttons beneath a decorative stack (the house rule this
  // site follows everywhere, e.g. TrainingPipeline). Every number below is
  // either a published open-research figure or an explicitly-labeled
  // published-family approximation — see the standing disclosure line
  // rendered under the title, and the honesty paragraph in the chapter
  // prose. Nothing here claims to show any lab's actual production
  // internals.

  interface Region {
    id: string;
    label: string;
    fact: string;
  }

  const REGIONS: Region[] = [
    {
      id: 'embedding',
      label: 'Embedding',
      fact: 'Byte-level BPE tokenizer, vocabulary ≈160,000 — the same lookup-table idea chapter 9 introduced at three-novel scale, re-tuned for a training corpus spanning the open web. Every token still becomes one vector in one big table; only the size of the table changed on the way to a frontier model.',
    },
    {
      id: 'attention',
      label: 'MLA Attention',
      fact: "Multi-head latent attention: the identical softmax(QKᵀ/√d)V from chapter 10, but instead of caching every head's full-size key and value, DeepSeek's published design compresses each token's key and value down to one shared ≈512-dimension latent per layer and reconstructs full size on demand — the KV-cache economics chapter 16 priced out, applied at the architecture level.",
    },
    {
      id: 'moe',
      label: 'MoE FFN',
      fact: "Each block's one MLP is replaced by 1 always-on shared expert plus 256 routed experts, with a router sending every token to its top 8 (chapter 15's mixture-of-experts, at frontier scale). Published DeepSeek-V3/Kimi-K2-family totals land around ≈1 trillion parameters total, ≈32 billion active per token — almost the entire weight count, almost entirely idle on any given token, by design.",
    },
    {
      id: 'lmhead',
      label: 'LM Head',
      fact: "A dense projection back to vocabulary size, identical in kind to chapter 13's next-token head, just aimed at a ≈160k-entry vocabulary. Context window: ≈128k tokens, affordable to serve because of chapter 16's KV-cache economics. Training adds a multi-token-prediction (MTP) objective on top of the ordinary next-token loss from chapter 14's pipeline — predict a couple of tokens ahead, not just one, which published results tie to a stronger training signal and, at inference, a path to speculative decoding.",
    },
  ];

  let selected = $state(0); // default-select Embedding, top of the stack

  const current = $derived(REGIONS[selected]);

  function select(i: number) {
    selected = i;
  }

  function isSel(id: string): boolean {
    return REGIONS[selected].id === id;
  }

  // ---- Params bar: total vs active, the ~30x sparsity win ----
  const TOTAL_PARAMS = 1_000_000_000_000; // ≈1T, published-family approximation
  const ACTIVE_PARAMS = 32_000_000_000; // ≈32B, published-family approximation
  const SPARSITY_RATIO = TOTAL_PARAMS / ACTIVE_PARAMS; // ≈31×

  function formatParams(n: number): string {
    if (n >= 1e12) return `${(n / 1e12).toFixed(n % 1e12 === 0 ? 0 : 1)}T`;
    if (n >= 1e9) return `${(n / 1e9).toFixed(n % 1e9 === 0 ? 0 : 1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    return `${n}`;
  }

  const activeBarPct = (ACTIVE_PARAMS / TOTAL_PARAMS) * 100;

  // ---- Diagram geometry (static labeled stack) ----
  const VIEW_W = 700;
  const VIEW_H = 480;
  const CX = 350;

  const EMBED_Y = 16;
  const EMBED_H = 46;
  const EMBED_W = 240;
  const EMBED_X = CX - EMBED_W / 2;

  const FRAME_Y = 90;
  const FRAME_H = 300;
  const FRAME_W = 340;
  const FRAME_X = CX - FRAME_W / 2;

  const ATTN_Y = FRAME_Y + 34;
  const ATTN_H = 54;
  const ATTN_W = 260;
  const ATTN_X = CX - ATTN_W / 2;

  const MOE_Y = ATTN_Y + ATTN_H + 22;
  const MOE_H = 170;
  const MOE_W = 320;
  const MOE_X = CX - MOE_W / 2;

  const LMHEAD_Y = FRAME_Y + FRAME_H + 28;
  const LMHEAD_H = 46;
  const LMHEAD_W = 240;
  const LMHEAD_X = CX - LMHEAD_W / 2;

  // Expert grid: 32 cols x 8 rows = 256 routed experts, decorative. A fixed
  // scattered subset of 8 is drawn in the "active" color to depict top-8
  // routing — this is illustrative of the mechanism (chapter 15), not a
  // record of which experts a real token would hit.
  const GRID_COLS = 32;
  const GRID_ROWS = 8;
  const GRID_X = MOE_X + 64;
  const GRID_Y = MOE_Y + 34;
  const CELL_DX = (MOE_W - 64 - 14) / GRID_COLS;
  const CELL_DY = 88 / GRID_ROWS;
  const CELL_W = CELL_DX - 1.5;
  const CELL_H = CELL_DY - 2;
  const ACTIVE_INDICES = new Set([3, 19, 40, 77, 102, 150, 201, 248]);

  const expertCells = Array.from({ length: GRID_COLS * GRID_ROWS }, (_, i) => ({
    i,
    x: GRID_X + (i % GRID_COLS) * CELL_DX,
    y: GRID_Y + Math.floor(i / GRID_COLS) * CELL_DY,
    active: ACTIVE_INDICES.has(i),
  }));
</script>

<FigureShell
  title="Anatomy of a frontier model"
  caption="Select a region for its published-family numbers and the chapter that explains it."
>
  {#snippet children()}
    <p class="disclosure">
      Representative of the published DeepSeek-V3 / Kimi-K2 family recipe —
      frontier labs' exact production internals (K3, Fable-class) are not public.
    </p>

    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <marker
          id="anatomy-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="var(--line)" />
        </marker>
      </defs>

      <!-- connecting arrows -->
      <line
        x1={CX}
        y1={EMBED_Y + EMBED_H}
        x2={CX}
        y2={FRAME_Y}
        stroke="var(--line)"
        stroke-width="1.5"
        marker-end="url(#anatomy-arrow)"
      />
      <line
        x1={CX}
        y1={FRAME_Y + FRAME_H}
        x2={CX}
        y2={LMHEAD_Y}
        stroke="var(--line)"
        stroke-width="1.5"
        marker-end="url(#anatomy-arrow)"
      />
      <line
        x1={CX}
        y1={ATTN_Y + ATTN_H}
        x2={CX}
        y2={MOE_Y}
        stroke="var(--line)"
        stroke-width="1.5"
        marker-end="url(#anatomy-arrow)"
      />

      <!-- Embedding -->
      <g>
        <rect
          x={EMBED_X}
          y={EMBED_Y}
          width={EMBED_W}
          height={EMBED_H}
          rx="10"
          fill={isSel('embedding') ? 'var(--paper-raised)' : 'var(--paper)'}
          stroke={isSel('embedding') ? 'var(--accent-red)' : 'var(--line)'}
          stroke-width={isSel('embedding') ? 3 : 1.5}
        />
        <text x={CX} y={EMBED_Y + 27} text-anchor="middle" class="box-label">Embedding</text>
        <text x={CX} y={EMBED_Y + 40} text-anchor="middle" class="box-sublabel">
          byte-level BPE · ≈160k vocab
        </text>
      </g>

      <!-- Block frame, repeated 61x -->
      <rect
        x={FRAME_X}
        y={FRAME_Y}
        width={FRAME_W}
        height={FRAME_H}
        rx="14"
        fill="none"
        stroke="var(--line)"
        stroke-width="1.5"
        stroke-dasharray="6 4"
      />
      <text x={FRAME_X + FRAME_W - 12} y={FRAME_Y + 20} text-anchor="end" class="frame-label">
        × 61 blocks
      </text>
      <text x={FRAME_X + 12} y={FRAME_Y + 20} class="frame-sublabel">one block, repeated</text>

      <!-- MLA Attention -->
      <g>
        <rect
          x={ATTN_X}
          y={ATTN_Y}
          width={ATTN_W}
          height={ATTN_H}
          rx="10"
          fill={isSel('attention') ? 'var(--paper-raised)' : 'var(--paper)'}
          stroke={isSel('attention') ? 'var(--accent-red)' : 'var(--line)'}
          stroke-width={isSel('attention') ? 3 : 1.5}
        />
        <text x={CX} y={ATTN_Y + 24} text-anchor="middle" class="box-label">MLA Attention</text>
        <text x={CX} y={ATTN_Y + 40} text-anchor="middle" class="box-sublabel">
          KV compressed → 512-d latent
        </text>
      </g>

      <!-- MoE FFN -->
      <g>
        <rect
          x={MOE_X}
          y={MOE_Y}
          width={MOE_W}
          height={MOE_H}
          rx="10"
          fill={isSel('moe') ? 'var(--paper-raised)' : 'var(--paper)'}
          stroke={isSel('moe') ? 'var(--accent-red)' : 'var(--line)'}
          stroke-width={isSel('moe') ? 3 : 1.5}
        />
        <text x={CX} y={MOE_Y + 20} text-anchor="middle" class="box-label">MoE FFN</text>

        <rect x={MOE_X + 20} y={MOE_Y + 26} width="18" height="18" rx="3" fill="var(--accent-gold)" />
        <text x={MOE_X + 44} y={MOE_Y + 39} class="chip-label">1 shared</text>

        {#each expertCells as cell (cell.i)}
          <rect
            x={cell.x}
            y={cell.y}
            width={CELL_W}
            height={CELL_H}
            fill={cell.active ? 'var(--accent-red)' : 'var(--paper)'}
            stroke="var(--line)"
            stroke-width="0.5"
          />
        {/each}

        <text x={CX} y={MOE_Y + MOE_H - 32} text-anchor="middle" class="box-sublabel">
          256 routed experts, top-8 active per token
        </text>
        <text x={CX} y={MOE_Y + MOE_H - 16} text-anchor="middle" class="box-sublabel">
          (highlighted squares: an illustrative top-8, not a real routing)
        </text>
      </g>

      <!-- LM head -->
      <g>
        <rect
          x={LMHEAD_X}
          y={LMHEAD_Y}
          width={LMHEAD_W}
          height={LMHEAD_H}
          rx="10"
          fill={isSel('lmhead') ? 'var(--paper-raised)' : 'var(--paper)'}
          stroke={isSel('lmhead') ? 'var(--accent-red)' : 'var(--line)'}
          stroke-width={isSel('lmhead') ? 3 : 1.5}
        />
        <text x={CX} y={LMHEAD_Y + 20} text-anchor="middle" class="box-label">LM head</text>
        <text x={CX} y={LMHEAD_Y + 36} text-anchor="middle" class="box-sublabel">
          ≈128k context · MTP objective
        </text>
      </g>
    </svg>

    <div class="region-buttons" role="group" aria-label="Architecture region">
      {#each REGIONS as region, i (region.id)}
        <button
          type="button"
          class:active={selected === i}
          aria-pressed={selected === i}
          onclick={() => select(i)}
        >
          {region.label}
        </button>
      {/each}
    </div>

    <p class="fact">
      <span class="phase label">{current.label}</span> {current.fact}
    </p>

    <div class="params-bar" role="group" aria-label="Total vs active parameters">
      <div class="bar-row">
        <span class="bar-label">Total</span>
        <span class="bar-track">
          <span class="bar-fill total" style="width: 100%"></span>
        </span>
        <span class="bar-value">{formatParams(TOTAL_PARAMS)}</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">Active</span>
        <span class="bar-track">
          <span class="bar-fill active" style={`width: ${activeBarPct}%`}></span>
        </span>
        <span class="bar-value">{formatParams(ACTIVE_PARAMS)}</span>
      </div>
      <p class="params-note">
        ≈{SPARSITY_RATIO.toFixed(0)}× fewer parameters active per token than the model holds in
        total — chapter 15's sparsity win, at frontier scale. Read "1 trillion parameters" as a
        knowledge-capacity number, not a per-token compute number.
      </p>
    </div>
  {/snippet}
</FigureShell>

<style>
  .disclosure {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    font-style: italic;
    color: var(--muted);
    text-align: center;
    margin: 0 0 0.85rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px dashed var(--line);
  }
  .box-label {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 700;
    fill: var(--ink-strong);
  }
  .box-sublabel {
    font-family: var(--font-ui);
    font-size: 10px;
    fill: var(--muted);
  }
  .frame-label {
    font-family: var(--font-ui);
    font-size: 12px;
    font-weight: 700;
    fill: var(--link);
  }
  .frame-sublabel {
    font-family: var(--font-ui);
    font-size: 10px;
    fill: var(--faint);
  }
  .chip-label {
    font-family: var(--font-ui);
    font-size: 10px;
    fill: var(--ink-strong);
  }
  .region-buttons {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .region-buttons button {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    padding: 0.35rem 0.9rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .region-buttons button:hover {
    border-color: var(--ink-strong);
  }
  .region-buttons button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
  .fact {
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
  .params-bar {
    margin-top: 1rem;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 3.2rem 1fr 4rem;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
  }
  .bar-label {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--ink-strong);
  }
  .bar-track {
    display: block;
    height: 1rem;
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 5px;
    overflow: hidden;
  }
  .bar-fill {
    display: block;
    height: 100%;
  }
  .bar-fill.total {
    background: var(--accent-gold);
  }
  .bar-fill.active {
    background: var(--accent-red);
  }
  .bar-value {
    font-family: var(--font-code);
    font-size: 0.8rem;
    text-align: right;
    color: var(--muted);
  }
  .params-note {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    color: var(--muted);
    text-align: center;
    margin: 0.5rem 0 0;
  }
</style>
