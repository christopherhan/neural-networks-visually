<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { createTicker } from '../../../lib/viz/ticker';

  // No model here — every number below is arithmetic on realistic,
  // published-scale constants (see the preset notes), not this site's own
  // toy transformer.

  // ---------------------------------------------------------------------
  // Part A: the "why cache" mini-stepper. Three steps, reusing the
  // step/play/reset pattern from BackpropStepper/BlockAssembler.
  // ---------------------------------------------------------------------

  const T_DEMO = 6; // toy sequence length for the diagram, generated 1..6

  interface StepDef {
    id: string;
    label: string;
    text: string;
  }

  const recomputeTotal = (T_DEMO * (T_DEMO + 1)) / 2; // 1+2+...+T
  const cacheTotal = T_DEMO; // one new (K,V) per step

  const STEPS: StepDef[] = [
    {
      id: 'recompute',
      label: 'no cache',
      text: `Generate token ${T_DEMO} with no cache and every single step recomputes the keys and values for every earlier token, from scratch, again. Across a ${T_DEMO}-token generation that's 1+2+…+${T_DEMO} = ${recomputeTotal} total (K,V) computations — the O(T²) chapter 10's footer already flagged.`,
    },
    {
      id: 'cache',
      label: 'with cache',
      text: `Save each token's key and value in memory the moment it's computed. Now every step does exactly ONE new (K,V) computation and reuses everything already sitting in the cache — ${cacheTotal} total computations for the same ${T_DEMO} tokens. O(T²) became O(T).`,
    },
    {
      id: 'price',
      label: 'the price',
      text: `Nothing here was free — it moved the cost, it didn't erase it. Every one of those cached keys and values has to stay resident in GPU memory for as long as generation continues, and that memory only grows, never shrinks. How fast, for real models, is exactly what the calculator below prices out.`,
    },
  ];
  const STEP_TOTAL = STEPS.length;

  let stepIndex = $state(0);
  let playing = $state(false);

  const phase = $derived(stepIndex === 0 ? 'ready' : STEPS[stepIndex - 1].id);
  const current = $derived(stepIndex === 0 ? null : STEPS[stepIndex - 1]);

  function advanceStep() {
    if (stepIndex < STEP_TOTAL) stepIndex += 1;
    if (stepIndex >= STEP_TOTAL) stopPlay();
  }

  const stepTicker = createTicker(advanceStep, { fps: 0.6 });
  onDestroy(() => stepTicker.stop());

  function stopPlay() {
    playing = false;
    stepTicker.stop();
  }

  function togglePlay() {
    playing = !playing;
    if (playing) {
      if (stepIndex >= STEP_TOTAL) stepIndex = 0;
      stepTicker.start();
    } else {
      stepTicker.stop();
    }
  }

  function resetStep() {
    stopPlay();
    stepIndex = 0;
  }

  // ---- Stepper diagram geometry: T_DEMO columns, each a stack of unit
  // squares — one square per (K,V) pair a step at that column touches.
  const COL_W = 96;
  const SQ_W = 40;
  const SQ_H = 17;
  const SQ_GAP = 3;
  const START_X = 56;
  const BASE_Y = 214;

  const columns = Array.from({ length: T_DEMO }, (_, i) => ({
    t: i + 1,
    x: START_X + i * COL_W,
  }));

  function squareY(row: number): number {
    return BASE_Y - (row + 1) * (SQ_H + SQ_GAP);
  }

  // ---------------------------------------------------------------------
  // Part B: the KV cache calculator. Formula: 2 · layers · kv_heads ·
  // d_head · 2 bytes(fp16), summed over the chosen context length.
  // ---------------------------------------------------------------------

  const D_HEAD = 128; // the head width most published frontier stacks converge on
  const PRECISION_BYTES = 2; // fp16

  const CONTEXT_EXP_MIN = 10; // 2^10 = 1,024 tokens
  const CONTEXT_EXP_MAX = 20; // 2^20 = 1,048,576 tokens
  const LAYERS_MIN = 24;
  const LAYERS_MAX = 96;

  type PresetId = 'mha' | 'gqa' | 'mla';

  interface PresetDef {
    id: PresetId;
    label: string;
    tag: string;
    kvHeads: number; // "effective" kv-heads at d_head=128 that reproduce the published footprint
    note: string;
  }

  // MHA: one KV head per query head, no sharing — the un-optimized baseline.
  // GQA: query heads share a much smaller pool of KV heads; Llama-70B ships
  // 8 KV heads. MLA doesn't have discrete KV heads at all — DeepSeek/K2
  // compress K and V into one shared low-rank latent per token per layer and
  // reconstruct on the fly. There's no honest single "head count" for that,
  // so MLA's row uses an *effective* kv_heads value, back-derived from the
  // published-family framing (roughly 1/12 of GQA's already-reduced
  // footprint) so it slots into the same formula as the other two rows —
  // labeled as such below, not presented as a literal head count.
  const PRESETS: PresetDef[] = [
    {
      id: 'mha',
      label: 'MHA',
      tag: '128 heads',
      kvHeads: 128,
      note: 'Multi-head attention: every query head gets its own key/value head, one-to-one, no sharing at all. The un-optimized baseline every other scheme improves on.',
    },
    {
      id: 'gqa',
      label: 'GQA',
      tag: '8 KV heads · Llama-70B-family',
      kvHeads: 8,
      note: 'Grouped-query attention: many query heads share one KV head. Llama-70B-family models publish 8 KV heads — a 16× smaller cache than MHA at the same layer count, for a small, well-studied quality cost.',
    },
    {
      id: 'mla',
      label: 'MLA',
      tag: 'latent · DeepSeek/K2-family',
      kvHeads: 8 / 12,
      note: "Multi-head latent attention: K and V are compressed into one shared low-rank latent per token per layer and reconstructed on the fly. Published DeepSeek/K2-family figures put the resulting cache at roughly 1/12 of GQA's already-reduced footprint — real numbers, not this toy's arithmetic alone.",
    },
  ];

  let contextExp = $state(17); // 2^17 = 131,072 ≈ 128k, the contract's default
  let layers = $state(61); // DeepSeek-V3's published layer count
  let presetId = $state<PresetId>('gqa');

  const contextTokens = $derived(2 ** contextExp);
  const preset = $derived(PRESETS.find((p) => p.id === presetId)!);

  function bytesPerToken(kvHeads: number): number {
    // Rounded to the nearest byte — MLA's kv_heads is an "effective" (and
    // thus fractional) figure, but a byte count is inherently a whole
    // number, so fractional heads shouldn't leak into it as a raw decimal.
    return Math.round(2 * layers * kvHeads * D_HEAD * PRECISION_BYTES);
  }

  function totalBytes(kvHeads: number): number {
    return bytesPerToken(kvHeads) * contextTokens;
  }

  function formatContext(exp: number): string {
    if (exp >= 20) return `${2 ** (exp - 20)}M`;
    return `${2 ** (exp - 10)}k`;
  }

  function formatBytes(bytes: number): string {
    const abs = Math.abs(bytes);
    if (abs >= 1e9) return `${(bytes / 1e9).toFixed(bytes / 1e9 >= 100 ? 0 : 1)} GB`;
    if (abs >= 1e6) return `${(bytes / 1e6).toFixed(bytes / 1e6 >= 100 ? 0 : 1)} MB`;
    if (abs >= 1e3) return `${(bytes / 1e3).toFixed(bytes / 1e3 >= 100 ? 0 : 1)} KB`;
    return `${bytes.toFixed(0)} B`;
  }

  // Fixed log-scale domain so bar widths stay visually stable as the
  // sliders move, instead of re-scaling relative to each other every drag.
  const LOG_MIN = 3; // ~1 KB
  const LOG_MAX = 12.3; // ~2 TB, comfortably above the sliders' max
  function barPct(bytes: number): number {
    const l = Math.log10(Math.max(bytes, 1));
    const t = ((l - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100;
    return Math.min(100, Math.max(3, t));
  }

  const rows = $derived(
    PRESETS.map((p) => ({
      ...p,
      bytesPerTok: bytesPerToken(p.kvHeads),
      total: totalBytes(p.kvHeads),
      pct: barPct(totalBytes(p.kvHeads)),
    }))
  );

  const fmt = (n: number) => n.toLocaleString('en-US');
</script>

<FigureShell
  title="Why bother caching?"
  caption="Same causal attention as chapter 10, viewed as a running cost. Step through: recompute everything every time, or compute one new pair and keep the rest."
>
  {#snippet children()}
    <svg
      viewBox={`0 0 700 250`}
      aria-label={`Diagram of generating ${T_DEMO} tokens: without a cache every step recomputes every earlier token's key and value; with a cache each step adds exactly one new pair and reuses the rest`}
    >
      {#each columns as col (col.t)}
        {#each Array.from({ length: stepIndex === 0 ? 0 : col.t }) as _, row}
          {@const isTop = row === col.t - 1}
          {@const cached = phase !== 'recompute' && !isTop}
          <rect
            x={col.x}
            y={squareY(row)}
            width={SQ_W}
            height={SQ_H}
            rx="3"
            fill={cached ? 'var(--paper)' : phase === 'recompute' ? 'var(--accent-gold)' : 'var(--accent-blue)'}
            stroke={phase === 'price' ? 'var(--accent-red)' : 'var(--line)'}
            stroke-width={phase === 'price' ? 2 : 1}
            opacity={cached ? 0.55 : 1}
          />
        {/each}
        <text x={col.x + SQ_W / 2} y={BASE_Y + 18} text-anchor="middle" class="col-label">
          t={col.t}
        </text>
      {/each}
      <text x={20} y={16} class="axis-label">(K,V) pairs touched, per generation step</text>
    </svg>

    <p class="explain" data-phase={phase}>
      {#if current}
        <span class="phase label">{current.label}</span> {current.text}
      {:else}
        <span class="phase label">ready</span> Press Step (or Play) to generate token by token and watch
        the running cost.
      {/if}
    </p>
  {/snippet}

  {#snippet controls()}
    <button type="button" onclick={advanceStep} disabled={stepIndex >= STEP_TOTAL}>Step</button>
    <button type="button" onclick={togglePlay}>{playing ? 'Pause' : 'Play'}</button>
    <button type="button" onclick={resetStep}>Reset</button>
    <span class="progress">{stepIndex} / {STEP_TOTAL}</span>
  {/snippet}
</FigureShell>

<FigureShell
  title="The KV cache calculator"
  caption="Same formula, three real caching strategies. Drag the sliders, switch presets — the bars compare all three at whatever context and layer count you land on."
>
  {#snippet children()}
    <p class="formula">
      KV cache = 2 · layers · kv_heads · d_head · 2 bytes (fp16), summed over every cached token
    </p>

    <div
      class="bars"
      role="group"
      aria-label="KV cache size comparison across MHA, GQA, and MLA at the current context length and layer count"
    >
      {#each rows as row (row.id)}
        <div class="bar-row" class:selected={row.id === presetId}>
          <span class="bar-label">{row.label}</span>
          <span class="bar-track">
            <span class="bar-fill" class:selected={row.id === presetId} style={`width: ${row.pct}%`}></span>
          </span>
          <span class="bar-value">{formatBytes(row.total)}</span>
        </div>
      {/each}
    </div>
    <p class="bar-note">
      Bars are log-scaled — MHA, GQA, and MLA differ here by two orders of magnitude, not a
      visually-honest linear amount.
    </p>

    <p class="plug">
      {#if preset.id === 'mla'}
        {@const gqaKvHeads = PRESETS.find((p) => p.id === 'gqa')!.kvHeads}
        <strong>{preset.label}</strong>: GQA's {fmt(bytesPerToken(gqaKvHeads))} bytes/token ÷ 12
        (published-family latent compression) → <strong>{fmt(bytesPerToken(preset.kvHeads))} bytes/token</strong>
        — × {fmt(contextTokens)} tokens ≈ <strong>{formatBytes(totalBytes(preset.kvHeads))}</strong> at
        {formatContext(contextExp)} context.
      {:else}
        <strong>{preset.label}</strong>: 2 · {layers} · {preset.kvHeads} · {D_HEAD} · 2 =
        <strong>{fmt(bytesPerToken(preset.kvHeads))} bytes/token</strong> —
        × {fmt(contextTokens)} tokens ≈ <strong>{formatBytes(totalBytes(preset.kvHeads))}</strong> at
        {formatContext(contextExp)} context.
      {/if}
    </p>

    <p class="fact"><span class="phase label">{preset.label}</span> {preset.note}</p>
  {/snippet}

  {#snippet controls()}
    <div class="preset-buttons" role="group" aria-label="KV caching strategy">
      {#each PRESETS as p (p.id)}
        <button
          type="button"
          class:active={presetId === p.id}
          aria-pressed={presetId === p.id}
          onclick={() => (presetId = p.id)}
        >
          {p.label} <span class="tag">{p.tag}</span>
        </button>
      {/each}
    </div>
    <label>
      context
      <input type="range" min={CONTEXT_EXP_MIN} max={CONTEXT_EXP_MAX} step="1" bind:value={contextExp} />
      {formatContext(contextExp)} ({fmt(contextTokens)} tokens)
    </label>
    <label>
      layers
      <input type="range" min={LAYERS_MIN} max={LAYERS_MAX} step="1" bind:value={layers} />
      {layers}
    </label>
    <span class="dhead-note">d_head fixed at 128</span>
  {/snippet}
</FigureShell>

<style>
  .col-label {
    font-family: var(--font-ui);
    font-size: 12px;
    fill: var(--muted);
  }
  .axis-label {
    font-family: var(--font-ui);
    font-size: 11px;
    fill: var(--faint);
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
  .explain[data-phase='price'] {
    border-left-color: var(--accent-red);
  }
  .phase {
    margin-right: 0.5rem;
  }
  .progress {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--faint);
  }

  .formula {
    font-family: var(--font-code);
    font-size: 0.85rem;
    text-align: center;
    color: var(--ink-strong);
    background: var(--paper);
    border: 1px dashed var(--line);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0 0 0.9rem;
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 3.2rem 1fr 6.5rem;
    align-items: center;
    gap: 0.5rem;
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
    background: var(--accent-gold);
  }
  .bar-fill.selected {
    background: var(--accent-red);
  }
  .bar-value {
    font-family: var(--font-code);
    font-size: 0.8rem;
    text-align: right;
    color: var(--muted);
  }
  .bar-row.selected .bar-label {
    color: var(--link);
  }
  .bar-note {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    color: var(--faint);
    text-align: center;
    margin: 0.4rem 0 0;
  }
  .plug {
    font-family: var(--font-code);
    font-size: 0.82rem;
    color: var(--ink);
    margin: 0.9rem 0 0;
    text-align: center;
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
  .preset-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .preset-buttons button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.1rem;
    font-family: var(--font-ui);
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.3rem 0.7rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .preset-buttons button:hover {
    border-color: var(--ink-strong);
  }
  .preset-buttons button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
  .preset-buttons .tag {
    font-family: var(--font-ui);
    font-size: 0.68rem;
    font-weight: 400;
    color: var(--muted);
  }
  .dhead-note {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    color: var(--faint);
  }
</style>
