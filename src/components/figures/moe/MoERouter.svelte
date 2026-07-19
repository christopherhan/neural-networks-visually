<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { createEncoder, type Tokenizer } from '../../../lib/nn/bpe';
  import { loadModel, softmaxTemp, type TransformerModel } from '../../../lib/nn/transformer';
  import { mulberry32, gaussian } from '../../../lib/nn/rng';
  import { url } from '../../../lib/site/url';

  type Status = 'loading' | 'error' | 'ready';
  type FlopsMode = 'dense' | 'sparse';

  const DEFAULT_TEXT = 'The Queen said Alice must run faster';
  const MAX_TOKENS = 24;
  const N_EXPERTS = 8;
  const TOP_K = 2;
  const D_MODEL = 64; // this site's model's embedding width (chapter 9/11)
  const MLP_HIDDEN = 256; // this site's model's MLP hidden width (chapter 11)
  // Gate weight scale: chosen empirically so the softmax over 8 experts is
  // visibly non-uniform without collapsing to a near-one-hot pick on token 1.
  // The matrix itself is never trained — see the honest note below.
  const GATE_SCALE = 0.25;

  let status = $state<Status>('loading');
  let encoder = $state<ReturnType<typeof createEncoder> | null>(null);
  let model = $state<TransformerModel | null>(null);
  let gateMatrix: number[][] | null = null; // [D_MODEL][N_EXPERTS], fixed once per load

  let text = $state(DEFAULT_TEXT);
  let tokenIds = $state<number[]>([]);
  let clipped = $state(false);
  let gateProbs = $state<number[][]>([]); // [token][expert] — softmax gate output per token
  // Default-select " Queen" (token index 1 on the default sentence: "The" | " Queen" | ...)
  // so the figure opens on a concrete, checkable routing decision rather than a blank state.
  let selected = $state(1);
  let flopsMode = $state<FlopsMode>('sparse');

  function buildGateMatrix(): number[][] {
    const rng = mulberry32(5);
    return Array.from({ length: D_MODEL }, () =>
      Array.from({ length: N_EXPERTS }, () => gaussian(rng) * GATE_SCALE),
    );
  }

  async function load() {
    status = 'loading';
    try {
      const [tokRes, wRes] = await Promise.all([
        fetch(url('models/tokenizer.json')),
        fetch(url('models/weights.json')),
      ]);
      if (!tokRes.ok) throw new Error(`HTTP ${tokRes.status}`);
      if (!wRes.ok) throw new Error(`HTTP ${wRes.status}`);
      const [tokJson, wJson] = await Promise.all([tokRes.json(), wRes.json()]);
      encoder = createEncoder(tokJson as Tokenizer);
      model = loadModel(wJson);
      gateMatrix = buildGateMatrix();
      status = 'ready';
      analyze();
    } catch {
      status = 'error';
    }
  }

  load();

  /** Project a token's real (trained) embedding through the untrained gate
   *  matrix and softmax the result — the whole router, in eight lines. */
  function gateFor(embedding: number[]): number[] {
    if (!gateMatrix) return [];
    const logits = new Array(N_EXPERTS).fill(0);
    for (let d = 0; d < D_MODEL; d++) {
      const xd = embedding[d];
      if (xd === 0) continue;
      const row = gateMatrix[d];
      for (let e = 0; e < N_EXPERTS; e++) logits[e] += xd * row[e];
    }
    return softmaxTemp(logits, 1);
  }

  function analyze() {
    if (!encoder || !model || status !== 'ready') return;
    const full = encoder.encode(text);
    const ids = full.slice(0, MAX_TOKENS);
    clipped = full.length > ids.length;
    tokenIds = ids;
    gateProbs = ids.map((id) => gateFor(model!.tok_emb[id]));
    selected = ids.length > 1 ? 1 : 0;
  }

  function topK(probs: number[]): { e: number; p: number }[] {
    return probs
      .map((p, e) => ({ e, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, TOP_K);
  }

  function cleanTok(tok: string): string {
    const trimmed = tok.trim();
    return trimmed || (tok === '\n' ? '⏎' : tok);
  }
  function chipDisplay(tok: string): string {
    return tok === '\n' ? '⏎' : tok.startsWith(' ') ? '␣' + tok.slice(1) : tok;
  }
  function tokLabel(id: number): string {
    return encoder ? cleanTok(encoder.tokenText(id)) : '';
  }

  function selectToken(i: number) {
    selected = i;
  }

  const T = $derived(tokenIds.length);
  const selectedProbs = $derived(gateProbs[selected] ?? []);
  const selectedTop = $derived(topK(selectedProbs));
  const selectedExperts = $derived(new Set(selectedTop.map((x) => x.e)));

  // Aggregate load: how many of the sentence's tokens picked each expert in
  // their top-2. An untrained, random gate has no reason to balance this —
  // that imbalance IS the point (real systems fight it with an aux loss).
  const loadCounts = $derived.by(() => {
    const counts = new Array(N_EXPERTS).fill(0);
    for (const probs of gateProbs) {
      for (const { e } of topK(probs)) counts[e] += 1;
    }
    return counts;
  });
  const maxLoad = $derived(Math.max(1, ...loadCounts));

  const readoutText = $derived.by(() => {
    if (!T || !selectedTop.length) return '';
    const label = tokLabel(tokenIds[selected]);
    const parts = selectedTop.map((t) => `Expert ${t.e + 1} ${Math.round(t.p * 100)}%`);
    return `«${label}» routes to: ${parts.join(' · ')}`;
  });

  // ---- FLOPs accounting: actual multiply counts for this toy, not this site's
  // real model (which has one dense MLP, no MoE) — a hypothetical MoE layer
  // built from N_EXPERTS copies of chapter 11's exact MLP shape (64 -> 256 -> 64).
  const EXPERT_MULTIPLIES = 2 * D_MODEL * MLP_HIDDEN; // 32,768 — chapter 11's MathAside number
  const ROUTER_MULTIPLIES = D_MODEL * N_EXPERTS; // 512 — this figure's own gate matmul
  const DENSE_MULTIPLIES = N_EXPERTS * EXPERT_MULTIPLIES; // 262,144 — every expert fires
  const SPARSE_MULTIPLIES = TOP_K * EXPERT_MULTIPLIES; // 65,536 — only the top-2 fire
  const FLOPS_RATIO = DENSE_MULTIPLIES / SPARSE_MULTIPLIES; // exactly 4

  function fmt(n: number): string {
    return n.toLocaleString('en-US');
  }

  // ---- diagram geometry ----
  const VIEW_W = 640;
  const VIEW_H = 210;
  const BOX_W = 62;
  const BOX_H = 46;
  const GAP = (VIEW_W - N_EXPERTS * BOX_W) / (N_EXPERTS - 1);
  const EXPERT_Y = 78;
  const SOURCE_W = 130;
  const SOURCE_H = 30;
  const SOURCE_X = VIEW_W / 2 - SOURCE_W / 2;
  const SOURCE_Y = 8;
  const LOAD_BASE_Y = EXPERT_Y + BOX_H + 46;
  const LOAD_MAX_H = 38;

  const expertX = Array.from({ length: N_EXPERTS }, (_, i) => i * (BOX_W + GAP));

  const sourceLabel = $derived(T ? chipDisplay(encoder ? encoder.tokenText(tokenIds[selected]) : '') : '');
</script>

<FigureShell
  title="Mixture-of-experts routing, live"
  caption="Honest note: this router's gate matrix is a random, untrained 64→8 projection — it was never optimized to route anything. Its embeddings are real, though (the trained model's own token vectors), so the routing you see here is the raw mechanics of the mechanism, not a demonstration of what a trained router would learn to do."
>
  {#snippet children()}
    {#if status === 'loading'}
      <div class="placeholder" role="status">loading the model (&asymp;1.2 MB)&hellip;</div>
    {:else if status === 'error'}
      <div class="placeholder error" role="alert">
        <p>Couldn&rsquo;t load the model.</p>
        <button type="button" onclick={load}>Retry</button>
      </div>
    {:else if T === 0}
      <div class="placeholder">Type a sentence and press Analyze.</div>
    {:else}
      <div class="chips" role="group" aria-label="Tokens — click or focus one to see which experts it routes to">
        {#each tokenIds as id, i (i)}
          <button
            type="button"
            class="chip"
            class:selected={i === selected}
            aria-pressed={i === selected}
            onclick={() => selectToken(i)}
            onfocus={() => selectToken(i)}
          >
            {chipDisplay(encoder ? encoder.tokenText(id) : '')}
          </button>
        {/each}
      </div>

      <svg
        viewBox="0 0 {VIEW_W} {VIEW_H}"
        class="diagram"
        role="img"
        aria-label="Router diagram: the selected token's real embedding is projected through the 64-by-8 gate matrix; its top-2 experts are highlighted and connected with arrows; a load bar under every expert shows how many of the sentence's tokens picked it."
      >
        <defs>
          <marker id="moe-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--accent-red)" />
          </marker>
        </defs>

        {#each selectedTop as t, rank (t.e)}
          <line
            x1={VIEW_W / 2}
            y1={SOURCE_Y + SOURCE_H}
            x2={expertX[t.e] + BOX_W / 2}
            y2={EXPERT_Y}
            stroke="var(--accent-red)"
            stroke-width={rank === 0 ? 2.5 : 1.5}
            opacity={rank === 0 ? 1 : 0.6}
            stroke-dasharray={rank === 0 ? 'none' : '4 3'}
            marker-end="url(#moe-arrow)"
          />
        {/each}

        <rect
          x={SOURCE_X}
          y={SOURCE_Y}
          width={SOURCE_W}
          height={SOURCE_H}
          rx="8"
          fill="var(--paper-raised)"
          stroke="var(--accent-red)"
          stroke-width="1.5"
        />
        <text x={VIEW_W / 2} y={SOURCE_Y + SOURCE_H / 2 + 4} text-anchor="middle" class="source-label">
          {sourceLabel}
        </text>

        {#each expertX as ex, e (e)}
          <g>
            <rect
              x={ex}
              y={EXPERT_Y}
              width={BOX_W}
              height={BOX_H}
              rx="8"
              fill={selectedExperts.has(e) ? 'var(--paper-raised)' : 'var(--paper)'}
              stroke={selectedExperts.has(e) ? 'var(--accent-red)' : 'var(--line)'}
              stroke-width={selectedExperts.has(e) ? 2.5 : 1.5}
            />
            <text x={ex + BOX_W / 2} y={EXPERT_Y + BOX_H / 2 + 4} text-anchor="middle" class="expert-label">
              E{e + 1}
            </text>

            <rect
              x={ex + 6}
              y={LOAD_BASE_Y - (loadCounts[e] / maxLoad) * LOAD_MAX_H}
              width={BOX_W - 12}
              height={(loadCounts[e] / maxLoad) * LOAD_MAX_H}
              fill="var(--accent-blue)"
              opacity="0.75"
            />
            <text x={ex + BOX_W / 2} y={LOAD_BASE_Y + 16} text-anchor="middle" class="load-count">
              {loadCounts[e]}
            </text>
          </g>
        {/each}

        <line
          x1="0"
          y1={LOAD_BASE_Y}
          x2={VIEW_W}
          y2={LOAD_BASE_Y}
          stroke="var(--line-soft)"
          stroke-width="1"
        />
        <text x="4" y={LOAD_BASE_Y + 32} class="load-caption">aggregate load (tokens routed here)</text>
      </svg>

      <p class="readout" aria-live="polite">{readoutText}</p>

      <div class="bars" role="group" aria-label="Gate probabilities for the selected token, all 8 experts">
        {#each selectedProbs as p, e (e)}
          <div class="bar-row">
            <span class="bar-label">E{e + 1}</span>
            <span class="bar-track">
              <span
                class="bar-fill"
                class:top={selectedExperts.has(e)}
                style="width: {Math.min(100, p * 100)}%"
              ></span>
            </span>
            <span class="bar-pct">{(p * 100).toFixed(1)}%</span>
          </div>
        {/each}
      </div>

      {#if clipped}
        <p class="note">clipped to the first {MAX_TOKENS} tokens</p>
      {/if}

      <div class="flops">
        <div class="flops-row">
          <span class="flops-label">Dense (all 8)</span>
          <span class="flops-track">
            <span class="flops-fill" class:active={flopsMode === 'dense'} style="width: 100%"></span>
          </span>
          <span class="flops-count">{fmt(DENSE_MULTIPLIES)}</span>
        </div>
        <div class="flops-row">
          <span class="flops-label">Sparse (top-2)</span>
          <span class="flops-track">
            <span
              class="flops-fill"
              class:active={flopsMode === 'sparse'}
              style="width: {(100 / FLOPS_RATIO).toFixed(1)}%"
            ></span>
          </span>
          <span class="flops-count">{fmt(SPARSE_MULTIPLIES)}</span>
        </div>
        <p class="flops-readout">
          {#if flopsMode === 'dense'}
            Dense: every one of the 8 experts fires on every token — 8 &times; {fmt(EXPERT_MULTIPLIES)} = <strong
              >{fmt(DENSE_MULTIPLIES)}</strong
            > multiplies per token, run against the model's full parameter pool.
          {:else}
            Sparse (top-{TOP_K}): only the winning {TOP_K} experts fire — {TOP_K} &times; {fmt(EXPERT_MULTIPLIES)} = <strong
              >{fmt(SPARSE_MULTIPLIES)}</strong
            > multiplies per token (the router itself adds a further {fmt(ROUTER_MULTIPLIES)}, rounding noise next to
            either total). That's exactly <strong>{FLOPS_RATIO}&times; less compute</strong> per token than dense —
            spent against the identical {fmt(DENSE_MULTIPLIES)}-multiply pool of expert parameters. Parameters scale
            with how many experts you own; compute scales with how many you fire.
          {/if}
        </p>
      </div>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>
      text
      <input
        type="text"
        maxlength="140"
        bind:value={text}
        disabled={status !== 'ready'}
        placeholder="Type a sentence to analyze"
      />
    </label>
    <button type="button" onclick={analyze} disabled={status !== 'ready'}>Analyze</button>

    <span class="group" role="group" aria-label="FLOPs mode">
      <button
        type="button"
        class:active={flopsMode === 'dense'}
        aria-pressed={flopsMode === 'dense'}
        onclick={() => (flopsMode = 'dense')}
      >
        Dense
      </button>
      <button
        type="button"
        class:active={flopsMode === 'sparse'}
        aria-pressed={flopsMode === 'sparse'}
        onclick={() => (flopsMode = 'sparse')}
      >
        Sparse (top-2)
      </button>
    </span>
  {/snippet}
</FigureShell>

<style>
  .placeholder {
    min-height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    font-family: var(--font-ui);
    font-size: 0.9rem;
    color: var(--muted);
    text-align: center;
  }
  .placeholder.error {
    color: var(--accent-red);
  }
  .placeholder button {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    padding: 0.3rem 0.8rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .placeholder button:hover {
    border-color: var(--ink-strong);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    justify-content: center;
    margin-bottom: 0.75rem;
  }
  .chip {
    font-family: var(--font-code);
    font-size: 0.8rem;
    padding: 0.15rem 0.4rem;
    white-space: pre;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line-soft);
    border-radius: 5px;
    cursor: pointer;
  }
  .chip:hover {
    border-color: var(--ink-strong);
  }
  .chip.selected {
    border-color: var(--accent-red);
    color: var(--accent-red);
    background: var(--paper-raised);
  }
  .diagram {
    display: block;
    width: 100%;
    max-width: 560px;
    height: auto;
    margin: 0 auto;
  }
  .source-label {
    font-family: var(--font-code);
    font-size: 12px;
    fill: var(--ink-strong);
  }
  .expert-label {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 700;
    fill: var(--ink-strong);
  }
  .load-count {
    font-family: var(--font-ui);
    font-size: 11px;
    fill: var(--muted);
  }
  .load-caption {
    font-family: var(--font-ui);
    font-size: 10px;
    fill: var(--faint);
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.75rem 0 0;
    min-height: 1.4em;
  }
  .note {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    text-align: center;
    color: var(--faint);
    margin: 0.25rem 0 0;
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    max-width: 380px;
    margin: 0.75rem auto 0;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 2.4rem 1fr 3.2rem;
    align-items: center;
    gap: 0.5rem;
  }
  .bar-label {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--ink-strong);
  }
  .bar-track {
    display: block;
    height: 0.85rem;
    background: var(--paper);
    border: 1px solid var(--line-soft);
    border-radius: 4px;
    overflow: hidden;
  }
  .bar-fill {
    display: block;
    height: 100%;
    background: var(--accent-blue);
    opacity: 0.55;
  }
  .bar-fill.top {
    background: var(--accent-red);
    opacity: 1;
  }
  .bar-pct {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    color: var(--muted);
    text-align: right;
  }
  .flops {
    margin-top: 1.25rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--line-soft);
    max-width: 480px;
    margin-left: auto;
    margin-right: auto;
  }
  .flops-row {
    display: grid;
    grid-template-columns: 7rem 1fr 5rem;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
  }
  .flops-label {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    color: var(--ink-strong);
  }
  .flops-track {
    display: block;
    height: 0.85rem;
    background: var(--paper);
    border: 1px solid var(--line-soft);
    border-radius: 4px;
    overflow: hidden;
  }
  .flops-fill {
    display: block;
    height: 100%;
    background: var(--accent-blue);
    opacity: 0.4;
  }
  .flops-fill.active {
    opacity: 1;
    background: var(--accent-red);
  }
  .flops-count {
    font-family: var(--font-code);
    font-size: 0.78rem;
    color: var(--muted);
    text-align: right;
  }
  .flops-readout {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--ink);
    margin: 0.5rem 0 0;
  }
  .group {
    display: inline-flex;
    gap: 0.35rem;
  }
  button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
</style>
