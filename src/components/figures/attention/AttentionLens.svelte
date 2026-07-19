<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { createEncoder, type Tokenizer } from '../../../lib/nn/bpe';
  import { loadModel, forward, type TransformerModel } from '../../../lib/nn/transformer';
  import { url } from '../../../lib/site/url';

  type Status = 'loading' | 'error' | 'ready';

  const DEFAULT_TEXT = 'The Queen turned angrily to Alice and said hold your tongue';
  const MAX_TOKENS = 32;

  let status = $state<Status>('loading');
  let encoder = $state<ReturnType<typeof createEncoder> | null>(null);
  let model = $state<TransformerModel | null>(null);
  let text = $state(DEFAULT_TEXT);

  // last-analyzed result — independent of the live text box, so editing the
  // input doesn't blank the figure until Analyze is pressed again.
  let tokenIds = $state<number[]>([]);
  let clipped = $state(false);
  let attention = $state<number[][][][] | null>(null); // [layer][head][query][key]
  // Default to layer 2 / head 3: on the default sentence this head puts the
  // bulk of its weight on a single, sensible token ("said" -> "Alice"),
  // which makes for a much clearer first impression than the diffuse layer
  // 1 / head 1 view. Verified empirically, not guessed — see the chapter's
  // guided walkthrough, which contrasts this against the fuzzier heads.
  let layer = $state(1);
  let headSel = $state<number | 'avg'>(2);
  let selectedQuery = $state(0);

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
      status = 'ready';
      analyze();
    } catch {
      status = 'error';
    }
  }

  load();

  function analyze() {
    if (!encoder || !model || status !== 'ready') return;
    const full = encoder.encode(text);
    const ids = full.slice(0, MAX_TOKENS);
    clipped = full.length > ids.length;
    tokenIds = ids;
    if (ids.length === 0) {
      attention = null;
      selectedQuery = 0;
      return;
    }
    attention = forward(model, ids).attention;
    selectedQuery = ids.length - 1;
  }

  const T = $derived(tokenIds.length);
  const cell = $derived(T > 0 ? Math.min(10, 320 / T) : 10);

  const matrix = $derived.by(() => {
    if (!attention || T === 0) return [] as number[][];
    const layerAttn = attention[layer]; // [head][q][k]
    if (headSel === 'avg') {
      const nHead = layerAttn.length;
      const out: number[][] = Array.from({ length: T }, () => Array(T).fill(0));
      for (let h = 0; h < nHead; h++) {
        for (let q = 0; q < T; q++) {
          for (let k = 0; k < T; k++) {
            out[q][k] += layerAttn[h][q][k] / nHead;
          }
        }
      }
      return out;
    }
    return layerAttn[headSel];
  });

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

  const top3 = $derived.by(() => {
    if (!matrix.length || selectedQuery >= matrix.length) return [] as { k: number; w: number }[];
    const row = matrix[selectedQuery];
    return row
      .map((w, k) => ({ k, w }))
      .filter((x) => x.k <= selectedQuery)
      .sort((a, b) => b.w - a.w)
      .slice(0, 3);
  });

  const readoutText = $derived.by(() => {
    if (!top3.length || selectedQuery >= tokenIds.length) return '';
    const label = tokLabel(tokenIds[selectedQuery]);
    const parts = top3.map((item) => `«${tokLabel(tokenIds[item.k])}» ${Math.round(item.w * 100)}%`);
    return `«${label}» attends to: ${parts.join(' · ')}`;
  });

  function selectQuery(i: number) {
    selectedQuery = i;
  }
</script>

<FigureShell
  title="Attention, live: who looks back at whom"
  caption="Honest note: this two-layer toy's heads are fuzzier than a frontier model's — but the mechanism, every token scoring every earlier token and then mixing their values, is identical. These weights aren't canned; they're computed live, in your tab, from the sentence you typed."
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
      <div class="chips" role="group" aria-label="Tokens — click or focus one to inspect what it attends to">
        {#each tokenIds as id, i (i)}
          <button
            type="button"
            class="chip"
            class:selected={i === selectedQuery}
            aria-pressed={i === selectedQuery}
            onclick={() => selectQuery(i)}
            onfocus={() => selectQuery(i)}
          >
            {chipDisplay(encoder ? encoder.tokenText(id) : '')}
          </button>
        {/each}
      </div>

      <svg
        viewBox="-2 -2 {T * cell + 4} {T * cell + 4}"
        class="heatmap"
        role="img"
        aria-label="{T} by {T} attention heatmap; rows are query positions, columns are key positions; darker cells mean stronger attention weight; the upper triangle is hatched because future positions are causally masked"
      >
        <defs>
          <pattern id="attn-hatch" width="5" height="5" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--line-soft)" stroke-width="1" />
          </pattern>
        </defs>
        <rect x="0" y="0" width={T * cell} height={T * cell} fill="url(#attn-hatch)" opacity="0.6" />
        {#each matrix as row, q (q)}
          {#each row as w, k (k)}
            {#if k <= q}
              <rect
                x={k * cell}
                y={q * cell}
                width={cell}
                height={cell}
                fill="var(--accent-blue)"
                opacity={0.06 + 0.94 * Math.max(0, Math.min(1, w))}
              />
            {/if}
          {/each}
        {/each}
        {#if selectedQuery < T}
          <rect
            x="0"
            y={selectedQuery * cell}
            width={T * cell}
            height={cell}
            fill="none"
            stroke="var(--accent-red)"
            stroke-width="2"
          />
        {/if}
      </svg>

      <p class="readout" aria-live="polite">{readoutText}</p>
      {#if clipped}
        <p class="note">clipped to the first {MAX_TOKENS} tokens</p>
      {/if}
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

    <span class="group" role="group" aria-label="Layer">
      {#each [0, 1] as l (l)}
        <button
          type="button"
          class:active={layer === l}
          aria-pressed={layer === l}
          disabled={status !== 'ready' || T === 0}
          onclick={() => (layer = l)}
        >
          Layer {l + 1}
        </button>
      {/each}
    </span>

    <span class="group" role="group" aria-label="Head">
      {#each [0, 1, 2, 3] as h (h)}
        <button
          type="button"
          class:active={headSel === h}
          aria-pressed={headSel === h}
          disabled={status !== 'ready' || T === 0}
          onclick={() => (headSel = h)}
        >
          Head {h + 1}
        </button>
      {/each}
      <button
        type="button"
        class:active={headSel === 'avg'}
        aria-pressed={headSel === 'avg'}
        disabled={status !== 'ready' || T === 0}
        onclick={() => (headSel = 'avg')}
      >
        Average
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
  .heatmap {
    display: block;
    width: 100%;
    max-width: 340px;
    height: auto;
    margin: 0 auto;
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
  .note {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    text-align: center;
    color: var(--faint);
    margin: 0.25rem 0 0;
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
