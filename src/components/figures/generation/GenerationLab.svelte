<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { createEncoder, type Tokenizer } from '../../../lib/nn/bpe';
  import { loadModel, forward, softmaxTemp, sampleNext, type TransformerModel } from '../../../lib/nn/transformer';
  import { createTicker } from '../../../lib/viz/ticker';
  import { mulberry32, type Rng } from '../../../lib/nn/rng';
  import { url } from '../../../lib/site/url';

  type Status = 'loading' | 'error' | 'ready';

  const DEFAULT_PROMPT = 'Alice went';
  const CTX = 64; // matches the model's trained context window
  const MAX_GENERATED = 40;
  const TOP_K = 8;

  let status = $state<Status>('loading');
  let encoder = $state<ReturnType<typeof createEncoder> | null>(null);
  let model = $state<TransformerModel | null>(null);

  let prompt = $state(DEFAULT_PROMPT);
  let started = $state(false);
  let ids = $state<number[]>([]);
  let promptLen = $state(0);
  let generatedCount = $state(0);
  let temperature = $state(0.8);
  let rng: Rng | null = null;

  // last computed distribution — top-K candidates plus which one got sampled.
  let topK = $state<{ id: number; p: number }[]>([]);
  let sampledId = $state<number | null>(null);

  let autoOn = $state(false);

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
    } catch {
      status = 'error';
    }
  }

  load();

  const contextFull = $derived(ids.length >= CTX);
  const autoDone = $derived(generatedCount >= MAX_GENERATED);
  const canStep = $derived(started && !contextFull && !autoDone);

  function stopAuto() {
    autoOn = false;
    ticker.stop();
  }

  function start() {
    if (!encoder || !model || status !== 'ready') return;
    stopAuto();
    const encoded = encoder.encode(prompt).slice(0, CTX);
    ids = encoded;
    promptLen = encoded.length;
    generatedCount = 0;
    topK = [];
    sampledId = null;
    // Fresh seed per Start — every run of the same prompt at the same
    // temperature samples the identical sequence, deterministically.
    rng = mulberry32(42);
    started = encoded.length > 0;
  }

  function currentWindow(): number[] {
    return ids.length > CTX ? ids.slice(ids.length - CTX) : ids;
  }

  /** Forward, price every vocabulary token, sample one, append it. Returns
   *  false (and stops Auto) when generation can't or shouldn't continue. */
  function stepOnce(): boolean {
    if (!model || !rng || !started || contextFull || autoDone) {
      stopAuto();
      return false;
    }
    const window = currentWindow();
    const { logits } = forward(model, window);
    const lastLogits = logits[logits.length - 1];
    const probs = softmaxTemp(lastLogits, temperature);
    const ranked = probs.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p);
    topK = ranked.slice(0, TOP_K);
    const next = sampleNext(probs, rng);
    sampledId = next;
    ids = [...ids, next];
    generatedCount += 1;
    if (ids.length >= CTX || generatedCount >= MAX_GENERATED) {
      stopAuto();
      return false;
    }
    return true;
  }

  function nextToken() {
    stepOnce();
  }

  const ticker = createTicker(
    () => {
      stepOnce();
    },
    { fps: 2 },
  );
  onDestroy(() => ticker.stop());

  function toggleAuto() {
    if (autoOn) {
      stopAuto();
      return;
    }
    if (!canStep) return;
    autoOn = true;
    ticker.start();
  }

  function reset() {
    stopAuto();
    started = false;
    ids = [];
    promptLen = 0;
    generatedCount = 0;
    topK = [];
    sampledId = null;
    rng = null;
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
</script>

<FigureShell
  title="Next token, one price list at a time"
  caption="The model never picks a word — it prices every token in the vocabulary, softmax turns those prices into a probability budget, and sampling is a weighted dice roll against that budget. Watch the bars change shape as temperature turns the dial between cautious and chaotic."
>
  {#snippet children()}
    {#if status === 'loading'}
      <div class="placeholder" role="status">loading the model (&asymp;1.2 MB)&hellip;</div>
    {:else if status === 'error'}
      <div class="placeholder error" role="alert">
        <p>Couldn&rsquo;t load the model.</p>
        <button type="button" onclick={load}>Retry</button>
      </div>
    {:else if !started}
      <div class="placeholder">Type a prompt and press Start.</div>
    {:else}
      <div class="chips" role="group" aria-label="Prompt and generated tokens">
        {#each ids as id, i (i)}
          <span class="chip" class:generated={i >= promptLen}>
            {chipDisplay(encoder ? encoder.tokenText(id) : '')}
          </span>
        {/each}
      </div>

      {#if topK.length}
        <div class="bars" role="group" aria-label="Top {TOP_K} next-token probabilities">
          {#each topK as item (item.id)}
            <div class="bar-row">
              <span class="bar-label">{tokLabel(item.id)}</span>
              <span class="bar-track">
                <span
                  class="bar-fill"
                  class:sampled={item.id === sampledId}
                  style="width: {Math.min(100, item.p * 100)}%"
                ></span>
              </span>
              <span class="bar-pct">{(item.p * 100).toFixed(1)}%</span>
            </div>
          {/each}
        </div>
      {/if}

      {#if contextFull}
        <p class="note">Context full — 64 tokens is this toy model's whole world.</p>
      {:else if autoDone}
        <p class="note">Stopped after {MAX_GENERATED} generated tokens.</p>
      {/if}
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>
      prompt
      <input
        type="text"
        maxlength="80"
        bind:value={prompt}
        disabled={status !== 'ready' || autoOn}
        placeholder="Type a prompt"
      />
    </label>
    <button type="button" onclick={start} disabled={status !== 'ready' || autoOn}>Start</button>
    <button type="button" onclick={nextToken} disabled={!canStep || autoOn}>Next token</button>
    <button
      type="button"
      class:active={autoOn}
      aria-pressed={autoOn}
      onclick={toggleAuto}
      disabled={!canStep && !autoOn}
    >
      {autoOn ? 'Pause' : 'Auto'}
    </button>
    <label class="temp">
      temperature: {temperature.toFixed(2)}
      <input type="range" min="0.1" max="2" step="0.05" bind:value={temperature} />
      <span class="hint">cold = greedy &amp; repetitive &middot; hot = adventurous &amp; unhinged</span>
    </label>
    <button type="button" onclick={reset}>Reset</button>
  {/snippet}
</FigureShell>

<style>
  .placeholder {
    min-height: 460px;
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
    margin-bottom: 1rem;
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
  }
  .chip.generated {
    border-color: var(--accent-red);
    color: var(--accent-red);
    background: var(--paper-raised);
  }
  .bars {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    max-width: 420px;
    margin: 0 auto;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 4.5rem 1fr 3.2rem;
    align-items: center;
    gap: 0.5rem;
  }
  .bar-label {
    font-family: var(--font-code);
    font-size: 0.8rem;
    color: var(--ink-strong);
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
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
  }
  .bar-fill.sampled {
    background: var(--accent-red);
  }
  .bar-pct {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    color: var(--muted);
    text-align: right;
  }
  .note {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    text-align: center;
    color: var(--faint);
    margin: 0.75rem 0 0;
  }
  .temp {
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.2rem;
  }
  .temp .hint {
    font-size: 0.72rem;
    color: var(--faint);
  }
  button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
</style>
