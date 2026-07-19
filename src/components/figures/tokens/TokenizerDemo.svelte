<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { createEncoder, type Tokenizer } from '../../../lib/nn/bpe';
  import { url } from '../../../lib/site/url';

  type Status = 'loading' | 'error' | 'ready';

  let status = $state<Status>('loading');
  let encoder = $state<ReturnType<typeof createEncoder> | null>(null);
  let text = $state('Attention is all you need!');

  async function load() {
    status = 'loading';
    try {
      const res = await fetch(url('models/tokenizer.json'));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as Tokenizer;
      encoder = createEncoder(data);
      status = 'ready';
    } catch {
      status = 'error';
    }
  }

  load();

  // Cycled chip tints — translucent so overlapping punctuation reads clean.
  const ACCENTS = [
    'rgba(38, 139, 210, 0.22)', // blue
    'rgba(181, 137, 0, 0.22)', // gold
    'rgba(133, 153, 0, 0.22)', // green
    'rgba(220, 50, 47, 0.22)', // red
  ];

  const tokenIds = $derived(encoder ? encoder.encode(text) : []);
  const chips = $derived(tokenIds.map((id) => encoder!.tokenText(id)));

  function display(tok: string): string {
    return tok.startsWith(' ') ? '␣' + tok.slice(1) : tok;
  }
</script>

<FigureShell
  title="Byte-pair tokenizer, live"
  caption="Rare words shatter into pieces; common words ride whole. Try &ldquo;extraordinarily&rdquo; next to &ldquo;the&rdquo; and watch the chip count."
>
  {#snippet children()}
    {#if status === 'loading'}
      <div class="placeholder" role="status">loading tokenizer&hellip;</div>
    {:else if status === 'error'}
      <div class="placeholder error" role="alert">
        <p>Couldn&rsquo;t load the tokenizer.</p>
        <button type="button" onclick={load}>Retry</button>
      </div>
    {:else}
      <div class="chips" aria-label="Tokenized text, one chip per token">
        {#each chips as chip, i (i)}
          <span class="chip" style:background={ACCENTS[i % ACCENTS.length]}>{display(chip)}</span>
        {/each}
      </div>
      <p class="readout">
        <strong>{text.length}</strong> characters &rarr; <strong>{tokenIds.length}</strong> tokens
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>
      text
      <input
        type="text"
        maxlength="120"
        bind:value={text}
        disabled={status !== 'ready'}
        placeholder="Type something to tokenize"
      />
    </label>
  {/snippet}
</FigureShell>

<style>
  .placeholder {
    min-height: 96px;
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
    gap: 0.3rem;
    line-height: 1.6;
    min-height: 96px;
  }
  .chip {
    font-family: var(--font-code);
    font-size: 0.85rem;
    padding: 0.15rem 0.4rem;
    border-radius: 5px;
    white-space: pre;
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.75rem 0 0;
  }
</style>
