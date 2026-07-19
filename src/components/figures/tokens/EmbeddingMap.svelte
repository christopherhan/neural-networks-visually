<script lang="ts">
  import FigureShell from '../FigureShell.svelte';
  import { linearScale } from '../../../lib/viz/scale';
  import { url } from '../../../lib/site/url';

  type Status = 'loading' | 'error' | 'ready';

  interface EmbMap {
    tokens: string[];
    xy: [number, number][];
  }

  let status = $state<Status>('loading');
  let data = $state<EmbMap | null>(null);
  let hovered = $state<string | null>(null);
  let query = $state('');

  async function load() {
    status = 'loading';
    try {
      const res = await fetch(url('models/emb2d.json'));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = (await res.json()) as EmbMap;
      status = 'ready';
    } catch {
      status = 'error';
    }
  }

  load();

  const sx = linearScale([-1.1, 1.1], [12, 328]);
  const sy = linearScale([-1.1, 1.1], [328, 12]);

  const q = $derived(query.trim().toLowerCase());
  const matches = $derived(
    q && data ? data.tokens.map((t, i) => (t.trim().toLowerCase().includes(q) ? i : -1)).filter((i) => i >= 0) : []
  );
  const firstMatchToken = $derived(matches.length && data ? data.tokens[matches[0]] : null);
  const readoutToken = $derived(hovered ?? firstMatchToken);
  const matchSet = $derived(new Set(matches));

  function displayTok(t: string): string {
    return t === '\n' ? '⏎' : t.startsWith(' ') ? '␣' + t.slice(1) : t;
  }
</script>

<FigureShell
  title="The embedding space, projected to 2D"
  caption="This arrangement is the model's own — trained purely by next-token prediction, not by any human notion of similarity."
>
  {#snippet children()}
    {#if status === 'loading'}
      <div class="placeholder" role="status">loading embeddings&hellip;</div>
    {:else if status === 'error'}
      <div class="placeholder error" role="alert">
        <p>Couldn&rsquo;t load the embedding map.</p>
        <button type="button" onclick={load}>Retry</button>
      </div>
    {:else if data}
      <svg
        viewBox="0 0 340 340"
        class="scatter"
        aria-label="Scatter plot of 250 token embeddings projected to two dimensions"
      >
        {#each data.tokens as tok, i (i)}
          {@const isMatch = matchSet.has(i)}
          <circle
            cx={sx(data.xy[i][0])}
            cy={sy(data.xy[i][1])}
            r={isMatch ? 5 : 3.5}
            fill={isMatch ? 'var(--accent-red)' : 'var(--accent-blue)'}
            opacity={isMatch ? 0.85 : 0.6}
            tabindex="0"
            role="img"
            aria-label={tok}
            onmouseenter={() => (hovered = tok)}
            onfocus={() => (hovered = tok)}
          />
        {/each}
      </svg>

      <p class="readout" aria-live="polite">
        {#if readoutToken !== null && readoutToken !== undefined}
          token: <strong class="tok">{displayTok(readoutToken)}</strong>
        {:else}
          hover, focus, or search a point to see its token
        {/if}
      </p>
    {/if}
  {/snippet}

  {#snippet controls()}
    <label>
      search tokens
      <input type="text" bind:value={query} disabled={status !== 'ready'} placeholder="e.g. said" />
    </label>
  {/snippet}
</FigureShell>

<style>
  .placeholder {
    min-height: 340px;
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
  .scatter {
    display: block;
    width: 100%;
    max-width: 400px;
    height: auto;
    margin: 0 auto;
    background: var(--paper);
    border-radius: 8px;
  }
  .scatter circle {
    cursor: pointer;
  }
  .scatter circle:focus-visible {
    outline: 2px solid var(--accent-gold);
    outline-offset: 1px;
  }
  .readout {
    font-family: var(--font-ui);
    font-size: 0.9rem;
    text-align: center;
    color: var(--muted);
    margin: 0.75rem 0 0;
    min-height: 1.4em;
  }
  .readout .tok {
    font-family: var(--font-code);
    color: var(--ink-strong);
  }
</style>
