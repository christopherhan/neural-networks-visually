<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    caption?: string;
    children: Snippet;
    controls?: Snippet;
  }

  let { title, caption = '', children, controls }: Props = $props();
</script>

<figure class="fig sketch">
  <figcaption class="fig-title label">{title}</figcaption>
  <div class="fig-body">{@render children()}</div>
  {#if controls}
    <div class="fig-controls">{@render controls()}</div>
  {/if}
  {#if caption}
    <p class="fig-caption">{caption}</p>
  {/if}
</figure>

<style>
  .fig {
    margin: 2rem 0;
    padding: 1.25rem;
  }
  .fig-title {
    margin-bottom: 0.75rem;
  }
  .fig-body :global(svg) {
    width: 100%;
    height: auto;
    display: block;
  }
  .fig-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem 1.25rem;
    margin-top: 1rem;
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--muted);
  }
  .fig-controls :global(label) {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .fig-controls :global(input[type='range']) {
    accent-color: var(--accent-blue);
    width: 110px;
  }
  .fig-controls :global(button) {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    padding: 0.3rem 0.8rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .fig-controls :global(button:hover) {
    border-color: var(--ink-strong);
  }
  .fig-caption {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0.75rem 0 0;
    text-align: center;
  }
</style>
