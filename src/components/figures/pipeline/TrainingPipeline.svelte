<script lang="ts">
  import FigureShell from '../FigureShell.svelte';

  // No model here — this figure is a labeled pipeline diagram. The SVG is
  // decorative (a picture of the five stages); the real interactive
  // elements are the buttons below it, one per stage, each a proper
  // <button> with aria-pressed — the accessible-simplest option per the
  // house rule (real buttons beneath a decorative diagram, rather than
  // clickable SVG groups wired up with tabindex/role/aria by hand).

  interface Stage {
    id: string;
    label: string;
    fact: string;
  }

  const STAGES: Stage[] = [
    {
      id: 'data',
      label: 'Data',
      fact: "Trillions of tokens scraped, filtered, deduplicated, and mixed — web text, code, books, licensed corpora. Data engineering is the least glamorous and most decisive stage: quality filters and mixture ratios move final capability more than most architecture tweaks. (Our toy's version: three Gutenberg novels, 547 KB.)",
    },
    {
      id: 'pretraining',
      label: 'Pretraining',
      fact: 'The next-token loop from chapter 13, run on thousands of GPUs for months. Frontier runs are measured in units of 10²⁵–10²⁶ FLOPs. The job is a distributed-systems marathon: data/tensor/pipeline parallelism, checkpoint-and-restart on hardware failure, loss-spike triage at 3am. Output: a base model — a magnificent autocomplete.',
    },
    {
      id: 'sft',
      label: 'SFT',
      fact: 'Supervised fine-tuning: thousands to millions of curated (prompt → good response) pairs teach the base model to behave like an assistant instead of finishing your sentence. Same training loop, tiny data, huge behavioral shift.',
    },
    {
      id: 'rlhf',
      label: 'RLHF',
      fact: "Humans (and increasingly AI assistants — RLAIF/constitutional methods) rank candidate responses; a reward model learns those preferences; the policy is optimized against it. This is where 'helpful, honest, harmless' gets shaped — and where reasoning training now teaches models to spend tokens thinking before answering (chapter 18).",
    },
    {
      id: 'deployment',
      label: 'Deployment',
      fact: "Quantize, distill, cache, and serve. Inference at scale is its own discipline: KV caches (chapter 16), batching schedulers, speculative decoding. Fun fact: over a model's life, inference compute usually dwarfs training compute.",
    },
  ];
  const TOTAL = STAGES.length;

  let selected = $state(1); // default-select Pretraining

  const current = $derived(STAGES[selected]);

  function select(i: number) {
    selected = i;
  }

  // ---- Diagram geometry ----
  const VIEW_W = 700;
  const VIEW_H = 150;
  const BOX_W = 110;
  const BOX_H = 56;
  const GAP = (VIEW_W - TOTAL * BOX_W) / (TOTAL - 1);
  const BOX_Y = (VIEW_H - BOX_H) / 2;

  const boxes = STAGES.map((s, i) => ({
    ...s,
    x: i * (BOX_W + GAP),
  }));
</script>

<FigureShell
  title="From tokens to assistant: the training pipeline"
  caption="Five stages, same shape every frontier lab runs. Select a stage for the details."
>
  {#snippet children()}
    <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} aria-hidden="true" focusable="false">
      <defs>
        <marker
          id="pipeline-arrow"
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

      {#each boxes as box, i}
        {#if i < TOTAL - 1}
          <line
            x1={box.x + BOX_W}
            y1={VIEW_H / 2}
            x2={box.x + BOX_W + GAP}
            y2={VIEW_H / 2}
            stroke="var(--line)"
            stroke-width="1.5"
            marker-end="url(#pipeline-arrow)"
          />
        {/if}
      {/each}

      {#each boxes as box, i}
        <g>
          <rect
            x={box.x}
            y={BOX_Y}
            width={BOX_W}
            height={BOX_H}
            rx="10"
            fill={selected === i ? 'var(--paper-raised)' : 'var(--paper)'}
            stroke={selected === i ? 'var(--accent-red)' : 'var(--line)'}
            stroke-width={selected === i ? 3 : 1.5}
          />
          <text x={box.x + BOX_W / 2} y={VIEW_H / 2 + 5} text-anchor="middle" class="box-label">
            {box.label}
          </text>
        </g>
      {/each}
    </svg>

    <div class="stage-buttons" role="group" aria-label="Pipeline stage">
      {#each boxes as box, i (box.id)}
        <button
          type="button"
          class:active={selected === i}
          aria-pressed={selected === i}
          onclick={() => select(i)}
        >
          {box.label}
        </button>
      {/each}
    </div>

    <p class="fact">
      <span class="phase label">{current.label}</span> {current.fact}
    </p>
  {/snippet}
</FigureShell>

<style>
  .box-label {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 700;
    fill: var(--ink-strong);
  }
  .stage-buttons {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .stage-buttons button {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    padding: 0.35rem 0.9rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .stage-buttons button:hover {
    border-color: var(--ink-strong);
  }
  .stage-buttons button.active {
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
</style>
