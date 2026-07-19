<script lang="ts">
  import { onDestroy } from 'svelte';
  import FigureShell from '../FigureShell.svelte';
  import { createTicker } from '../../../lib/viz/ticker';

  // No model here — this is a labeled loop diagram, stepped through with the
  // established step/play/reset pattern (BackpropStepper, KVCalculator's
  // mini-stepper). Two modes share one mechanism: sample from a policy,
  // score the samples, nudge the policy toward the higher-scoring ones,
  // repeat. What differs is where the score comes from — a learned reward
  // model (preference loop / RLHF) or a checkable ground truth (reasoning
  // training / RL on verifiable rewards).

  type Mode = 'preference' | 'reasoning';

  interface Stage {
    id: string;
    label: string;
    text: string;
  }

  const PREFERENCE_STAGES: Stage[] = [
    {
      id: 'generate',
      label: 'Generate',
      text: 'The current policy — the assistant model itself — samples several candidate answers to the same prompt, the exact sampling loop chapter 13 built, just run more than once so there is something to compare.',
    },
    {
      id: 'preferences',
      label: 'Preferences',
      text: 'A rater — a human, or increasingly another model checking candidates against a written constitution — picks the better answer, not a perfect one. That comparative judgment, not any absolute score, is the whole signal this loop runs on.',
    },
    {
      id: 'reward-model',
      label: 'Reward model',
      text: "A separate model is trained to predict which candidate a rater would prefer, turning thousands of pairwise judgments into a fast, differentiable stand-in for taste. It is a classifier wearing a scorer's hat — only as honest as the comparisons it learned from.",
    },
    {
      id: 'optimize',
      label: 'Optimize',
      text: "RL nudges the policy toward answers the reward model scores higher — chapter 3's \"follow the gradient downhill,\" except the loss now comes from a learned proxy for approval. Push that nudge too hard for too long and the policy learns to please the proxy instead of the rater it approximated — reward hacking.",
    },
    {
      id: 'repeat',
      label: 'Repeat',
      text: 'The improved policy generates the next round of candidates and the loop runs again — sharper preferences, a better-calibrated reward model, another optimization pass. It is a loop, not a pipeline stage, and it keeps tightening as long as it keeps running.',
    },
  ];

  const REASONING_STAGES: Stage[] = [
    {
      id: 'problem',
      label: 'Problem',
      text: 'The prompt this time is a math problem or coding task with one checkable property: does the final answer come out right? No rater opinion needed — a unit test or an answer key settles it outright.',
    },
    {
      id: 'scratchpad',
      label: 'Scratchpad',
      text: 'Before committing to an answer, the model samples a long chain of intermediate tokens — a scratchpad — working the problem the way generation always worked in chapter 13, just handed far more tokens to spend before it has to commit.',
    },
    {
      id: 'check',
      label: 'Check',
      text: "The final answer gets checked against ground truth — did the code pass its tests, does the number match — and that pass/fail verdict is the entire reward, computed by a script instead of a rater's judgment.",
    },
    {
      id: 'reinforce',
      label: 'Reinforce',
      text: "RL reinforces whatever scratchpad habits preceded checked-out answers and suppresses whatever preceded wrong ones — nobody labeled a single line of it \"good reasoning.\" \"Thinking\" here just means sampled tokens that the reward made genuinely useful.",
    },
  ];

  const MODE_META: Record<Mode, { title: string; caption: string; center: string }> = {
    preference: {
      title: 'The preference loop (RLHF)',
      caption:
        'Step through the loop that turns a raw next-token predictor into something shaped by comparative judgment — human, and increasingly AI, raters.',
      center: 'RLHF',
    },
    reasoning: {
      title: 'Reasoning training (RL on verifiable rewards)',
      caption:
        'Same loop, a different reward source: no rater at all, just a checkable answer deciding what gets reinforced.',
      center: 'RLVR',
    },
  };

  let mode = $state<Mode>('preference');
  let stepIndex = $state(0);
  let playing = $state(false);

  const stages = $derived(mode === 'preference' ? PREFERENCE_STAGES : REASONING_STAGES);
  const TOTAL = $derived(stages.length);
  const meta = $derived(MODE_META[mode]);

  const phase = $derived(stepIndex === 0 ? 'ready' : stages[stepIndex - 1].id);
  const current = $derived(stepIndex === 0 ? null : stages[stepIndex - 1]);

  function advance() {
    if (stepIndex < TOTAL) stepIndex += 1;
    if (stepIndex >= TOTAL) stopPlay();
  }

  const ticker = createTicker(advance, { fps: 0.8 });
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

  function setMode(m: Mode) {
    if (mode === m) return;
    mode = m;
    stopPlay();
    stepIndex = 0;
  }

  // ---- Circular loop diagram: N nodes evenly spaced around a circle,
  // edges drawn as the full cycle (including the wrap-around edge back to
  // node 0) — the loop shape IS the "repeats" step, no special-casing
  // needed. Works unchanged for N=5 (preference) or N=4 (reasoning).
  const CX = 350;
  const CY = 178;
  const R = 138;
  const NODE_W = 132;
  const NODE_H = 58;
  const TRIM = 60; // approx node "radius" for line trimming, in px

  const nodes = $derived(
    stages.map((s, i) => {
      const theta = -Math.PI / 2 + i * ((2 * Math.PI) / stages.length);
      return { ...s, x: CX + R * Math.cos(theta), y: CY + R * Math.sin(theta) };
    })
  );

  interface Edge {
    id: string;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    active: boolean;
  }

  const edges = $derived(
    nodes.map((a, i): Edge => {
      const b = nodes[(i + 1) % nodes.length];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / dist;
      const uy = dy / dist;
      return {
        id: `${a.id}->${b.id}`,
        x1: a.x + ux * TRIM,
        y1: a.y + uy * TRIM,
        x2: b.x - ux * TRIM,
        y2: b.y - uy * TRIM,
        // the edge feeding INTO the current node is the transition just made
        active: current?.id === b.id,
      };
    })
  );
</script>

<FigureShell title={meta.title} caption={meta.caption}>
  {#snippet children()}
    <svg
      viewBox="0 0 700 380"
      aria-label={`Loop diagram of ${TOTAL} stages: ${stages.map((s) => s.label).join(' → ')} → back to the start`}
    >
      <defs>
        <marker
          id="alignment-arrow"
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
          id="alignment-arrow-active"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent-red)" />
        </marker>
      </defs>

      {#each edges as e (e.id)}
        <line
          x1={e.x1}
          y1={e.y1}
          x2={e.x2}
          y2={e.y2}
          stroke={e.active ? 'var(--accent-red)' : 'var(--line)'}
          stroke-width={e.active ? 2.5 : 1.5}
          marker-end={e.active ? 'url(#alignment-arrow-active)' : 'url(#alignment-arrow)'}
        />
      {/each}

      <text x={CX} y={CY - 6} text-anchor="middle" class="center-label">{meta.center}</text>
      <text x={CX} y={CY + 14} text-anchor="middle" class="center-sublabel">
        step {stepIndex} / {TOTAL}
      </text>

      {#each nodes as n, i (n.id)}
        {@const isCurrent = current?.id === n.id}
        <g>
          <rect
            x={n.x - NODE_W / 2}
            y={n.y - NODE_H / 2}
            width={NODE_W}
            height={NODE_H}
            rx="12"
            fill={isCurrent ? 'var(--paper-raised)' : 'var(--paper)'}
            stroke={isCurrent ? 'var(--accent-red)' : 'var(--line)'}
            stroke-width={isCurrent ? 3 : 1.5}
          />
          <text x={n.x} y={n.y - 6} text-anchor="middle" class="n-num">{i + 1}</text>
          <text x={n.x} y={n.y + 14} text-anchor="middle" class="n-label">{n.label}</text>
        </g>
      {/each}
    </svg>

    <p class="explain" data-phase={phase}>
      {#if current}
        <span class="phase label">{current.label}</span> {current.text}
      {:else}
        <span class="phase label">ready</span> Press Step (or Play) to walk the loop, stage by stage.
      {/if}
    </p>
  {/snippet}

  {#snippet controls()}
    <div class="mode-buttons" role="group" aria-label="Post-training mode">
      <button
        type="button"
        class:active={mode === 'preference'}
        aria-pressed={mode === 'preference'}
        onclick={() => setMode('preference')}
      >
        Preference loop
      </button>
      <button
        type="button"
        class:active={mode === 'reasoning'}
        aria-pressed={mode === 'reasoning'}
        onclick={() => setMode('reasoning')}
      >
        Reasoning training
      </button>
    </div>
    <button type="button" onclick={advance} disabled={stepIndex >= TOTAL}>Step</button>
    <button type="button" onclick={togglePlay}>{playing ? 'Pause' : 'Play'}</button>
    <button type="button" onclick={reset}>Reset</button>
    <span class="progress">{stepIndex} / {TOTAL}</span>
  {/snippet}
</FigureShell>

<style>
  .center-label {
    font-family: var(--font-ui);
    font-size: 15px;
    font-weight: 700;
    fill: var(--faint);
    letter-spacing: 0.05em;
  }
  .center-sublabel {
    font-family: var(--font-ui);
    font-size: 11px;
    fill: var(--faint);
  }
  .n-num {
    font-family: var(--font-code);
    font-size: 11px;
    fill: var(--muted);
  }
  .n-label {
    font-family: var(--font-ui);
    font-size: 13px;
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
  .explain[data-phase='optimize'],
  .explain[data-phase='reinforce'] {
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
  .mode-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-right: 0.5rem;
  }
  .mode-buttons button {
    font-family: var(--font-ui);
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    background: var(--paper);
    color: var(--ink-strong);
    border: 1.5px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
  }
  .mode-buttons button:hover {
    border-color: var(--ink-strong);
  }
  .mode-buttons button.active {
    border-color: var(--accent-blue);
    color: var(--link);
  }
</style>
