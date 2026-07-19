# Stage 4: Part III Chapters 9–14 Implementation Plan

> **Execution mode (per user direction):** implementer subagent per task, coordinator verifies inline. The Python trainer (Task 0) is coordinator-implemented and already running. This plan specifies behavior contracts + key algorithms at full precision and delegates figure internals to implementers following the established FigureShell/house-rules pattern (Stages 2a–3 addenda) — a deliberate altitude choice for throughput, with coordinator inline verification as the gate.

**Goal:** Part III — the transformer chapters: tokens & embeddings (9), the Attention flagship (10), the transformer block (11), positional encoding (12), next-token generation (13), scaling & training pipelines (14) — powered by a real tiny transformer trained offline in Python and re-implemented for inference in TypeScript.

**Architecture:** `tools/train-tiny-transformer/` (done: BPE at 383 tokens over a 547KB Gutenberg corpus; 2-layer / 4-head / d64 / ctx64 GPT-style LM with tied embeddings, numpy, grad-checked at 7.7e-09). Artifacts ship to `public/models/`: `tokenizer.json`, `weights.json` (~1.5MB), `emb2d.json`, `ref_logits.json` (+`tokenizer_ref.json`). TS side: `src/lib/nn/bpe.ts` (encoder mirroring the Python rule exactly) and `src/lib/nn/transformer.ts` (pure-matrix inference: forward with per-head attention capture, sampling) — both parity-tested against the Python references. Figures fetch model JSON lazily on hydrate (loading state + retry per spec).

**Model/data contracts (exact):**

- `tokenizer.json = {vocab: string[], merges: [string,string][]}`. Encode rule: split text with regex `/ ?[A-Za-z']+| ?[.,;:!?"()\-]|\n/g`, per word repeatedly merge the adjacent pair with lowest merge rank until none apply, map symbols→ids, skip unknown symbols. Decode = concat vocab strings.
- `weights.json = {config: {vocab_size, d_model, n_head, n_layer, ctx, mlp}, tok_emb: number[][], pos_emb: number[][], lnf_g: number[], lnf_b: number[], layers: [{ln1_g,ln1_b,wq,wk,wv,wo,ln2_g,ln2_b,w1,b1,w2,b2}]}`. Forward (pre-LN GPT): `x = tok_emb[ids] + pos_emb[:T]`; per layer `x += attn(ln1(x))` then `x += mlp(ln2(x))` with causal softmax(QKᵀ/√(d/h)) attention (weights layout: `q = x·wq` row-vector convention, heads split contiguously) and ReLU MLP; `logits = lnf(x)·tok_embᵀ` (tied). LN eps 1e-5.
- `ref_logits.json = {ids: number[], last_logits_first32: number[]}` — TS forward on `ids` must reproduce these within 2e-3 abs tolerance (weights are rounded to 5dp).
- `tokenizer_ref.json = [{text, ids}]` — TS encoder must match ids exactly.
- `emb2d.json = {tokens: string[], xy: [number,number][]}` (250 most frequent tokens, PCA, normalized to [-1,1]).

**Figure inventory (all: Svelte 5 island, `client:visible`, FigureShell, house rules; model-backed figures load JSON via `fetch(url('models/...'))` with a "loading model…" state and a retry button on failure; SSR fallback = the loading state):**

| ch | component | behavior contract |
|---|---|---|
| 9 | `tokens/TokenizerDemo.svelte` | Text input (default "Attention is all you need!", maxlength 120). Live BPE segmentation as colored chips (cycle 4 accent colors; leading-space shown as ␣). Readout: N chars → M tokens. Uses bpe.ts + tokenizer.json. |
| 9 | `tokens/EmbeddingMap.svelte` | Scatter of emb2d.json in an SVG (linearScale both axes). Hover/focus a point shows the token string (points are focusable circles with aria-labels — house rule: non-redundant info must be reachable). Search box highlights matching tokens. Caption notes clusters are the model's own arrangement. |
| 10 | `attention/AttentionLens.svelte` | FLAGSHIP. Text input (default "The Queen turned angrily to Alice and said hold your tongue"), Analyze button → tokenize (clip to 32 tokens), forward pass with attention capture. Layer picker (1/2), head picker (1–4 + "average"). Heatmap T×T (rows=queries, cols=keys, opacity=weight, causal lower-triangle). Click/focus a row token → top-3 attended tokens listed in a readout with weights as %. Honest note that this tiny model's heads are fuzzier than GPT-4's but the mechanism is identical. |
| 11 | `block/BlockAssembler.svelte` | No model. Stepper (reuse BackpropStepper's step/play/reset pattern) revealing the block: embeddings → +positions → LN → Q/K/V attention → +residual → LN → MLP → +residual → unembed, one explanation per step (exact strings in Task 5), SVG flow diagram with the current stage highlighted. ~10 steps. |
| 12 | `positional/PositionalWaves.svelte` | No model. Closed-form sin/cos PE heatmap (pos 0–63 × dim 0–63, computed in TS: `sin(pos/10000^(2i/d))` pairs). Slider selects a position → that row's wave values drawn as a curve; second slider selects another position → dot-product similarity readout showing nearby positions are more similar. |
| 13 | `generation/GenerationLab.svelte` | Prompt box (default "Alice went"), Generate-next-token button + Auto (ticker ~2 tok/s, max 40 tokens) + Reset. Each step: forward, softmax with temperature slider (0.1–2, default 0.8), show top-8 probability bars (token label + %), sample via seeded mulberry32, append chip to the output. Divergence-free by construction; cap at ctx 64 with a "context full" note. |
| 14 | `pipeline/TrainingPipeline.svelte` | No model. Clickable 5-stage pipeline SVG (Data → Pretraining → SFT → RLHF/RLAIF → Deployment); clicking a stage shows a fact panel (exact copy in Task 8) with cloud-engineer framing (GPU-hours, data scale, checkpoints, eval gates). Buttons are real `<button>`s with aria-pressed. |

**Chapter files (frontmatter must match curriculum.ts):** 9 `tokens-and-embeddings` "Tokens & Embeddings" · 10 `attention` "Attention" · 11 `the-transformer-block` "Multi-Head Attention & the Transformer Block" · 12 `positional-encoding` "Positional Encoding" · 13 `next-token-prediction` "Next-Token Prediction" · 14 `scaling-and-training-pipelines` "Scaling Laws & Training Pipelines". Each chapter: intuition-first prose in the established voice, MathAside, EngineerFooter, figure(s) via `client:visible`.

**Task list:**

- Task 0 (coordinator, running): train model, copy artifacts to `public/models/`, commit tool + artifacts.
- Task 1: `src/lib/nn/bpe.ts` + parity tests (fixtures imported from `tools/train-tiny-transformer/tokenizer.json` + `tokenizer_ref.json` via Vitest JSON imports; encode matches ids exactly; decode round-trips).
- Task 2: `src/lib/nn/transformer.ts` + parity tests (`loadModel(json)`, `forward(ids) → {logits: number[][], attention: number[layer][head][T][T]}`, `sampleNext(logits, temperature, rng)`; parity vs ref_logits within 2e-3; softmax/temperature unit tests; causality test: attention row t has zero weight beyond t).
- Task 3: Chapter 9 (TokenizerDemo + EmbeddingMap + MDX).
- Task 4: Chapter 10 (AttentionLens + MDX) — flagship.
- Task 5: Chapter 11 (BlockAssembler + MDX).
- Task 6: Chapter 12 (PositionalWaves + MDX).
- Task 7: Chapter 13 (GenerationLab + MDX).
- Task 8: Chapter 14 (TrainingPipeline + MDX).
- Task 9: Integration (suite, links chain 8→9→…→14, trail map 14 links, base-path guard incl. fetched model URLs base-prefixed via `url()` helper), merge, deploy, live-verify.

**House rules riders for this stage:** model JSON fetches MUST use the `url()` helper for base-path safety; figures must handle fetch failure with a visible retry (spec: "loading state and a retry affordance"); no figure may block chapter reading while loading (fixed-height placeholder).
