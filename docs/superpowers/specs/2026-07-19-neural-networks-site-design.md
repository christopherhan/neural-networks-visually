# Neural Networks Visual Learning Site — Design Spec

**Date:** 2026-07-19
**Status:** Approved by Chris (brainstorming session)

## Purpose

A public, polished static website that teaches neural networks to technical people (software and cloud engineers) through heavy use of visuals, interactive diagrams, and animations. The arc runs from a single neuron all the way to frontier architectures (Kimi K3-class MoE models, Fable-class systems). Success = another engineer lands on the site and genuinely understands each concept by playing with it, not just reading about it.

## Audience & pedagogy

- **Audience:** engineers comfortable with code and systems, not assumed to know ML or calculus beyond intuition.
- **Math depth:** intuition-first. Visuals and analogies carry every explanation; equations live in expandable "math aside" blocks for readers who want rigor. Nothing in the main flow requires reading an equation.
- **Engineer framing:** each chapter ends with a "For engineers" footer restating the idea in code/systems terms (e.g., backprop as reverse-order graph traversal; MoE routing as load balancing), plus further-reading links.

## Visual & UX direction

- **Style: warm field notes** — warm paper tones (Solarized-light-adjacent palette: `#fdf6e3` paper, `#586e75` ink, accent colors `#268bd2` / `#dc322f` / `#859900` / `#b58900`), serif prose (Georgia stack), sans-serif UI accents, hand-drawn/sketch feel (dashed borders, rounded corners, Excalidraw/3Blue1Brown spirit).
- **Layout: course map + essay chapters** — the landing page is an illustrated "trail map" of the whole journey with numbered stops; each chapter is a focused single-column scrollytelling essay with a top progress bar and prev/next navigation. No persistent sidebar.
- **Scrollytelling:** prose steps trigger figure state changes as the reader scrolls; every scroll-driven figure is also directly manipulable (controls work independently of scroll).

## Curriculum — 4 parts, 18 chapters

### Part I — Foundations
1. **The Neuron** — from `if` statements to weighted sums. Covers the perceptron (weighted sum + hard threshold) as the on-ramp: build one interactively, adjust weights to separate points, then hit the XOR wall — the plot hook motivating activations (ch. 2) and depth (ch. 6).
2. **Activation Functions** — why nonlinearity matters; bend a network's output live.
3. **Loss & Gradient Descent** — roll a ball down the loss surface; scrub the learning rate.
4. **Backpropagation** — blame flows backward; step through the chain rule on a real computation graph. Shows the site's own autograd source.
5. **Training Playground** (flagship) — train a real MLP in-browser on toy datasets (spirals, moons, circles); watch the decision boundary form live.

### Part II — Classic architectures
6. **Going Deep: MLPs** — depth vs width, vanishing gradients.
7. **CNNs** — drag a kernel across an image; watch feature maps light up.
8. **RNNs & LSTMs** — unroll a loop through time; why long memories are hard.

### Part III — Transformers & LLMs
9. **Tokens & Embeddings** — type text, see it tokenize; words as points in space.
10. **Attention** (flagship) — run real attention over the reader's own sentence using the tiny pretrained model.
11. **Multi-Head Attention & the Transformer Block** — assemble the block piece by piece.
12. **Positional Encoding** — order injected as waves.
13. **Next-Token Prediction** — watch the tiny model generate, with live probability bars.
14. **Scaling Laws & Training Pipelines** — pretraining → SFT → RLHF in cloud-engineer terms (GPUs, data, checkpoints).

### Part IV — Frontier architectures
15. **Mixture-of-Experts** — router in action: tokens dispatched to experts; sparse vs dense compute.
16. **Attention at Scale** — KV caches, GQA, MLA (DeepSeek/Kimi-style latent attention), long context.
17. **Anatomy of a Frontier Model** — annotated, explorable block diagram of a K3-class architecture.
18. **Post-Training & Reasoning** — RLHF, reasoning tokens, what "thinking" models actually do.

## Technical architecture

**Stack:** Astro (fully static output) + MDX chapter content + Svelte 5 islands for interactive figures + TypeScript throughout. No CSS framework — small hand-written design system using CSS custom properties. KaTeX rendered at build time (zero client JS for math) inside collapsible aside blocks.

**Project structure:**

```
src/
  content/chapters/        # 18 MDX files (frontmatter: part, number, title, summary)
  components/
    figures/               # Svelte islands, one folder per figure
    layout/                # ChapterShell, TrailMap, PartDivider, MathAside, EngineerFooter
  lib/
    nn/                    # micro-ML engine: scalar autograd, layers, optimizers (~300-400 lines)
    viz/                   # shared SVG/Canvas helpers: scales, axes, animation loop, ScrollStage
  pages/                   # index (trail map), chapters/[slug]
tools/
  train-tiny-transformer/  # Python; trains the attention-demo model offline, exports weights JSON
public/models/             # exported weight files (~1-2 MB total budget)
```

**Key decisions:**

- **Figures are lazy Svelte islands** (`client:visible`): chapter pages load as pure static HTML/CSS; each figure hydrates only when scrolled near.
- **Micro-ML engine, hand-rolled:** scalar autograd (micrograd-style) powers all live-training figures. Chosen over TensorFlow.js deliberately — small bundles, fully inspectable, and the engine's source *is* teaching material.
- **Tiny transformer, trained offline:** a small model (character/BPE-lite tokenizer, ~2 layers, few heads) trained by the Python tool; weights shipped as static JSON; inference re-implemented in TS. All Part III demos (attention, embeddings, generation) share this one model.
- **ScrollStage:** small IntersectionObserver-based helper for scrollytelling; no third-party scrollytelling dependency.
- **Rendering:** SVG for diagrams/small figures; Canvas for dense ones (decision boundaries, feature maps).

## Figure interaction grammar

Consistent across all figures:

- **Play/step controls** on anything animated: play, pause, single-step, reset.
- **Scrub anything scrubbable:** sliders for weights/learning rate/kernel values; direct drag on data points.
- **Hover = inspect:** hovering any neuron, edge, or token shows its current value — debugger-tooltip feel.
- **Reduced motion:** `prefers-reduced-motion` swaps animations for stepped stills.

## Error handling & resilience

- Every island renders a static SVG snapshot of its initial state as fallback content — with JS disabled or before hydration, chapters still read as illustrated essays.
- Training demos cap iteration counts and guard NaN/divergence; a diverging loss is surfaced as a teaching moment ("your learning rate is too high — here's what that looks like"), not a crash.
- Model-weight fetches show a lightweight loading state and a retry affordance.

## Testing

- **Vitest unit tests for `lib/nn`:** gradient checks against numerical differentiation; tokenizer and inference parity tests against reference outputs exported by the Python trainer.
- **Component smoke tests** for figures (render + basic interaction).
- **CI gate:** Astro build must succeed; link check over the built site.

## Deployment

GitHub Pages via GitHub Actions on push to `main`. Fully static; no server-side anything. No host-specific features used beyond Pages base-path config.

## Build order (each stage ships a deployable site)

1. **Skeleton + design system** — Astro project, trail-map landing page, chapter shell, typography/palette, CI + Pages deploy working end to end.
2. **Part I** — micro-ML engine + chapters 1–5 (Training Playground flagship).
3. **Part II** — chapters 6–8 (CNN kernel demo is the centerpiece).
4. **Part III** — Python tiny-transformer trainer + chapters 9–14 (Attention flagship).
5. **Part IV** — chapters 15–18 (MoE router demo, K3 anatomy diagram).

Each stage gets its own implementation plan; this spec governs the whole site.

## Out of scope

- User accounts, progress persistence beyond `localStorage`, comments, analytics beyond a privacy-friendly counter (optional, later).
- Mobile-first parity for complex figures: chapters must *read* well on mobile; the heaviest interactive figures may show simplified variants on small screens.
- Training anything non-toy in the browser; all "real" models are trained offline.
