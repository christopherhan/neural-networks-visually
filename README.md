# Neural Networks, Visually

An illustrated, interactive journey from a single neuron to frontier AI
architectures (MoE, K3-class models) — built for software and cloud engineers.

- **Spec:** `docs/superpowers/specs/2026-07-19-neural-networks-site-design.md`
- **Stack:** Astro 5 + MDX + Svelte islands + hand-rolled micro-ML engine
- **Style:** warm field notes — paper tones, sketch borders, intuition-first prose

## Develop

    npm install
    npm run dev        # http://localhost:4321
    npm test           # Vitest unit tests
    npm run check      # astro typecheck
    npm run build      # static build to dist/

## Deploy

Pushes to `main` run tests, build, and deploy to GitHub Pages
(`.github/workflows/deploy.yml`). Repo Settings → Pages → Source must be
set to **GitHub Actions** (one-time setup).

## Content

Chapters are MDX in `src/content/chapters/`; the canonical 18-chapter
curriculum (order, parts, slugs) is `src/lib/site/curriculum.ts`. A chapter
appears on the trail map as soon as its MDX file exists with a slug matching
the curriculum.
