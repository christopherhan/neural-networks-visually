# Stage 1: Skeleton + Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deployable Astro site with the warm field-notes design system, trail-map landing page, chapter shell (progress bar, math asides, engineer footer, prev/next), one stub chapter proving the MDX pipeline, tests, and CI + GitHub Pages deployment.

**Architecture:** Astro 5 static site. Chapter content lives in MDX (content collection); the canonical 18-chapter curriculum lives in a typed TS module (`src/lib/site/curriculum.ts`) so the trail map can show unwritten chapters as "coming soon." Layout components are `.astro` (no client JS in Stage 1); Svelte integration is installed now so Stage 2 figures drop in without config churn. KaTeX renders at build time via remark-math/rehype-katex.

**Tech Stack:** Astro 5, MDX, Svelte 5 (integration only), TypeScript strict, Vitest, KaTeX, GitHub Actions + GitHub Pages (via `withastro/action`, which injects `site`/`base` for the repo automatically).

**Spec:** `docs/superpowers/specs/2026-07-19-neural-networks-site-design.md`

---

### Task 1: Astro project scaffold

**Files:**
- Create: `package.json`, `astro.config.mjs`, `tsconfig.json`, `src/pages/index.astro`
- Modify: `.gitignore`

- [ ] **Step 1: Write `package.json`**

```json
{
  "name": "neural-networks-site",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "test": "vitest run",
    "check": "astro check"
  },
  "dependencies": {
    "@astrojs/mdx": "^4.0.0",
    "@astrojs/svelte": "^7.0.0",
    "astro": "^5.0.0",
    "katex": "^0.16.11",
    "rehype-katex": "^7.0.1",
    "remark-math": "^6.0.0",
    "svelte": "^5.0.0"
  },
  "devDependencies": {
    "@astrojs/check": "^0.9.4",
    "linkinator": "^6.1.0",
    "typescript": "^5.6.0",
    "vitest": "^3.0.0"
  }
}
```

- [ ] **Step 2: Write `astro.config.mjs`**

```js
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import svelte from '@astrojs/svelte';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// site/base are injected at deploy time by withastro/action (GitHub Pages).
// Local dev and CI test builds use the default base "/".
export default defineConfig({
  integrations: [mdx(), svelte()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
```

- [ ] **Step 3: Write `tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "src"],
  "exclude": ["dist"]
}
```

- [ ] **Step 4: Write placeholder `src/pages/index.astro`** (replaced in Task 7)

```astro
---
---
<html lang="en">
  <head><meta charset="utf-8" /><title>Neural Networks, Visually</title></head>
  <body><h1>Neural Networks, Visually</h1></body>
</html>
```

- [ ] **Step 5: Append Astro artifacts to `.gitignore`**

Append these lines to the existing `.gitignore`:

```
.astro/
```

- [ ] **Step 6: Install and verify build**

Run: `npm install && npm run build`
Expected: install succeeds; build ends with `Complete!` and `dist/index.html` exists.

- [ ] **Step 7: Commit**

```bash
git add package.json package-lock.json astro.config.mjs tsconfig.json src/pages/index.astro .gitignore
git commit -m "feat: scaffold Astro project with MDX, Svelte, KaTeX integrations"
```

---

### Task 2: Vitest setup

**Files:**
- Create: `vitest.config.ts`, `src/lib/site/smoke.test.ts`

- [ ] **Step 1: Write `vitest.config.ts`**

```ts
/// <reference types="vitest" />
import { getViteConfig } from 'astro/config';

export default getViteConfig({
  test: {
    include: ['src/**/*.test.ts'],
  },
});
```

- [ ] **Step 2: Write `src/lib/site/smoke.test.ts`**

```ts
import { describe, it, expect } from 'vitest';

describe('vitest wiring', () => {
  it('runs', () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 3: Run tests**

Run: `npm test`
Expected: `1 passed` (smoke test).

- [ ] **Step 4: Commit**

```bash
git add vitest.config.ts src/lib/site/smoke.test.ts
git commit -m "test: add Vitest via astro getViteConfig with smoke test"
```

---

### Task 3: Curriculum data + helpers (TDD)

The canonical list of all 18 chapters. The MDX collection holds *written* content; this module is the source of truth for structure, ordering, and the trail map.

**Files:**
- Create: `src/lib/site/curriculum.ts`
- Test: `src/lib/site/curriculum.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/site/curriculum.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import {
  CURRICULUM,
  PARTS,
  chapterBySlug,
  chapterByNumber,
  partOf,
  prevNext,
} from './curriculum';

describe('curriculum', () => {
  it('has 18 chapters numbered 1..18 contiguously', () => {
    expect(CURRICULUM).toHaveLength(18);
    expect(CURRICULUM.map((c) => c.number)).toEqual(
      Array.from({ length: 18 }, (_, i) => i + 1)
    );
  });

  it('has unique slugs', () => {
    const slugs = CURRICULUM.map((c) => c.slug);
    expect(new Set(slugs).size).toBe(18);
  });

  it('has 4 parts and every chapter maps to a valid part', () => {
    expect(PARTS).toHaveLength(4);
    for (const c of CURRICULUM) {
      expect(PARTS.map((p) => p.number)).toContain(c.part);
    }
  });

  it('parts partition the chapters per the spec (5/3/6/4)', () => {
    const count = (part: number) =>
      CURRICULUM.filter((c) => c.part === part).length;
    expect([count(1), count(2), count(3), count(4)]).toEqual([5, 3, 6, 4]);
  });

  it('looks up chapters by slug and number', () => {
    expect(chapterBySlug('the-neuron')?.number).toBe(1);
    expect(chapterByNumber(10)?.slug).toBe('attention');
    expect(chapterBySlug('nope')).toBeUndefined();
    expect(chapterByNumber(99)).toBeUndefined();
  });

  it('partOf returns the part metadata for a chapter', () => {
    const ch15 = chapterByNumber(15)!;
    expect(partOf(ch15)).toEqual({
      number: 4,
      numeral: 'IV',
      title: 'Frontier Architectures',
    });
  });

  it('prevNext handles interior chapters and boundaries', () => {
    const mid = prevNext(10);
    expect(mid.prev?.number).toBe(9);
    expect(mid.next?.number).toBe(11);
    expect(prevNext(1).prev).toBeUndefined();
    expect(prevNext(18).next).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./curriculum`.

- [ ] **Step 3: Write `src/lib/site/curriculum.ts`**

```ts
export interface PartMeta {
  number: number;
  numeral: string;
  title: string;
}

export interface ChapterMeta {
  number: number;
  slug: string;
  title: string;
  part: number;
}

export const PARTS: PartMeta[] = [
  { number: 1, numeral: 'I', title: 'Foundations' },
  { number: 2, numeral: 'II', title: 'Classic Architectures' },
  { number: 3, numeral: 'III', title: 'Transformers & LLMs' },
  { number: 4, numeral: 'IV', title: 'Frontier Architectures' },
];

export const CURRICULUM: ChapterMeta[] = [
  { number: 1, slug: 'the-neuron', title: 'The Neuron', part: 1 },
  { number: 2, slug: 'activation-functions', title: 'Activation Functions', part: 1 },
  { number: 3, slug: 'loss-and-gradient-descent', title: 'Loss & Gradient Descent', part: 1 },
  { number: 4, slug: 'backpropagation', title: 'Backpropagation', part: 1 },
  { number: 5, slug: 'training-playground', title: 'The Training Playground', part: 1 },
  { number: 6, slug: 'going-deep-mlps', title: 'Going Deep: MLPs', part: 2 },
  { number: 7, slug: 'convolutional-networks', title: 'Convolutional Networks', part: 2 },
  { number: 8, slug: 'rnns-and-lstms', title: 'RNNs & LSTMs', part: 2 },
  { number: 9, slug: 'tokens-and-embeddings', title: 'Tokens & Embeddings', part: 3 },
  { number: 10, slug: 'attention', title: 'Attention', part: 3 },
  { number: 11, slug: 'the-transformer-block', title: 'Multi-Head Attention & the Transformer Block', part: 3 },
  { number: 12, slug: 'positional-encoding', title: 'Positional Encoding', part: 3 },
  { number: 13, slug: 'next-token-prediction', title: 'Next-Token Prediction', part: 3 },
  { number: 14, slug: 'scaling-and-training-pipelines', title: 'Scaling Laws & Training Pipelines', part: 3 },
  { number: 15, slug: 'mixture-of-experts', title: 'Mixture-of-Experts', part: 4 },
  { number: 16, slug: 'attention-at-scale', title: 'Attention at Scale', part: 4 },
  { number: 17, slug: 'anatomy-of-a-frontier-model', title: 'Anatomy of a Frontier Model', part: 4 },
  { number: 18, slug: 'post-training-and-reasoning', title: 'Post-Training & Reasoning', part: 4 },
];

export function chapterBySlug(slug: string): ChapterMeta | undefined {
  return CURRICULUM.find((c) => c.slug === slug);
}

export function chapterByNumber(n: number): ChapterMeta | undefined {
  return CURRICULUM.find((c) => c.number === n);
}

export function partOf(chapter: ChapterMeta): PartMeta {
  const part = PARTS.find((p) => p.number === chapter.part);
  if (!part) throw new Error(`Chapter ${chapter.number} has invalid part ${chapter.part}`);
  return part;
}

export function prevNext(n: number): {
  prev?: ChapterMeta;
  next?: ChapterMeta;
} {
  return { prev: chapterByNumber(n - 1), next: chapterByNumber(n + 1) };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS — all curriculum tests green.

- [ ] **Step 5: Delete the smoke test** (served its purpose; `git rm` deletes and stages in one step)

```bash
git rm src/lib/site/smoke.test.ts
```

- [ ] **Step 6: Commit**

```bash
git add src/lib/site/curriculum.ts src/lib/site/curriculum.test.ts
git commit -m "feat: canonical 18-chapter curriculum with lookup/nav helpers"
```

---

### Task 4: Base-path URL helper (TDD)

GitHub Pages serves under `/<repo>/`; every internal href must be base-aware.

**Files:**
- Create: `src/lib/site/url.ts`
- Test: `src/lib/site/url.test.ts`

- [ ] **Step 1: Write the failing test `src/lib/site/url.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { url } from './url';

describe('url', () => {
  it('joins against root base', () => {
    expect(url('chapters/the-neuron/', '/')).toBe('/chapters/the-neuron/');
  });

  it('joins against a repo base without trailing slash', () => {
    expect(url('chapters/the-neuron/', '/neural_networks')).toBe(
      '/neural_networks/chapters/the-neuron/'
    );
  });

  it('joins against a repo base with trailing slash', () => {
    expect(url('chapters/the-neuron/', '/neural_networks/')).toBe(
      '/neural_networks/chapters/the-neuron/'
    );
  });

  it('strips a leading slash on the path', () => {
    expect(url('/about/', '/neural_networks')).toBe('/neural_networks/about/');
  });

  it('returns the base itself for empty path (home link)', () => {
    expect(url('', '/neural_networks')).toBe('/neural_networks/');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — cannot resolve `./url`.

- [ ] **Step 3: Write `src/lib/site/url.ts`**

```ts
/**
 * Join a site-relative path against the configured base path.
 * Defaults to Astro's BASE_URL ("/" locally, "/<repo>/" on GitHub Pages).
 */
export function url(
  path: string,
  base: string = import.meta.env.BASE_URL
): string {
  const cleanBase = base.endsWith('/') ? base : `${base}/`;
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return cleanBase + cleanPath;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lib/site/url.ts src/lib/site/url.test.ts
git commit -m "feat: base-aware url helper for GitHub Pages paths"
```

---

### Task 5: Design system + BaseLayout

**Files:**
- Create: `src/styles/global.css`, `src/components/layout/BaseLayout.astro`

- [ ] **Step 1: Write `src/styles/global.css`**

```css
/* Warm field-notes design system — see spec §Visual & UX direction */
:root {
  --paper: #fdf6e3;
  --paper-raised: #fffdf5;
  --ink: #433422;
  --ink-strong: #586e75;
  --muted: #657b83;
  --faint: #93a1a1;
  --line: #d9c8a9;
  --line-soft: #e8dcc3;
  --accent-blue: #268bd2;
  --accent-red: #dc322f;
  --accent-green: #859900;
  --accent-gold: #b58900;

  --font-prose: Georgia, 'Iowan Old Style', 'Times New Roman', serif;
  --font-ui: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  --font-code: 'SF Mono', Menlo, Consolas, monospace;

  --measure: 42rem; /* prose column width */
}

* {
  box-sizing: border-box;
}

html {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font-prose);
  line-height: 1.7;
  -webkit-text-size-adjust: 100%;
}

body {
  margin: 0;
  font-size: 1.075rem;
}

h1, h2, h3 {
  color: var(--ink-strong);
  line-height: 1.25;
  font-weight: 700;
}

a {
  color: var(--accent-blue);
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}

code, pre {
  font-family: var(--font-code);
  font-size: 0.86em;
}

pre {
  background: var(--paper-raised);
  border: 1px solid var(--line-soft);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  overflow-x: auto;
}

/* Small uppercase label (part markers, chapter numbers) */
.label {
  font-family: var(--font-ui);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-gold);
}

/* Sketchy container used by figures and asides */
.sketch {
  background: var(--paper-raised);
  border: 2px dashed var(--line);
  border-radius: 12px;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

- [ ] **Step 2: Write `src/components/layout/BaseLayout.astro`**

```astro
---
import '../../styles/global.css';
import 'katex/dist/katex.min.css';

interface Props {
  title: string;
  description: string;
}

const { title, description } = Astro.props;
---

<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content={description} />
    <title>{title}</title>
  </head>
  <body>
    <slot />
  </body>
</html>
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: `Complete!` (BaseLayout is not referenced yet; this catches syntax/import errors via the page graph in later tasks — here it just confirms nothing broke).

- [ ] **Step 4: Commit**

```bash
git add src/styles/global.css src/components/layout/BaseLayout.astro
git commit -m "feat: warm field-notes design tokens and base layout"
```

---

### Task 6: Content collection + stub chapter + chapter shell

**Files:**
- Create: `src/content.config.ts`, `src/content/chapters/the-neuron.mdx`,
  `src/components/layout/MathAside.astro`, `src/components/layout/EngineerFooter.astro`,
  `src/components/layout/ChapterShell.astro`, `src/pages/chapters/[slug].astro`

- [ ] **Step 1: Write `src/content.config.ts`**

```ts
import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const chapters = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/chapters' }),
  schema: z.object({
    number: z.number().int().min(1).max(18),
    title: z.string(),
    summary: z.string(),
  }),
});

export const collections = { chapters };
```

- [ ] **Step 2: Write `src/components/layout/MathAside.astro`**

```astro
---
interface Props {
  title?: string;
}
const { title = 'Show the math' } = Astro.props;
---

<details class="math-aside sketch">
  <summary class="label">{title}</summary>
  <div class="math-aside-body"><slot /></div>
</details>

<style>
  .math-aside {
    margin: 1.5rem 0;
    padding: 0.75rem 1.25rem;
  }
  summary {
    cursor: pointer;
    color: var(--accent-blue);
  }
  .math-aside-body {
    padding-top: 0.5rem;
  }
</style>
```

- [ ] **Step 3: Write `src/components/layout/EngineerFooter.astro`**

```astro
---
---
<aside class="engineer-footer">
  <h2 class="label">For engineers</h2>
  <slot />
</aside>

<style>
  .engineer-footer {
    margin-top: 3rem;
    padding: 1.25rem 1.5rem;
    background: var(--paper-raised);
    border-left: 4px solid var(--accent-green);
    border-radius: 0 12px 12px 0;
  }
  .engineer-footer h2 {
    color: var(--accent-green);
    margin-top: 0;
  }
</style>
```

- [ ] **Step 4: Write `src/components/layout/ChapterShell.astro`**

```astro
---
import BaseLayout from './BaseLayout.astro';
import { CURRICULUM, partOf, prevNext } from '../../lib/site/curriculum';
import type { ChapterMeta } from '../../lib/site/curriculum';
import { url } from '../../lib/site/url';

interface Props {
  chapter: ChapterMeta;
  summary: string;
  /** slugs of chapters that have written content (for prev/next linking) */
  writtenSlugs: string[];
}

const { chapter, summary, writtenSlugs } = Astro.props;
const part = partOf(chapter);
const { prev, next } = prevNext(chapter.number);
const progress = (chapter.number / CURRICULUM.length) * 100;
const isWritten = (slug: string | undefined) =>
  slug !== undefined && writtenSlugs.includes(slug);
---

<BaseLayout title={`${chapter.title} · Neural Networks, Visually`} description={summary}>
  <div class="progress" aria-hidden="true">
    <div class="progress-fill" style={`width: ${progress}%`}></div>
  </div>

  <article class="chapter">
    <header>
      <p class="label">
        Part {part.numeral} · Chapter {chapter.number} of {CURRICULUM.length}
      </p>
      <h1>{chapter.title}</h1>
      <p class="summary">{summary}</p>
    </header>

    <slot />

    <nav class="chapter-nav" aria-label="Chapter navigation">
      {
        prev && isWritten(prev.slug) ? (
          <a href={url(`chapters/${prev.slug}/`)}>← {prev.title}</a>
        ) : (
          <span />
        )
      }
      <a href={url('')}>Trail map</a>
      {
        next && isWritten(next.slug) ? (
          <a href={url(`chapters/${next.slug}/`)}>{next.title} →</a>
        ) : (
          <span class="soon">Next: coming soon</span>
        )
      }
    </nav>
  </article>
</BaseLayout>

<style>
  .progress {
    position: sticky;
    top: 0;
    height: 4px;
    background: var(--line-soft);
    z-index: 10;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent-gold);
  }
  .chapter {
    max-width: var(--measure);
    margin: 0 auto;
    padding: 3rem 1.25rem 5rem;
  }
  header {
    text-align: center;
    margin-bottom: 3rem;
  }
  h1 {
    font-size: 2.4rem;
    margin: 0.4rem 0;
  }
  .summary {
    color: var(--muted);
    font-style: italic;
  }
  .chapter-nav {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 2px dashed var(--line);
    font-family: var(--font-ui);
    font-size: 0.9rem;
  }
  .soon {
    color: var(--faint);
  }
</style>
```

- [ ] **Step 5: Write `src/content/chapters/the-neuron.mdx`** (stub with real opening content — enough to exercise math asides, the engineer footer, and KaTeX)

````mdx
---
number: 1
title: The Neuron
summary: From if statements to weighted sums — the smallest unit of learning.
---

import MathAside from '../../components/layout/MathAside.astro';
import EngineerFooter from '../../components/layout/EngineerFooter.astro';

You already write neurons. Every time you code a threshold —

```ts
if (0.6 * temperature + 0.4 * humidity > 75) turnOnFan();
```

— you've built one: a weighted sum of inputs compared against a threshold.
That is the perceptron, the 1958 original, and it is the atom this entire
site builds from. The only thing it can't do yet is *choose its own weights*.

<MathAside>

A neuron computes $y = \sigma(w \cdot x + b)$, where $w$ are the weights,
$b$ is the bias, and $\sigma$ is an activation function. The perceptron's
$\sigma$ is a hard step: output $1$ if the sum is positive, else $0$.

</MathAside>

*(Interactive figure coming in Stage 2: adjust the weights yourself and try
to separate two clouds of points — then meet the XOR arrangement no single
line can split.)*

<EngineerFooter>

A neuron is `dot(w, x) + b` piped through a nonlinearity — a fused
multiply-accumulate with a squashing function. A layer is a matrix-vector
product. That's the whole trick; the rest of this site is about choosing
`w` automatically and wiring billions of these together.

</EngineerFooter>
````

- [ ] **Step 6: Write `src/pages/chapters/[slug].astro`**

```astro
---
import { getCollection, render } from 'astro:content';
import ChapterShell from '../../components/layout/ChapterShell.astro';
import { chapterBySlug } from '../../lib/site/curriculum';

export async function getStaticPaths() {
  const entries = await getCollection('chapters');
  return entries.map((entry) => ({
    params: { slug: entry.id },
    props: { entry, writtenSlugs: entries.map((e) => e.id) },
  }));
}

const { entry, writtenSlugs } = Astro.props;
const { Content } = await render(entry);
const meta = chapterBySlug(entry.id);
if (!meta) {
  throw new Error(
    `Chapter file "${entry.id}" has no matching slug in src/lib/site/curriculum.ts`
  );
}
if (meta.number !== entry.data.number) {
  throw new Error(
    `Chapter "${entry.id}": frontmatter number ${entry.data.number} != curriculum number ${meta.number}`
  );
}
---

<ChapterShell chapter={meta} summary={entry.data.summary} writtenSlugs={writtenSlugs}>
  <Content />
</ChapterShell>
```

- [ ] **Step 7: Verify build and output**

Run: `npm run build && ls dist/chapters/the-neuron/`
Expected: build `Complete!`; `index.html` listed. Then run
`grep -c "katex" dist/chapters/the-neuron/index.html` — expected: a number ≥ 1
(KaTeX markup rendered at build time).

- [ ] **Step 8: Run tests and typecheck**

Run: `npm test && npm run check`
Expected: tests PASS; `astro check` reports 0 errors.

- [ ] **Step 9: Commit**

```bash
git add src/content.config.ts src/content/chapters/the-neuron.mdx \
  src/components/layout/MathAside.astro src/components/layout/EngineerFooter.astro \
  src/components/layout/ChapterShell.astro src/pages/chapters/
git commit -m "feat: chapter shell, math asides, engineer footer, stub chapter 1"
```

---

### Task 7: Trail map + landing page

**Files:**
- Create: `src/components/layout/TrailMap.astro`
- Modify: `src/pages/index.astro` (replace placeholder entirely)

- [ ] **Step 1: Write `src/components/layout/TrailMap.astro`**

```astro
---
import { CURRICULUM, PARTS, partOf } from '../../lib/site/curriculum';
import { url } from '../../lib/site/url';

interface Props {
  /** slugs of chapters that have written content */
  writtenSlugs: string[];
}
const { writtenSlugs } = Astro.props;

// Hand-placed stops: a meandering trail, bottom-left to top-right.
// One entry per chapter, index = chapter number - 1.
const STOPS: Array<{ x: number; y: number }> = [
  { x: 70, y: 520 },  { x: 150, y: 480 }, { x: 230, y: 500 }, { x: 310, y: 450 },
  { x: 380, y: 480 }, { x: 450, y: 430 }, { x: 520, y: 460 }, { x: 590, y: 410 },
  { x: 640, y: 340 }, { x: 570, y: 290 }, { x: 490, y: 310 }, { x: 410, y: 270 },
  { x: 330, y: 290 }, { x: 250, y: 250 }, { x: 320, y: 180 }, { x: 400, y: 150 },
  { x: 480, y: 170 }, { x: 560, y: 100 },
];

const PART_COLORS: Record<number, string> = {
  1: 'var(--accent-blue)',
  2: 'var(--accent-green)',
  3: 'var(--accent-gold)',
  4: 'var(--accent-red)',
};

const pathD = STOPS.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

const stops = CURRICULUM.map((chapter) => ({
  chapter,
  pos: STOPS[chapter.number - 1],
  color: PART_COLORS[chapter.part],
  written: writtenSlugs.includes(chapter.slug),
}));
---

<figure class="trail sketch">
  <svg viewBox="0 0 720 560" role="img" aria-label="Course trail map: 18 chapters from the neuron to frontier models">
    <path d={pathD} fill="none" stroke="var(--line)" stroke-width="3" stroke-dasharray="7 6" stroke-linecap="round"></path>
    {
      stops.map(({ chapter, pos, color, written }) =>
        written ? (
          <a href={url(`chapters/${chapter.slug}/`)}>
            <circle cx={pos.x} cy={pos.y} r="14" fill={color} />
            <text x={pos.x} y={pos.y + 4} text-anchor="middle" class="stop-num">{chapter.number}</text>
            <text x={pos.x} y={pos.y + 32} text-anchor="middle" class="stop-title">{chapter.title}</text>
          </a>
        ) : (
          <g class="soon">
            <circle cx={pos.x} cy={pos.y} r="14" fill="none" stroke={color} stroke-width="2.5" stroke-dasharray="4 3" />
            <text x={pos.x} y={pos.y + 4} text-anchor="middle" class="stop-num soon-num">{chapter.number}</text>
            <text x={pos.x} y={pos.y + 32} text-anchor="middle" class="stop-title">{chapter.title}</text>
          </g>
        )
      )
    }
  </svg>
  <figcaption>
    {
      PARTS.map((part) => (
        <span class="legend-item">
          <span class="legend-dot" style={`background: ${PART_COLORS[part.number]}`} />
          Part {part.numeral}: {part.title}
        </span>
      ))
    }
  </figcaption>
</figure>

<style>
  .trail {
    padding: 1.5rem;
    margin: 0;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
  }
  .stop-num {
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 700;
    fill: #fffdf5;
  }
  .soon-num {
    fill: var(--faint);
  }
  .stop-title {
    font-family: var(--font-ui);
    font-size: 11px;
    fill: var(--muted);
  }
  .soon .stop-title {
    fill: var(--faint);
  }
  a:hover circle {
    stroke: var(--ink-strong);
    stroke-width: 3;
  }
  figcaption {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.5rem;
    justify-content: center;
    margin-top: 1rem;
    font-family: var(--font-ui);
    font-size: 0.85rem;
    color: var(--muted);
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
</style>
```

- [ ] **Step 2: Replace `src/pages/index.astro`**

```astro
---
import { getCollection } from 'astro:content';
import BaseLayout from '../components/layout/BaseLayout.astro';
import TrailMap from '../components/layout/TrailMap.astro';

const entries = await getCollection('chapters');
const writtenSlugs = entries.map((e) => e.id);
---

<BaseLayout
  title="Neural Networks, Visually"
  description="An illustrated, interactive journey from a single neuron to frontier AI architectures — built for engineers."
>
  <main>
    <header class="hero">
      <p class="label">A visual field guide for engineers</p>
      <h1>Neural Networks, Visually</h1>
      <p class="tagline">
        From a single neuron to frontier architectures — every idea explained
        by something you can see, poke, and train yourself. No math required
        (it's there if you want it).
      </p>
    </header>
    <TrailMap writtenSlugs={writtenSlugs} />
  </main>
</BaseLayout>

<style>
  main {
    max-width: 52rem;
    margin: 0 auto;
    padding: 3.5rem 1.25rem 5rem;
  }
  .hero {
    text-align: center;
    margin-bottom: 2.5rem;
  }
  h1 {
    font-size: 3rem;
    margin: 0.4rem 0;
  }
  .tagline {
    color: var(--muted);
    font-style: italic;
    max-width: 34rem;
    margin: 0 auto;
  }
</style>
```

- [ ] **Step 3: Verify build and links**

Run: `npm run build && npx linkinator ./dist --recurse --silent`
Expected: build `Complete!`; linkinator exits 0 (no broken links).

- [ ] **Step 4: Visual check**

Run: `npm run dev` and view `http://localhost:4321/` — trail map renders with
stop 1 as a solid clickable link and stops 2–18 dashed "coming soon"; clicking
stop 1 opens the chapter with progress bar, math aside, and engineer footer.
(If working non-interactively: `curl -s http://localhost:4321/ | grep -c "coming soon\|the-neuron"` ≥ 1.)

- [ ] **Step 5: Commit**

```bash
git add src/components/layout/TrailMap.astro src/pages/index.astro
git commit -m "feat: trail-map landing page with written/coming-soon stops"
```

---

### Task 8: CI + GitHub Pages deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml`, `README.md`

- [ ] **Step 1: Write `.github/workflows/deploy.yml`**

```yaml
name: CI & Deploy

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
      - run: npm ci
      - run: npm test
      - run: npm run check
      - run: npm run build
      - run: npx linkinator ./dist --recurse --silent

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # Builds with site/base derived from the repo's Pages config
      - uses: withastro/action@v3

  deploy:
    if: github.ref == 'refs/heads/main'
    needs: [test, build]
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

- [ ] **Step 2: Write `README.md`**

```markdown
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
```

- [ ] **Step 3: Verify workflow YAML parses**

Run: `npx --yes js-yaml .github/workflows/deploy.yml > /dev/null && echo YAML-OK && npm run build`
Expected: `YAML-OK` printed (file parses), then build `Complete!`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/deploy.yml README.md
git commit -m "ci: test + build + GitHub Pages deploy workflow"
```

---

### Task 9: Publish to GitHub (REQUIRES USER CONFIRMATION)

Creating a public repo and pushing is outward-facing — **confirm with Chris before executing this task.**

- [ ] **Step 1: Confirm with the user** — repo name (default `neural_networks`), visibility (public, since it's a public learning resource), and that they're ready to publish.

- [ ] **Step 2: Create repo and push** (after confirmation)

```bash
gh repo create neural_networks --public --source=. --push
```

Expected: repo created; `main` pushed.

- [ ] **Step 3: Enable Pages via Actions source**

```bash
gh api -X POST repos/{owner}/neural_networks/pages -f build_type=workflow 2>/dev/null \
  || gh api -X PUT repos/{owner}/neural_networks/pages -f build_type=workflow
```

Expected: Pages configured with `build_type: workflow`. (If the API call fails, instruct the user: Settings → Pages → Source → GitHub Actions.)

- [ ] **Step 4: Watch the first deploy**

```bash
gh run watch --exit-status
```

Expected: `CI & Deploy` succeeds; site live at `https://<owner>.github.io/neural_networks/`. Verify the trail map loads and chapter 1's links work under the `/neural_networks/` base path.

---

## Out of scope for Stage 1

- Svelte figure components, ScrollStage, the micro-ML engine (`lib/nn`) — Stage 2.
- Chapters 2–18 content.
- Figure fallback-snapshot mechanism (arrives with the first real figure in Stage 2).

## Addendum: review-driven deviations (recorded post-execution)

Code review during execution amended the plan as follows — later stages should treat these as canonical:

1. **Base-path premise was wrong (Critical).** `withastro/action@v3` does NOT inject `site`/`base` (verified against its action.yml). Fix: `astro.config.mjs` reads `site: process.env.SITE_URL` / `base: process.env.BASE_PATH`; the workflow's build job passes `SITE_URL=https://${{ github.repository_owner }}.github.io` and `BASE_PATH=/${{ github.event.repository.name }}` as step env on `withastro/action@v3`. Local/test builds keep base `/`.
2. **Linkinator skip pattern**: `--skip "^https?://(?!localhost)"` (a bare `^https?://` skips everything in directory mode — silently disables the gate).
3. **WCAG text tokens**: added `--accent-gold-ink: #8a6800` (used by `.label`) and `--link: #1a6fa8` (used by `a`); `--accent-gold`/`--accent-blue` remain decorative-only. `scroll-behavior: auto !important` added to the reduced-motion block.
4. **KaTeX CSS** imports in `ChapterShell.astro`, not `BaseLayout.astro` (non-chapter pages don't pay for it).
5. **Prose rhythm CSS** is scoped `.chapter :where(p, ul, ol, blockquote, hr, figure, figcaption)` — never bare element selectors (MDX output has no Astro scope attr; bare selectors leak into UI chrome).
6. **`[slug].astro` validates frontmatter `title`** against the curriculum in addition to `number` (build fails on drift).
7. **TrailMap accessibility/mobile**: no `role="img"` on the interactive SVG — `<nav aria-label>` wrapper, `<title>`+`aria-labelledby`, per-link `aria-label`s, transparent r=24 hit targets, `a:focus-visible .stop-dot` ring, `STOPS.length === CURRICULUM.length` build guard, and a `<ol class="trail-list">` fallback swapped in under 640px (SVG hidden — exactly one nav exposed per breakpoint).
8. **`package.json`** gained `"engines": { "node": ">=20.3.0" }`.
9. Consciously accepted (documented, not bugs): white numerals on green/gold trail stops are below AA (decorative; titles carry meaning), `--faint` "coming soon" text is low-contrast, curriculum `numeral` field is redundant with `number`.
