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
