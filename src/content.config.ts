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
