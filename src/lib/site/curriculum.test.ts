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

  it('part assignments are contiguous and non-decreasing', () => {
    const parts = CURRICULUM.map((c) => c.part);
    for (let i = 1; i < parts.length; i++) {
      expect(parts[i], `chapter ${i + 1} part`).toBeGreaterThanOrEqual(parts[i - 1]);
    }
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
