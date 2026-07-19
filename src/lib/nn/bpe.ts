// TypeScript mirror of tools/train-tiny-transformer/bpe.py's encode/decode rule.
// Must match the Python implementation exactly (see bpe.py's `encode_word`/`encode`).

export interface Tokenizer {
  vocab: string[];
  merges: [string, string][];
}

const WORD_RE = / ?[A-Za-z']+| ?[.,;:!?"()\-]|\n/g;

export function createEncoder(t: Tokenizer): {
  encode(text: string): number[];
  decode(ids: number[]): string;
  tokenText(id: number): string;
} {
  const ranks = new Map<string, number>();
  t.merges.forEach(([a, b], i) => ranks.set(`${a} ${b}`, i));

  const ids = new Map<string, number>();
  t.vocab.forEach((tok, i) => ids.set(tok, i));

  function encodeWord(word: string): string[] {
    let seq = Array.from(word);
    while (seq.length > 1) {
      let bestRank: number | undefined;
      let bestI = -1;
      for (let i = 0; i < seq.length - 1; i++) {
        const r = ranks.get(`${seq[i]} ${seq[i + 1]}`);
        if (r !== undefined && (bestRank === undefined || r < bestRank)) {
          bestRank = r;
          bestI = i;
        }
      }
      if (bestI === -1) break;
      const merged = seq[bestI] + seq[bestI + 1];
      seq = [...seq.slice(0, bestI), merged, ...seq.slice(bestI + 2)];
    }
    return seq;
  }

  function encode(text: string): number[] {
    const out: number[] = [];
    const words = text.match(WORD_RE) ?? [];
    const wordCache = new Map<string, string[]>();
    for (const w of words) {
      let symbols = wordCache.get(w);
      if (!symbols) {
        symbols = encodeWord(w);
        wordCache.set(w, symbols);
      }
      for (const sym of symbols) {
        const id = ids.get(sym);
        if (id !== undefined) out.push(id);
      }
    }
    return out;
  }

  function decode(idList: number[]): string {
    return idList.map((id) => t.vocab[id]).join('');
  }

  function tokenText(id: number): string {
    return t.vocab[id] ?? '';
  }

  return { encode, decode, tokenText };
}
