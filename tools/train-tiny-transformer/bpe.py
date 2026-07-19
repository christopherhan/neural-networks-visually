"""Train a small BPE tokenizer on the bundled public-domain corpus.

Word-level BPE (fast): split text into space-prefixed words, count unique
words, learn merges over their character sequences. The exported format is
deliberately simple so the site's TypeScript encoder can mirror it exactly:

  tokenizer.json = {
    "vocab": ["a", "b", ..., " the", ...],   # id = index
    "merges": [["t","h"], ["th","e"], ...],  # rank = index
  }

Encoding rule (identical in TS): split text into words with the WORD_RE
pattern, start each word as a list of characters, repeatedly merge the
adjacent pair with the LOWEST merge rank until no mergeable pair remains,
then map symbols to ids (unknown symbols are skipped).
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
CORPUS_FILES = ["alice.txt", "looking-glass.txt", "peter-pan.txt"]
N_MERGES = 320
WORD_RE = re.compile(r" ?[A-Za-z']+| ?[.,;:!?\"()\-]|\n")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Strip Gutenberg header/footer.
    start = text.find("*** START")
    if start != -1:
        text = text[text.find("\n", start) + 1 :]
    end = text.find("*** END")
    if end != -1:
        text = text[:end]
    text = text.replace("\r", "")
    text = re.sub(r"_+", "", text)  # italics markers
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_corpus() -> str:
    parts = [normalize((HERE / f).read_text(encoding="utf-8")) for f in CORPUS_FILES]
    return "\n".join(parts)


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


def train(text: str, n_merges: int) -> tuple[list[str], list[tuple[str, str]]]:
    word_counts = Counter(words_of(text))
    # Each unique word as a tuple of single-character symbols.
    seqs: dict[tuple[str, ...], int] = {tuple(w): c for w, c in word_counts.items()}

    base_symbols = sorted({ch for w in seqs for ch in w})
    merges: list[tuple[str, str]] = []

    for _ in range(n_merges):
        pair_counts: Counter = Counter()
        for seq, count in seqs.items():
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] += count
        if not pair_counts:
            break
        (a, b), top = pair_counts.most_common(1)[0]
        if top < 2:
            break
        merges.append((a, b))
        merged = a + b
        new_seqs: dict[tuple[str, ...], int] = {}
        for seq, count in seqs.items():
            out: list[str] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_seqs[tuple(out)] = new_seqs.get(tuple(out), 0) + count
        seqs = new_seqs

    vocab = base_symbols + [a + b for a, b in merges]
    return vocab, merges


def encode_word(word: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    seq = list(word)
    while len(seq) > 1:
        best_rank, best_i = None, None
        for i, pair in enumerate(zip(seq, seq[1:])):
            r = ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_i is None:
            break
        seq[best_i : best_i + 2] = [seq[best_i] + seq[best_i + 1]]
    return seq


def encode(text: str, vocab: list[str], merges: list[tuple[str, str]]) -> list[int]:
    ranks = {pair: i for i, pair in enumerate(merges)}
    ids = {tok: i for i, tok in enumerate(vocab)}
    out: list[int] = []
    cache: dict[str, list[str]] = {}
    for w in words_of(text):
        if w not in cache:
            cache[w] = encode_word(w, ranks)
        for sym in cache[w]:
            if sym in ids:
                out.append(ids[sym])
    return out


def main() -> None:
    text = load_corpus()
    print(f"corpus: {len(text):,} chars")
    vocab, merges = train(text, N_MERGES)
    print(f"vocab: {len(vocab)} tokens ({len(merges)} merges)")

    (HERE / "tokenizer.json").write_text(
        json.dumps({"vocab": vocab, "merges": [list(m) for m in merges]})
    )

    token_ids = encode(text, vocab, merges)
    print(f"encoded corpus: {len(token_ids):,} tokens "
          f"({len(text) / len(token_ids):.2f} chars/token)")
    (HERE / "corpus_ids.json").write_text(json.dumps(token_ids))

    # Reference encodings for the TS parity test.
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Alice was beginning to get very tired.",
        "Attention is all you need!",
    ]
    refs = [{"text": s, "ids": encode(s, vocab, merges)} for s in samples]
    (HERE / "tokenizer_ref.json").write_text(json.dumps(refs))
    for r in refs:
        print(f"  {r['text']!r} -> {len(r['ids'])} tokens")


if __name__ == "__main__":
    main()
