# Prose Style Guide

The site's voice is a knowledgeable colleague explaining things well: warm,
concrete, intuition-first, unafraid of a good metaphor. What it must never
sound like is machine-generated text. The tics below are the tells. Remove
them; keep the personality.

## The rules

1. **Em-dashes are rare.** Budget: at most one per ~150 words, never two in
   one sentence. Default replacements: a period and a new sentence, a comma,
   a colon, or parentheses. If a sentence needs a dash to work, it usually
   needs restructuring instead.

2. **No three-beat rhythm.** Lists of exactly three ("stacked, widened,
   sparsified") are the most recognizable LLM cadence. Break them: use two
   items, four items, or a plain sentence. Exception: things that genuinely
   come in threes (query/key/value, the LSTM's three gates).

3. **No colon-setup openers.** "Here's the uncomfortable truth:", "The
   insight:", "The pattern to notice:" and relatives. Just say the thing.

4. **No single-word dramatic sentences.** ("Layers!") No "that's the whole
   trick" / "that's the entire secret" flourishes.

5. **Ration metaphors.** Vivid images are part of the voice (the loss
   landscape hike, the attention library). Keep the load-bearing one or two
   per chapter; cut decorative ones.

6. **Vary sentence length honestly.** Some facts deserve a plain declarative
   sentence with no rhetorical shaping at all.

7. **Bold sparingly.** Bold introduces a term the first time it appears.
   It is not emphasis seasoning.

## Hard constraints for any revision pass

- Every number, percentage, model dimension, name, date, URL, and code
  identifier stays exactly as written. These were empirically verified;
  style edits must not drift them.
- Math inside `$...$` and code inside backticks or fences is untouchable.
- Frontmatter, imports, component tags (`<Figure... />`, `<MathAside>`,
  `<EngineerFooter>`) and their placement are untouchable.
- Technical claims keep their meaning precisely. When in doubt, restructure
  the sentence around the claim rather than paraphrasing the claim.
