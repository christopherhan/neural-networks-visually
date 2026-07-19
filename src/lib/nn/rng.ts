/** Deterministic PRNG. All randomness in lib/nn flows through an injected Rng
 *  so tests and figures are reproducible. */
export type Rng = () => number;

/** mulberry32 — tiny, fast, good-enough 32-bit PRNG. Returns values in [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal sample via Box–Muller. */
export function gaussian(rng: Rng): number {
  // avoid log(0) — mulberry32 can emit exactly 0
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
