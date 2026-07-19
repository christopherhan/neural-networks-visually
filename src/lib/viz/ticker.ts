/** requestAnimationFrame loop throttled to a target fps.
 *  raf/caf are injectable so tests can drive time explicitly. */
export interface Ticker {
  start(): void;
  stop(): void;
  readonly running: boolean;
}

interface TickerOptions {
  fps?: number;
  raf?: (cb: (t: number) => void) => number;
  caf?: (id: number) => void;
}

export function createTicker(tick: () => void, options: TickerOptions = {}): Ticker {
  const fps = options.fps ?? 30;
  const raf = options.raf ?? ((cb: (t: number) => void) => globalThis.requestAnimationFrame(cb));
  const caf = options.caf ?? ((id: number) => globalThis.cancelAnimationFrame(id));
  const interval = 1000 / fps;
  let id: number | null = null;
  let last = -Infinity;

  const loop = (t: number) => {
    id = raf(loop);
    // Anchor to the actual frame time (not last += interval): after a
    // backgrounded-tab stall this yields ONE catch-up tick, never a burst.
    if (t - last >= interval) {
      last = t;
      tick();
    }
  };

  return {
    start() {
      if (id === null) {
        last = -Infinity;
        id = raf(loop);
      }
    },
    stop() {
      if (id !== null) {
        caf(id);
        id = null;
      }
    },
    get running() {
      return id !== null;
    },
  };
}
