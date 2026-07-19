import { describe, it, expect } from 'vitest';
import { createTicker } from './ticker';

/** Fake rAF: caller pumps frames with explicit timestamps. */
function fakeRaf() {
  let nextId = 1;
  const pending = new Map<number, (t: number) => void>();
  return {
    raf: (cb: (t: number) => void) => {
      const id = nextId++;
      pending.set(id, cb);
      return id;
    },
    caf: (id: number) => {
      pending.delete(id);
    },
    pump(t: number) {
      const cbs = [...pending.values()];
      pending.clear();
      for (const cb of cbs) cb(t);
    },
    get scheduled() {
      return pending.size;
    },
  };
}

describe('createTicker', () => {
  it('fires at most once per interval', () => {
    const frames = fakeRaf();
    let ticks = 0;
    const t = createTicker(() => ticks++, { fps: 10, raf: frames.raf, caf: frames.caf });
    t.start();
    frames.pump(0);      // first frame always ticks
    frames.pump(50);     // 50ms < 100ms interval — no tick
    frames.pump(100);    // 100ms since last tick — ticks
    frames.pump(140);    // no
    frames.pump(210);    // yes
    expect(ticks).toBe(3);
  });

  it('stop cancels the pending frame; start/stop are idempotent', () => {
    const frames = fakeRaf();
    let ticks = 0;
    const t = createTicker(() => ticks++, { fps: 60, raf: frames.raf, caf: frames.caf });
    t.start();
    t.start();
    expect(frames.scheduled).toBe(1);
    t.stop();
    t.stop();
    expect(frames.scheduled).toBe(0);
    frames.pump(1000);
    expect(ticks).toBe(0);
    expect(t.running).toBe(false);
  });

  it('reports running state', () => {
    const frames = fakeRaf();
    const t = createTicker(() => {}, { raf: frames.raf, caf: frames.caf });
    expect(t.running).toBe(false);
    t.start();
    expect(t.running).toBe(true);
    t.stop();
    expect(t.running).toBe(false);
  });

  it('handles stop() called from inside the tick callback', () => {
    const frames = fakeRaf();
    let ticks = 0;
    const t = createTicker(
      () => {
        ticks++;
        if (ticks === 2) t.stop();
      },
      { fps: 1000, raf: frames.raf, caf: frames.caf }
    );
    t.start();
    frames.pump(0);
    frames.pump(10);
    expect(ticks).toBe(2);
    expect(t.running).toBe(false);
    expect(frames.scheduled).toBe(0);
    frames.pump(20);
    expect(ticks).toBe(2);
  });
});
