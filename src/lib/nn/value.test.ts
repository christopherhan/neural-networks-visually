import { describe, it, expect } from 'vitest';
import { Value } from './value';

/** Numerical derivative of f at x via central difference. */
function numericGrad(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

describe('Value forward pass', () => {
  it('computes basic ops', () => {
    const a = new Value(2);
    const b = new Value(-3);
    expect(a.add(b).data).toBe(-1);
    expect(a.mul(b).data).toBe(-6);
    expect(a.sub(b).data).toBe(5);
    expect(a.div(new Value(4)).data).toBe(0.5);
    expect(a.pow(3).data).toBe(8);
    expect(a.neg().data).toBe(-2);
    expect(new Value(0).exp().data).toBe(1);
    expect(new Value(0).tanh().data).toBe(0);
    expect(new Value(-1).relu().data).toBe(0);
    expect(new Value(2).relu().data).toBe(2);
    expect(new Value(0).sigmoid().data).toBeCloseTo(0.5);
  });

  it('accepts plain numbers as operands', () => {
    const a = new Value(2);
    expect(a.add(1).data).toBe(3);
    expect(a.mul(3).data).toBe(6);
    expect(a.sub(1).data).toBe(1);
    expect(a.div(2).data).toBe(1);
  });
});

describe('Value backward pass — gradient checks vs numerical differentiation', () => {
  const cases: Array<{
    name: string;
    fv: (x: Value) => Value;
    fn: (x: number) => number;
    at: number[];
  }> = [
    { name: 'add', fv: (x) => x.add(3), fn: (x) => x + 3, at: [2, -1.5] },
    { name: 'mul', fv: (x) => x.mul(-4), fn: (x) => x * -4, at: [2, 0.3] },
    { name: 'pow', fv: (x) => x.pow(3), fn: (x) => x ** 3, at: [2, -1.2] },
    { name: 'div', fv: (x) => new Value(1).div(x), fn: (x) => 1 / x, at: [2, -3] },
    { name: 'exp', fv: (x) => x.exp(), fn: (x) => Math.exp(x), at: [0.5, -1] },
    { name: 'tanh', fv: (x) => x.tanh(), fn: (x) => Math.tanh(x), at: [0.5, -2] },
    { name: 'relu+', fv: (x) => x.relu(), fn: (x) => Math.max(0, x), at: [1.5] },
    { name: 'relu-', fv: (x) => x.relu(), fn: (x) => Math.max(0, x), at: [-1.5] },
    {
      name: 'sigmoid',
      fv: (x) => x.sigmoid(),
      fn: (x) => 1 / (1 + Math.exp(-x)),
      at: [0.7, -0.7],
    },
    {
      name: 'composite tanh(x*3 + x^2)',
      fv: (x) => x.mul(3).add(x.pow(2)).tanh(),
      fn: (x) => Math.tanh(x * 3 + x ** 2),
      at: [0.4, -0.6],
    },
  ];

  for (const { name, fv, fn, at } of cases) {
    for (const x0 of at) {
      it(`${name} at x=${x0}`, () => {
        const x = new Value(x0);
        const y = fv(x);
        y.backward();
        expect(x.grad).toBeCloseTo(numericGrad(fn, x0), 4);
      });
    }
  }

  it('accumulates gradients when a value is used twice', () => {
    // y = x*x + x  =>  dy/dx = 2x + 1
    const x = new Value(3);
    const y = x.mul(x).add(x);
    y.backward();
    expect(x.grad).toBeCloseTo(7);
  });

  it('propagates through multi-variable expressions', () => {
    // z = (a*b + a).tanh(); check both partials numerically
    const f = (a: number, b: number) => Math.tanh(a * b + a);
    const a = new Value(0.5);
    const b = new Value(-1.3);
    const z = a.mul(b).add(a).tanh();
    z.backward();
    expect(a.grad).toBeCloseTo(numericGrad((x) => f(x, -1.3), 0.5), 4);
    expect(b.grad).toBeCloseTo(numericGrad((x) => f(0.5, x), -1.3), 4);
  });

  it('backward on a diamond graph visits each node once', () => {
    // y = (x+x) * (x+x) = 4x^2  =>  dy/dx = 8x
    const x = new Value(2);
    const s = x.add(x);
    const y = s.mul(s);
    y.backward();
    expect(x.grad).toBeCloseTo(16);
  });
});
