import { describe, it, expect } from 'vitest';
import { mulberry32 } from './rng';
import { MLP, mseLoss, trainStep, zeroGrad } from './network';
import { Value } from './value';

describe('MLP structure', () => {
  it('has the right parameter count', () => {
    // [2,4,1]: layer1 = 4*(2+1) = 12, layer2 = 1*(4+1) = 5 → 17
    const mlp = new MLP([2, 4, 1], mulberry32(1));
    expect(mlp.parameters()).toHaveLength(17);
  });

  it('forward returns one Value per output and predict returns numbers', () => {
    const mlp = new MLP([2, 3, 2], mulberry32(1));
    const out = mlp.forward([0.5, -0.5]);
    expect(out).toHaveLength(2);
    expect(out[0]).toBeInstanceOf(Value);
    const nums = mlp.predict([0.5, -0.5]);
    expect(nums).toHaveLength(2);
    expect(typeof nums[0]).toBe('number');
  });

  it('tanh outputs are bounded in (-1, 1)', () => {
    const mlp = new MLP([2, 4, 1], mulberry32(3));
    for (const p of [[1, 1], [-1, 1], [0.3, -0.9]]) {
      const [y] = mlp.predict(p);
      expect(Math.abs(y)).toBeLessThan(1);
    }
  });

  it('is deterministic for a given seed', () => {
    const a = new MLP([2, 4, 1], mulberry32(42)).predict([0.2, 0.8]);
    const b = new MLP([2, 4, 1], mulberry32(42)).predict([0.2, 0.8]);
    expect(a).toEqual(b);
  });
});

describe('loss and training', () => {
  it('mseLoss is zero for perfect predictions and positive otherwise', () => {
    expect(mseLoss([new Value(1), new Value(-1)], [1, -1]).data).toBe(0);
    expect(mseLoss([new Value(0.5)], [1]).data).toBeCloseTo(0.25);
  });

  it('zeroGrad clears gradients', () => {
    const mlp = new MLP([2, 2, 1], mulberry32(1));
    const loss = mseLoss(mlp.forward([1, -1]), [1]);
    loss.backward();
    const params = mlp.parameters();
    expect(params.some((p) => p.grad !== 0)).toBe(true);
    zeroGrad(params);
    expect(params.every((p) => p.grad === 0)).toBe(true);
  });

  it('a trainStep reduces loss on a simple batch', () => {
    const mlp = new MLP([2, 4, 1], mulberry32(7));
    const inputs = [[0.5, 0.5], [-0.5, -0.5]];
    const targets = [[1], [-1]];
    const first = trainStep(mlp, inputs, targets, 0.1);
    let last = first;
    for (let i = 0; i < 20; i++) last = trainStep(mlp, inputs, targets, 0.1);
    expect(last).toBeLessThan(first);
  });

  it('learns XOR — the integration test for the whole engine', () => {
    // If the seed proves unlucky for convergence, try small integer seeds and
    // pin the first that converges — determinism is the point, not seed 42.
    const mlp = new MLP([2, 8, 1], mulberry32(1));
    const inputs = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
    const targets = [[-1], [1], [1], [-1]];
    let loss = Infinity;
    for (let epoch = 0; epoch < 800; epoch++) {
      loss = trainStep(mlp, inputs, targets, 0.3);
    }
    expect(loss).toBeLessThan(0.05);
    for (let i = 0; i < inputs.length; i++) {
      const [y] = mlp.predict(inputs[i]);
      expect(Math.sign(y)).toBe(Math.sign(targets[i][0]));
    }
  });
});
