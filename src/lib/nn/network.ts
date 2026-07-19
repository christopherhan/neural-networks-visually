import { Value } from './value';
import type { Rng } from './rng';

export type Activation = 'tanh' | 'relu' | 'linear';

export class Neuron {
  readonly w: Value[];
  readonly b: Value;
  readonly activation: Activation;

  constructor(nin: number, activation: Activation, rng: Rng) {
    this.w = Array.from({ length: nin }, () => new Value(rng() * 2 - 1));
    this.b = new Value(0);
    this.activation = activation;
  }

  forward(x: Value[]): Value {
    let act: Value = this.b;
    for (let i = 0; i < this.w.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    if (this.activation === 'tanh') return act.tanh();
    if (this.activation === 'relu') return act.relu();
    return act;
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  readonly neurons: Neuron[];

  constructor(nin: number, nout: number, activation: Activation, rng: Rng) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, activation, rng));
  }

  forward(x: Value[]): Value[] {
    return this.neurons.map((n) => n.forward(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

export class MLP {
  readonly layers: Layer[];

  /** sizes e.g. [2, 8, 1]: 2 inputs, one hidden layer of 8, 1 output.
   *  Hidden layers use `hidden` activation; the last layer uses `output`. */
  constructor(
    sizes: number[],
    rng: Rng,
    hidden: Activation = 'tanh',
    output: Activation = 'tanh'
  ) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const isLast = i === sizes.length - 2;
      this.layers.push(new Layer(sizes[i], sizes[i + 1], isLast ? output : hidden, rng));
    }
  }

  forward(xs: number[]): Value[] {
    let v = xs.map((x) => new Value(x));
    for (const layer of this.layers) v = layer.forward(v);
    return v;
  }

  predict(xs: number[]): number[] {
    return this.forward(xs).map((v) => v.data);
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }
}

export function mseLoss(preds: Value[], targets: number[]): Value {
  let sum = new Value(0);
  for (let i = 0; i < preds.length; i++) {
    sum = sum.add(preds[i].sub(targets[i]).pow(2));
  }
  return sum.div(preds.length);
}

export function zeroGrad(params: Value[]): void {
  for (const p of params) p.grad = 0;
}

export function sgdStep(params: Value[], lr: number): void {
  for (const p of params) p.data -= lr * p.grad;
}

/** One full-batch gradient step. Returns the batch loss before the update. */
export function trainStep(
  mlp: MLP,
  inputs: number[][],
  targets: number[][],
  lr: number
): number {
  const params = mlp.parameters();
  zeroGrad(params);
  let total = new Value(0);
  for (let i = 0; i < inputs.length; i++) {
    total = total.add(mseLoss(mlp.forward(inputs[i]), targets[i]));
  }
  const loss = total.div(inputs.length);
  loss.backward();
  sgdStep(params, lr);
  return loss.data;
}
