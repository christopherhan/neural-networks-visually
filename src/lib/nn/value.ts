/**
 * Scalar autograd, micrograd-style. Every arithmetic op returns a new Value
 * that remembers its inputs and how to route gradients back to them.
 * `backward()` topologically sorts the graph and applies the chain rule.
 *
 * This file is deliberately small and readable: Chapter 4 shows it as
 * teaching material.
 */
export class Value {
  data: number;
  grad = 0;
  private _backward: () => void = () => {};
  private readonly _prev: readonly Value[];

  constructor(data: number, prev: readonly Value[] = []) {
    this.data = data;
    this._prev = prev;
  }

  private static of(x: number | Value): Value {
    return x instanceof Value ? x : new Value(x);
  }

  add(other: number | Value): Value {
    const o = Value.of(other);
    const out = new Value(this.data + o.data, [this, o]);
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: number | Value): Value {
    const o = Value.of(other);
    const out = new Value(this.data * o.data, [this, o]);
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  /** Raise to a constant power (exponent is not differentiated). */
  pow(n: number): Value {
    const out = new Value(this.data ** n, [this]);
    out._backward = () => {
      this.grad += n * this.data ** (n - 1) * out.grad;
    };
    return out;
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: number | Value): Value {
    return this.add(Value.of(other).neg());
  }

  div(other: number | Value): Value {
    return this.mul(Value.of(other).pow(-1));
  }

  exp(): Value {
    const out = new Value(Math.exp(this.data), [this]);
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this]);
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this]);
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this]);
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  /** Backpropagate: seed d(this)/d(this) = 1 and apply the chain rule in
   *  reverse topological order. */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }
}
