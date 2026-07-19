// Pure-matrix inference for the tiny GPT trained by tools/train-tiny-transformer/train.py.
// No autograd, no dependencies — just number[][] arithmetic. Mirrors the Python
// forward() exactly (pre-LN, causal multi-head attention, ReLU MLP, tied embeddings)
// so it can be checked for parity against ref_logits.json. See train.py's `forward`,
// `attn_fwd`, and `mlp_fwd` for the reference implementation this port follows line
// for line.

import type { Rng } from './rng';

export interface LayerWeights {
  ln1_g: number[];
  ln1_b: number[];
  wq: number[][];
  wk: number[][];
  wv: number[][];
  wo: number[][];
  ln2_g: number[];
  ln2_b: number[];
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
}

export interface TransformerConfig {
  vocab_size: number;
  d_model: number;
  n_head: number;
  n_layer: number;
  ctx: number;
  mlp: number;
}

export interface TransformerModel {
  config: TransformerConfig;
  tok_emb: number[][];
  pos_emb: number[][];
  lnf_g: number[];
  lnf_b: number[];
  layers: LayerWeights[];
}

const EPS = 1e-5;

/** Minimal structural validation — enough to fail loudly on a malformed weights.json. */
export function loadModel(json: unknown): TransformerModel {
  if (typeof json !== 'object' || json === null) {
    throw new Error('loadModel: expected an object');
  }
  const m = json as Partial<TransformerModel>;
  if (!m.config || !Array.isArray(m.tok_emb) || !Array.isArray(m.pos_emb)) {
    throw new Error('loadModel: missing config/tok_emb/pos_emb');
  }
  if (!Array.isArray(m.lnf_g) || !Array.isArray(m.lnf_b) || !Array.isArray(m.layers)) {
    throw new Error('loadModel: missing lnf_g/lnf_b/layers');
  }
  const { vocab_size, d_model, n_head, n_layer } = m.config;
  if (
    typeof vocab_size !== 'number' ||
    typeof d_model !== 'number' ||
    typeof n_head !== 'number' ||
    typeof n_layer !== 'number'
  ) {
    throw new Error('loadModel: config missing numeric fields');
  }
  if (m.layers.length !== n_layer) {
    throw new Error(`loadModel: expected ${n_layer} layers, got ${m.layers.length}`);
  }
  return m as TransformerModel;
}

// ---------------------------------------------------------------- matrix helpers

/** row-vector convention: (T, In) @ (In, Out) -> (T, Out) */
function matmul(x: number[][], w: number[][]): number[][] {
  const T = x.length;
  const inDim = w.length;
  const outDim = w[0].length;
  const out: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const xt = x[t];
    const row = new Array(outDim).fill(0);
    for (let i = 0; i < inDim; i++) {
      const xi = xt[i];
      if (xi === 0) continue;
      const wi = w[i];
      for (let o = 0; o < outDim; o++) {
        row[o] += xi * wi[o];
      }
    }
    out[t] = row;
  }
  return out;
}

function addBias(x: number[][], b: number[]): number[][] {
  return x.map((row) => row.map((v, i) => v + b[i]));
}

function addMatrix(a: number[][], b: number[][]): number[][] {
  return a.map((row, t) => row.map((v, i) => v + b[t][i]));
}

function layerNorm(x: number[][], g: number[], b: number[]): number[][] {
  return x.map((row) => {
    const d = row.length;
    let mu = 0;
    for (const v of row) mu += v;
    mu /= d;
    let variance = 0;
    for (const v of row) variance += (v - mu) * (v - mu);
    variance /= d;
    const inv = 1 / Math.sqrt(variance + EPS);
    return row.map((v, i) => g[i] * (v - mu) * inv + b[i]);
  });
}

function relu(x: number[][]): number[][] {
  return x.map((row) => row.map((v) => Math.max(0, v)));
}

// -------------------------------------------------------------------- attention

/** Per-layer causal multi-head attention. Returns the attention output (T, D) and,
 *  for teaching/inspection, the per-head softmax weights (H, T, T). Heads split the
 *  D-dim q/k/v vectors into contiguous chunks of size hd = D / H: head h owns
 *  columns [h*hd, (h+1)*hd). */
function attention(
  xn: number[][],
  layer: LayerWeights,
  nHead: number,
): { out: number[][]; attn: number[][][] } {
  const T = xn.length;
  const D = xn[0].length;
  const hd = D / nHead;

  const q = matmul(xn, layer.wq);
  const k = matmul(xn, layer.wk);
  const v = matmul(xn, layer.wv);

  const scale = 1 / Math.sqrt(hd);
  const headOut: number[][][] = []; // [head][t][hd]
  const headAttn: number[][][] = []; // [head][t][k]

  for (let h = 0; h < nHead; h++) {
    const lo = h * hd;
    const hi = lo + hd;
    const attnRows: number[][] = new Array(T);
    const outRows: number[][] = new Array(T);
    for (let t = 0; t < T; t++) {
      // causal: only keys 0..t are visible
      const scores = new Array(t + 1);
      let maxScore = -Infinity;
      for (let kk = 0; kk <= t; kk++) {
        let s = 0;
        for (let c = lo; c < hi; c++) {
          s += q[t][c] * k[kk][c];
        }
        s *= scale;
        scores[kk] = s;
        if (s > maxScore) maxScore = s;
      }
      let sum = 0;
      const weights = new Array(T).fill(0);
      for (let kk = 0; kk <= t; kk++) {
        const e = Math.exp(scores[kk] - maxScore);
        weights[kk] = e;
        sum += e;
      }
      for (let kk = 0; kk <= t; kk++) weights[kk] /= sum;
      attnRows[t] = weights;

      const outRow = new Array(hd).fill(0);
      for (let kk = 0; kk <= t; kk++) {
        const w = weights[kk];
        if (w === 0) continue;
        for (let c = 0; c < hd; c++) {
          outRow[c] += w * v[kk][lo + c];
        }
      }
      outRows[t] = outRow;
    }
    headOut.push(outRows);
    headAttn.push(attnRows);
  }

  // concat heads back into (T, D), contiguous per head
  const concat: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const row = new Array(D);
    for (let h = 0; h < nHead; h++) {
      const src = headOut[h][t];
      for (let c = 0; c < hd; c++) row[h * hd + c] = src[c];
    }
    concat[t] = row;
  }

  const out = matmul(concat, layer.wo);
  return { out, attn: headAttn };
}

function mlp(xn: number[][], layer: LayerWeights): number[][] {
  const z = addBias(matmul(xn, layer.w1), layer.b1);
  const r = relu(z);
  return addBias(matmul(r, layer.w2), layer.b2);
}

/** logits = h @ tok_embᵀ (tied output projection) */
function unembed(h: number[][], tokEmb: number[][]): number[][] {
  const T = h.length;
  const V = tokEmb.length;
  const D = tokEmb[0].length;
  const out: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const ht = h[t];
    const row = new Array(V);
    for (let vv = 0; vv < V; vv++) {
      const emb = tokEmb[vv];
      let s = 0;
      for (let c = 0; c < D; c++) s += ht[c] * emb[c];
      row[vv] = s;
    }
    out[t] = row;
  }
  return out;
}

export function forward(
  model: TransformerModel,
  ids: number[],
): { logits: number[][]; attention: number[][][][] } {
  const { tok_emb, pos_emb, layers, lnf_g, lnf_b, config } = model;

  let x: number[][] = ids.map((id, t) => tok_emb[id].map((v, i) => v + pos_emb[t][i]));

  const allAttn: number[][][][] = [];
  for (const layer of layers) {
    const n1 = layerNorm(x, layer.ln1_g, layer.ln1_b);
    const { out: attnOut, attn } = attention(n1, layer, config.n_head);
    x = addMatrix(x, attnOut);
    allAttn.push(attn);

    const n2 = layerNorm(x, layer.ln2_g, layer.ln2_b);
    const mlpOut = mlp(n2, layer);
    x = addMatrix(x, mlpOut);
  }

  const hf = layerNorm(x, lnf_g, lnf_b);
  const logits = unembed(hf, tok_emb);

  return { logits, attention: allAttn };
}

// -------------------------------------------------------------------- sampling

/** Temperature-scaled softmax over raw logits. */
export function softmaxTemp(logits: number[], temperature: number): number[] {
  const scaled = logits.map((v) => v / temperature);
  const maxV = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - maxV));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/** Inverse-CDF sampling from a discrete probability distribution. */
export function sampleNext(probs: number[], rand: Rng): number {
  const r = rand();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}
