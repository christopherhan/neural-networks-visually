"""Train the site's tiny transformer language model (numpy, CPU, no deps).

GPT-style decoder: pre-LayerNorm, causal multi-head attention, ReLU MLP,
learned positional embeddings, tied input/output embeddings. Small on
purpose — the exported weights ship to the browser as JSON and power the
Part III figures (attention lens, generation lab, embedding map).

Everything is deterministic (seeded). Run bpe.py first.

Outputs (in this directory; the deploy step copies them to public/models/):
  weights.json      model config + all parameters (5 decimal places)
  ref_logits.json   parity reference for the TypeScript inference port
  emb2d.json        PCA projection of token embeddings for the ch9 map
"""

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# ---------------------------------------------------------------- config
SEED = 1337
D = 64          # model width
H = 4           # attention heads
L = 2           # transformer layers
T = 64          # context length
MLP = 256       # hidden width of the feed-forward
STEPS = 6000
BATCH = 24
LR = 1e-3
WARMUP = 100
CLIP = 1.0
EPS = 1e-5

rng = np.random.default_rng(SEED)

tokenizer = json.loads((HERE / "tokenizer.json").read_text())
VOCAB = tokenizer["vocab"]
V = len(VOCAB)
data = np.array(json.loads((HERE / "corpus_ids.json").read_text()), dtype=np.int32)
split = int(len(data) * 0.97)
train_data, val_data = data[:split], data[split:]
print(f"V={V} train={len(train_data):,} val={len(val_data):,} tokens")

# ---------------------------------------------------------------- params
def init(shape, scale=0.02):
    return (rng.standard_normal(shape) * scale).astype(np.float64)

params = {"tok_emb": init((V, D)), "pos_emb": init((T, D)),
          "lnf_g": np.ones(D), "lnf_b": np.zeros(D)}
for i in range(L):
    params.update({
        f"l{i}_ln1_g": np.ones(D), f"l{i}_ln1_b": np.zeros(D),
        f"l{i}_wq": init((D, D)), f"l{i}_wk": init((D, D)),
        f"l{i}_wv": init((D, D)), f"l{i}_wo": init((D, D), 0.02 / np.sqrt(2 * L)),
        f"l{i}_ln2_g": np.ones(D), f"l{i}_ln2_b": np.zeros(D),
        f"l{i}_w1": init((D, MLP)), f"l{i}_b1": np.zeros(MLP),
        f"l{i}_w2": init((MLP, D), 0.02 / np.sqrt(2 * L)), f"l{i}_b2": np.zeros(D),
    })

# ------------------------------------------------------------- primitives
def layernorm_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mu) * inv
    return g * xhat + b, (xhat, inv, g)

def layernorm_bwd(dy, cache):
    xhat, inv, g = cache
    dg = (dy * xhat).sum((0, 1))
    db = dy.sum((0, 1))
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

MASK = np.triu(np.ones((T, T), dtype=bool), k=1)

def attn_fwd(xn, p, i):
    B, Tc, _ = xn.shape
    hd = D // H
    q = (xn @ p[f"l{i}_wq"]).reshape(B, Tc, H, hd).transpose(0, 2, 1, 3)
    k = (xn @ p[f"l{i}_wk"]).reshape(B, Tc, H, hd).transpose(0, 2, 1, 3)
    v = (xn @ p[f"l{i}_wv"]).reshape(B, Tc, H, hd).transpose(0, 2, 1, 3)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    s = np.where(MASK[:Tc, :Tc], -1e30, s)
    a = softmax(s)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, Tc, D)
    y = o @ p[f"l{i}_wo"]
    return y, (xn, q, k, v, a, o)

def attn_bwd(dy, cache, p, i, grads):
    xn, q, k, v, a, o = cache
    B, Hh, Tc, hd = q.shape
    grads[f"l{i}_wo"] += o.reshape(-1, D).T @ dy.reshape(-1, D)
    do = (dy @ p[f"l{i}_wo"].T).reshape(B, Tc, Hh, hd).transpose(0, 2, 1, 3)
    da = do @ v.transpose(0, 1, 3, 2)
    dv = a.transpose(0, 1, 3, 2) @ do
    ds = a * (da - (da * a).sum(-1, keepdims=True))
    ds /= np.sqrt(hd)
    dq = ds @ k
    dk = ds.transpose(0, 1, 3, 2) @ q
    dxn = np.zeros_like(xn)
    for name, dhead in (("wq", dq), ("wk", dk), ("wv", dv)):
        flat = dhead.transpose(0, 2, 1, 3).reshape(-1, D)
        grads[f"l{i}_{name}"] += xn.reshape(-1, D).T @ flat
        dxn += (flat @ p[f"l{i}_{name}"].T).reshape(B, Tc, D)
    return dxn

def mlp_fwd(xn, p, i):
    z = xn @ p[f"l{i}_w1"] + p[f"l{i}_b1"]
    r = np.maximum(z, 0)
    y = r @ p[f"l{i}_w2"] + p[f"l{i}_b2"]
    return y, (xn, z, r)

def mlp_bwd(dy, cache, p, i, grads):
    xn, z, r = cache
    grads[f"l{i}_w2"] += r.reshape(-1, MLP).T @ dy.reshape(-1, D)
    grads[f"l{i}_b2"] += dy.sum((0, 1))
    dr = dy @ p[f"l{i}_w2"].T
    dz = dr * (z > 0)
    grads[f"l{i}_w1"] += xn.reshape(-1, D).T @ dz.reshape(-1, MLP)
    grads[f"l{i}_b1"] += dz.sum((0, 1))
    return (dz @ p[f"l{i}_w1"].T)

# ------------------------------------------------------------ full model
def forward(ids, p, targets=None):
    B, Tc = ids.shape
    x = p["tok_emb"][ids] + p["pos_emb"][:Tc]
    caches = []
    for i in range(L):
        n1, c1 = layernorm_fwd(x, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        ay, ca = attn_fwd(n1, p, i)
        x = x + ay
        n2, c2 = layernorm_fwd(x, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        my, cm = mlp_fwd(n2, p, i)
        x = x + my
        caches.append((c1, ca, c2, cm))
    hf, cf = layernorm_fwd(x, p["lnf_g"], p["lnf_b"])
    logits = hf @ p["tok_emb"].T
    if targets is None:
        return logits, None
    probs = softmax(logits)
    n = B * Tc
    loss = -np.log(probs[np.arange(B)[:, None], np.arange(Tc), targets] + 1e-12).mean()
    return logits, (loss, probs, hf, cf, caches, ids, targets)

def backward(state, p):
    loss, probs, hf, cf, caches, ids, targets = state
    B, Tc = ids.shape
    n = B * Tc
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(Tc), targets] -= 1.0
    dlogits /= n
    grads["tok_emb"] += dlogits.reshape(-1, V).T @ hf.reshape(-1, D)
    dhf = dlogits @ p["tok_emb"]
    dx, dg, db = layernorm_bwd(dhf, cf)
    grads["lnf_g"] += dg
    grads["lnf_b"] += db
    for i in reversed(range(L)):
        c1, ca, c2, cm = caches[i]
        dn2 = mlp_bwd(dx, cm, p, i, grads)
        dres, dg, db = layernorm_bwd(dn2, c2)
        grads[f"l{i}_ln2_g"] += dg
        grads[f"l{i}_ln2_b"] += db
        dx = dx + dres
        dn1 = attn_bwd(dx, ca, p, i, grads)
        dres, dg, db = layernorm_bwd(dn1, c1)
        grads[f"l{i}_ln1_g"] += dg
        grads[f"l{i}_ln1_b"] += db
        dx = dx + dres
    np.add.at(grads["tok_emb"], ids, dx)
    grads["pos_emb"][:Tc] += dx.sum(0)
    return grads

# ----------------------------------------------------------- grad check
def grad_check():
    global V, D, H, L, T, MLP, MASK, params
    saved = (V, D, H, L, T, MLP, MASK, params)
    V_, D_, H_, L_, T_, MLP_ = 13, 8, 2, 2, 5, 16
    V, D, H, L, T, MLP = V_, D_, H_, L_, T_, MLP_
    MASK = np.triu(np.ones((T, T), dtype=bool), k=1)
    p = {"tok_emb": init((V, D), 0.3), "pos_emb": init((T, D), 0.3),
         "lnf_g": np.ones(D) + init(D, 0.1), "lnf_b": init(D, 0.1)}
    for i in range(L):
        p.update({
            f"l{i}_ln1_g": np.ones(D) + init(D, 0.1), f"l{i}_ln1_b": init(D, 0.1),
            f"l{i}_wq": init((D, D), 0.3), f"l{i}_wk": init((D, D), 0.3),
            f"l{i}_wv": init((D, D), 0.3), f"l{i}_wo": init((D, D), 0.3),
            f"l{i}_ln2_g": np.ones(D) + init(D, 0.1), f"l{i}_ln2_b": init(D, 0.1),
            f"l{i}_w1": init((D, MLP), 0.3), f"l{i}_b1": init(MLP, 0.1),
            f"l{i}_w2": init((MLP, D), 0.3), f"l{i}_b2": init(D, 0.1),
        })
    ids = rng.integers(0, V, (3, T))
    tgt = rng.integers(0, V, (3, T))
    _, state = forward(ids, p, tgt)
    grads = backward(state, p)
    worst = 0.0
    for name in ["tok_emb", "pos_emb", "l0_wq", "l0_wo", "l1_w1", "l1_ln1_g",
                 "lnf_g", "l0_b1", "l1_wv"]:
        flat = p[name].reshape(-1)
        gflat = grads[name].reshape(-1)
        for idx in rng.integers(0, flat.size, 4):
            h = 1e-5
            orig = flat[idx]
            flat[idx] = orig + h
            lp = forward(ids, p, tgt)[1][0]
            flat[idx] = orig - h
            lm = forward(ids, p, tgt)[1][0]
            flat[idx] = orig
            num = (lp - lm) / (2 * h)
            ana = gflat[idx]
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-8)
            worst = max(worst, rel)
    V, D, H, L, T, MLP, MASK, params = saved
    print(f"grad check worst relative error: {worst:.2e}")
    assert worst < 1e-4, "GRADIENT CHECK FAILED"

# -------------------------------------------------------------- training
def batch(source):
    ix = rng.integers(0, len(source) - T - 1, BATCH)
    x = np.stack([source[i : i + T] for i in ix])
    y = np.stack([source[i + 1 : i + T + 1] for i in ix])
    return x, y

def val_loss():
    losses = []
    for _ in range(8):
        x, y = batch(val_data)
        losses.append(forward(x, params, y)[1][0])
    return float(np.mean(losses))

def main():
    grad_check()
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v2 = {k: np.zeros_like(v) for k, v in params.items()}
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = batch(train_data)
        _, state = forward(x, params, y)
        grads = backward(state, params)
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, CLIP / (norm + 1e-9))
        lr = LR * min(1.0, step / WARMUP)
        b1, b2 = 0.9, 0.99
        for k in params:
            g = grads[k] * scale
            m[k] = b1 * m[k] + (1 - b1) * g
            v2[k] = b2 * v2[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1 ** step)
            vh = v2[k] / (1 - b2 ** step)
            params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if step % 200 == 0 or step == 1:
            print(f"step {step:5d} train {state[0]:.4f} val {val_loss():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    export()

# --------------------------------------------------------------- exports
def rounded(a):
    return np.round(a, 5).tolist()

def export():
    layers = []
    for i in range(L):
        layers.append({k[len(f"l{i}_"):]: rounded(params[k])
                       for k in params if k.startswith(f"l{i}_")})
    out = {"config": {"vocab_size": V, "d_model": D, "n_head": H,
                      "n_layer": L, "ctx": T, "mlp": MLP},
           "tok_emb": rounded(params["tok_emb"]),
           "pos_emb": rounded(params["pos_emb"]),
           "lnf_g": rounded(params["lnf_g"]), "lnf_b": rounded(params["lnf_b"]),
           "layers": layers}
    (HERE / "weights.json").write_text(json.dumps(out))
    size = (HERE / "weights.json").stat().st_size
    print(f"weights.json: {size/1e6:.2f} MB")

    # Parity reference: logits at the last position for a fixed prompt.
    ref_ids = np.array([json.loads((HERE / "tokenizer_ref.json").read_text())[1]["ids"]])
    logits, _ = forward(ref_ids, params)
    (HERE / "ref_logits.json").write_text(json.dumps({
        "ids": ref_ids[0].tolist(),
        "last_logits_first32": np.round(logits[0, -1, :32], 4).tolist(),
    }))

    # 2D embedding map for chapter 9: PCA of tok_emb, most frequent tokens.
    counts = np.bincount(data, minlength=V)
    top = np.argsort(-counts)[:250]
    E = params["tok_emb"] - params["tok_emb"].mean(0)
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    xy = E[top] @ vt[:2].T
    xy = xy / (np.abs(xy).max() + 1e-9)
    (HERE / "emb2d.json").write_text(json.dumps({
        "tokens": [VOCAB[i] for i in top],
        "xy": np.round(xy, 4).tolist(),
    }))
    print("exports complete")

if __name__ == "__main__":
    main()
