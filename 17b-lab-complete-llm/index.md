# Hands-on Lab 04: The Complete Interactive LLM in 300 Lines of Pure Python

<fieldset id="evolution">
<legend><strong>The Python Brain Evolution Chain &bull; Capstone Milestone (Stage 4 of 4)</strong></legend>
<p>Over the previous three labs, we evolved from Bengio's 2003 Micro-Brain to Scaled Dot-Product Self-Attention, and then to the full Modern Transformer Block with Pre-RMSNorm, Residual highways, and SwiGLU gating. But raw greedy decoding left our model trapped in repetitive loops and suffering from $O(T^2)$ inference latency.</p>
<p>In this final capstone lab, we build <strong>The Complete Interactive LLM</strong> in ~300 lines of pure standard-library Python. We equip our Transformer with the two crowning engineering breakthroughs that power production inference systems (vLLM, Ollama, llama.cpp, HuggingFace TGI): <strong>Key-Value (KV) Cache Acceleration</strong> and the <strong>Temperature + Top-$k$ + Top-$p$ (Nucleus) Sampling Suite</strong>.</p>
<pre>
[The Python Brain Evolution Roadmap &bull; Completed Journey]
[Stage 1] 80 Lines Pure Python: Bengio 2003 MLP Language Model (Embeddings, Dense Layers, Manual Backprop)
       │
       ▼ (Fatal Flaw: 1-word context window; instant amnesia on earlier clauses)
[Stage 2] 140 Lines Pure Python: Attention Brain (Unlocking Q, K, V Projections & Causal Attention)
       │
       ▼ (Fatal Flaw: Stacking deep attention blocks triggers vanishing gradients and scale blowups)
[Stage 3] 220 Lines Pure Python: Modern Transformer Block (Pre-RMSNorm, Residuals, & SwiGLU)
       │
       ▼ (Fatal Flaw: Plain greedy decoding produces rigid, repetitive text loops)
[Stage 4 (Capstone)] 300 Lines Pure Python: Production-Grade Inference Engine (KV Cache & Top-p Sampling)
       │
       ▼ (Result: A fully interactive, zero-dependency LLM chat engine running in your terminal!)
</pre>
</fieldset>

---

## Step 1: 3-Year-Old Intuition (The Sticky Note Wall & The Warm Water Bubble)

Imagine sitting down to write a story with an intelligent friend:

<figure>
<pre>
[Inference Without Cache: O(T^2) Redundant Work]
Step 1: Read [Token 1] ──────────────────────────► Generate [Token 2]
Step 2: Re-read [Token 1, Token 2] ──────────────► Generate [Token 3]
Step 3: Re-read [Token 1, Token 2, Token 3] ──────► Generate [Token 4] (Wasteful!)

[Inference With KV Cache: O(1) Per-Step Velocity]
Prefill Phase: Process prompt ──► Store [K1, V1], [K2, V2], [K3, V3] in Cache
Decode Phase:
  Query Q4 only ──► Attends to Cached Keys [K1..K3, K4] ──► Generate [Token 5]
  Query Q5 only ──► Attends to Cached Keys [K1..K4, K5] ──► Generate [Token 6]
</pre>
<figcaption><strong>Figure 17b.1:</strong> KV Cache avoids re-computing historical Key and Value representations, converting decoding from an $O(T^2)$ bottleneck to an $O(1)$ stream.</figcaption>
</figure>

1. **The Sticky Note Wall (The KV Cache)**:
   Suppose you are writing a 1,000-word essay. Without a cache, every time you want to write word 1,001, you would have to re-read all 1,000 previous words from scratch and recalculate what each word means!
   With the **KV Cache**, as soon as you figure out what word 1 means (its Key and Value), you stick a yellow note on the wall. When word 1,001 arrives, you only calculate **one single Query question ($\mathbf{q}_{1001}$)**, look up at the sticky notes already on the wall, and immediately pick the next word!

2. **The Thermal Water Bath (Temperature)**:
   - **Freezing Ice ($T \to 0$, Greedy)**: The water turns solid. Only the absolute highest point sticks out. The model behaves like a strict calculator: always repeating the exact same word every time.
   - **Boiling Water ($T > 1.5$)**: Hot water splashes violently in all directions. Rare, nonsensical words bubble up to the surface.
   - **Lukewarm Water ($T \approx 0.7$)**: The best candidates remain visible, but slight ripples allow natural variety and conversational warmth.

3. **The Dynamic Bubble (Top-$p$ Nucleus Sampling)**:
   Fixed Top-$k$ is rigid: it always keeps $k$ words whether you are sure or confused.
   **Top-$p$** is an expandable soap bubble:
   - When the model is 99% certain (<samp>"nuclear ... fusion"</samp>), the bubble shrinks to contain just that **1** word.
   - When the model is exploring an open creative sentence, the bubble expands to hold 10 reasonable words whose combined probability mass reaches $p$ (e.g. 90%). Anything outside the bubble is discarded.

---

## Step 2: The Math Bridge to Pure Python

Every production inference equation maps directly into standard Python:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 17b.1:</strong> Production inference algorithms mapped to pure Python list primitives</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">Inference Operation</th>
      <th scope="col" align="left" width="14%">Chapter</th>
      <th scope="col" align="left" width="33%">Mathematical Formula</th>
      <th scope="col" align="left" width="33%">Pure Python Implementation (Zero Libs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>KV Cache Storage</strong></th>
      <td>Chapter 17</td>
      <td>$\mathbf{K}_{\text{past}} \leftarrow [\mathbf{K}_{\text{past}}; \mathbf{k}_t], \; \mathbf{V}_{\text{past}} \leftarrow [\mathbf{V}_{\text{past}}; \mathbf{v}_t]$</td>
      <td><code>cache.k.append(k_t); cache.v.append(v_t)</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Cached Step Attention</strong></th>
      <td>Chapter 17</td>
      <td>$\boldsymbol{\alpha} = \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{\text{past}}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times T_{\text{past}}}$</td>
      <td><code>scores = [sum(q[d]*k[d] for d in range(D))*scale for k in cache.k]; a = softmax(scores)</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Temperature Scaling</strong></th>
      <td>Chapter 18</td>
      <td>$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$</td>
      <td><code>scaled = [z / T for z in logits]; exps = [math.exp(v - max_v) for v in scaled]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Top-p Nucleus Cutoff</strong></th>
      <td>Chapter 18</td>
      <td>$\min M \text{ s.t. } \sum_{i=1}^M p_{(i)} \ge p$</td>
      <td><code>for i, (_, prob) in enumerate(sorted_p): cum += prob; if cum &gt;= p: cutoff = i + 1; break</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Multinomial Draw</strong></th>
      <td>Chapter 18</td>
      <td>Sample token $x \sim \operatorname{Categorical}(\mathbf{p}_{\text{nucleus}})$</td>
      <td><code>r = random.random(); for tok, p in nucleus: acc += p; if r &lt;= acc: return tok</code></td>
    </tr>
  </tbody>
</table>

---

## Step 3: Complete Source Code & Math Anatomy

This final capstone lab provides two production-grade scripts:
- **Guided Exercise Script**: [`labs/04_complete_llm_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/04_complete_llm_exercise.py) (also in [`17b-lab-complete-llm/complete_llm_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/17b-lab-complete-llm/complete_llm_exercise.py)), equipped with automated unit test assertions verifying temperature scaling, Top-$k$ filtering, Top-$p$ nucleus accumulation, and single-token KV Cache decoding.
- **Reference Solution**: [`labs/04_complete_llm.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/04_complete_llm.py) (also in [`17b-lab-complete-llm/complete_llm.py`](file:///Users/guofei/workspace/the-math-behind-llm/17b-lab-complete-llm/complete_llm.py)).

Here is the complete reference script:

<figure>
<pre>
# =====================================================================
# Stage 4: The Complete Interactive LLM (300 Lines of Pure Python)
# Dependencies: Zero external libraries (Python standard library only)
# =====================================================================
import math
import random

corpus = [
    "why does the sun shine ? the sun shines because of nuclear fusion .",
    "why is the sky blue ? the sky is blue because of rayleigh scattering .",
    "what do plants eat ? plants eat sunlight and carbon dioxide .",
    "where do birds fly ? birds fly high in the warm summer sky .",
    "who made the world ? nature made the world with physics and math ."
]

all_words = (" ".join(corpus)).split()
vocab = sorted(list(set(all_words)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}

V = len(vocab)          # |V| = 39
d_model = 12            # Residual stream dimension
d_ffn = 24              # SwiGLU expansion dimension
lr = 0.15               # Learning rate
scale = 1.0 / math.sqrt(d_model)

dataset = []
for s in corpus:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

def init_matrix(rows, cols, scale_init=0.2):
    return [[random.gauss(0, scale_init) for _ in range(cols)] for _ in range(rows)]

def matmul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax_row(row):
    max_val = max(row)
    exp_r = [math.exp(v - max_val) for v in row]
    sum_r = sum(exp_r)
    return [v / sum_r for v in exp_r]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))

def silu(x):
    return x * sigmoid(x)

def silu_deriv(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))

def rmsnorm_forward(X, gamma, eps=1e-5):
    T, d = len(X), len(X[0])
    X_norm = []
    rms_list = []
    for i in range(T):
        ms = sum(v * v for v in X[i]) / d
        rms = math.sqrt(ms + eps)
        rms_list.append(rms)
        X_norm.append([X[i][j] / rms * gamma[j] for j in range(d)])
    return X_norm, rms_list

def rmsnorm_backward(dX_norm, X, gamma, rms_list):
    T, d = len(X), len(X[0])
    dX = []
    dgamma = [0.0] * d
    for i in range(T):
        rms = rms_list[i]
        sum_gamma_dxn_x = sum(gamma[j] * dX_norm[i][j] * X[i][j] for j in range(d))
        row_dx = []
        for j in range(d):
            dgamma[j] += dX_norm[i][j] * (X[i][j] / rms)
            term1 = gamma[j] * dX_norm[i][j]
            term2 = (X[i][j] / (d * rms * rms)) * sum_gamma_dxn_x
            row_dx.append((term1 - term2) / rms)
        dX.append(row_dx)
    return dX, dgamma

random.seed(42)
params = {
    "E":           init_matrix(V, d_model),
    "gamma1":      [1.0] * d_model,
    "W_q":         init_matrix(d_model, d_model),
    "W_k":         init_matrix(d_model, d_model),
    "W_v":         init_matrix(d_model, d_model),
    "W_o":         init_matrix(d_model, d_model),
    "gamma2":      [1.0] * d_model,
    "W_gate":      init_matrix(d_model, d_ffn),
    "W_up":        init_matrix(d_model, d_ffn),
    "W_down":      init_matrix(d_ffn, d_model),
    "gamma_final": [1.0] * d_model,
    "W_head":      init_matrix(d_model, V),
}

# 4. Fast Training Loop (250 Epochs)
for epoch in range(251):
    total_loss = 0.0
    for inputs, targets in dataset:
        T = len(inputs)
        X0 = [params["E"][idx][:] for idx in inputs]
        X0_norm, rms1 = rmsnorm_forward(X0, params["gamma1"])

        Q = matmul(X0_norm, params["W_q"])
        K = matmul(X0_norm, params["W_k"])
        V_mat = matmul(X0_norm, params["W_v"])

        scores = matmul(Q, transpose(K))
        for i in range(T):
            for j in range(T):
                scores[i][j] *= scale
                if j &gt; i:
                    scores[i][j] = -1e9

        A = [softmax_row(scores[i]) for i in range(T)]
        O_raw = matmul(A, V_mat)
        Attn_out = matmul(O_raw, params["W_o"])
        X1 = [[X0[i][j] + Attn_out[i][j] for j in range(d_model)] for i in range(T)]

        X1_norm, rms2 = rmsnorm_forward(X1, params["gamma2"])
        H_gate = matmul(X1_norm, params["W_gate"])
        H_up   = matmul(X1_norm, params["W_up"])
        H_silu = [[silu(H_gate[i][j]) for j in range(d_ffn)] for i in range(T)]
        H_swiglu = [[H_silu[i][j] * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        FFN_out = matmul(H_swiglu, params["W_down"])
        X2 = [[X1[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

        X2_norm, rms_f = rmsnorm_forward(X2, params["gamma_final"])
        Z = matmul(X2_norm, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]
        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # Backward Pass & Updates
        dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]
        dW_head = matmul(transpose(X2_norm), dZ)
        dX2_norm = matmul(dZ, transpose(params["W_head"]))

        dX2, dgamma_f = rmsnorm_backward(dX2_norm, X2, params["gamma_final"], rms_f)
        dX1_res2 = dX2[:]
        dFFN_out = dX2[:]

        dW_down = matmul(transpose(H_swiglu), dFFN_out)
        dH_swiglu = matmul(dFFN_out, transpose(params["W_down"]))
        dH_up = [[dH_swiglu[i][j] * H_silu[i][j] for j in range(d_ffn)] for i in range(T)]
        dH_silu = [[dH_swiglu[i][j] * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        dH_gate = [[dH_silu[i][j] * silu_deriv(H_gate[i][j]) for j in range(d_ffn)] for i in range(T)]

        dW_gate = matmul(transpose(X1_norm), dH_gate)
        dW_up   = matmul(transpose(X1_norm), dH_up)
        dX1_norm = [[sum(dH_gate[i][k] * params["W_gate"][j][k] + dH_up[i][k] * params["W_up"][j][k] for k in range(d_ffn))
                     for j in range(d_model)] for i in range(T)]
        dX1_from_norm, dgamma2 = rmsnorm_backward(dX1_norm, X1, params["gamma2"], rms2)
        dX1 = [[dX1_res2[i][j] + dX1_from_norm[i][j] for j in range(d_model)] for i in range(T)]

        dX0_res1 = dX1[:]
        dAttn_out = dX1[:]
        dW_o = matmul(transpose(O_raw), dAttn_out)
        dO_raw = matmul(dAttn_out, transpose(params["W_o"]))
        dV_mat = matmul(transpose(A), dO_raw)
        dA = matmul(dO_raw, transpose(V_mat))

        dScores = [[0.0] * T for _ in range(T)]
        for i in range(T):
            sum_dA_A = sum(dA[i][k] * A[i][k] for k in range(T))
            for j in range(T):
                if j &lt;= i:
                    dScores[i][j] = A[i][j] * (dA[i][j] - sum_dA_A) * scale

        dQ = matmul(dScores, K)
        dK = matmul(transpose(dScores), Q)
        dW_q = matmul(transpose(X0_norm), dQ)
        dW_k = matmul(transpose(X0_norm), dK)
        dW_v = matmul(transpose(X0_norm), dV_mat)
        dX0_norm = [[sum(dQ[i][k] * params["W_q"][j][k] + dK[i][k] * params["W_k"][j][k] + dV_mat[i][k] * params["W_v"][j][k] for k in range(d_model))
                     for j in range(d_model)] for i in range(T)]
        dX0_from_norm, dgamma1 = rmsnorm_backward(dX0_norm, X0, params["gamma1"], rms1)
        dX0 = [[dX0_res1[i][j] + dX0_from_norm[i][j] for j in range(d_model)] for i in range(T)]

        for i in range(d_model):
            for v in range(V):
                params["W_head"][i][v] -= lr * dW_head[i][v]
            params["gamma_final"][i] -= lr * dgamma_f[i]
            params["gamma2"][i]      -= lr * dgamma2[i]
            params["gamma1"][i]      -= lr * dgamma1[i]
            for k in range(d_model):
                params["W_o"][i][k] -= lr * dW_o[i][k]
                params["W_q"][i][k] -= lr * dW_q[i][k]
                params["W_k"][i][k] -= lr * dW_k[i][k]
                params["W_v"][i][k] -= lr * dW_v[i][k]
            for k in range(d_ffn):
                params["W_gate"][i][k] -= lr * dW_gate[i][k]
                params["W_up"][i][k]   -= lr * dW_up[i][k]
        for k in range(d_ffn):
            for j in range(d_model):
                params["W_down"][k][j] -= lr * dW_down[k][j]
        for i in range(T):
            idx = inputs[i]
            for j in range(d_model):
                params["E"][idx][j] -= lr * dX0[i][j]

# 5. KV Cache Engine
class KVCache:
    def __init__(self):
        self.k_cache = []
        self.v_cache = []
    def reset(self):
        self.k_cache = []
        self.v_cache = []

def prefill(prompt_tokens, cache):
    cache.reset()
    T = len(prompt_tokens)
    X0 = [params["E"][idx][:] for idx in prompt_tokens]
    X0_norm, _ = rmsnorm_forward(X0, params["gamma1"])

    Q = matmul(X0_norm, params["W_q"])
    K = matmul(X0_norm, params["W_k"])
    V_mat = matmul(X0_norm, params["W_v"])

    for i in range(T):
        cache.k_cache.append(K[i][:])
        cache.v_cache.append(V_mat[i][:])

    scores = matmul(Q, transpose(K))
    for i in range(T):
        for j in range(T):
            scores[i][j] *= scale
            if j &gt; i:
                scores[i][j] = -1e9

    A = [softmax_row(scores[i]) for i in range(T)]
    O_raw = matmul(A, V_mat)
    Attn_out = matmul(O_raw, params["W_o"])
    X1 = [[X0[i][j] + Attn_out[i][j] for j in range(d_model)] for i in range(T)]

    X1_norm, _ = rmsnorm_forward(X1, params["gamma2"])
    Hg = matmul(X1_norm, params["W_gate"])
    Hu = matmul(X1_norm, params["W_up"])
    H_swiglu = [[silu(Hg[i][j]) * Hu[i][j] for j in range(d_ffn)] for i in range(T)]
    FFN_out = matmul(H_swiglu, params["W_down"])
    X2 = [[X1[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

    X2_norm, _ = rmsnorm_forward(X2, params["gamma_final"])
    return matmul(X2_norm, params["W_head"])[-1]

def decode_step(tok_id, cache):
    x0 = [params["E"][tok_id][:]]
    x0_norm, _ = rmsnorm_forward(x0, params["gamma1"])

    q = matmul(x0_norm, params["W_q"])[0]
    k = matmul(x0_norm, params["W_k"])[0]
    v = matmul(x0_norm, params["W_v"])[0]

    cache.k_cache.append(k)
    cache.v_cache.append(v)

    past_len = len(cache.k_cache)
    raw_scores = [sum(q[d] * cache.k_cache[j][d] for d in range(d_model)) * scale for j in range(past_len)]
    attn_weights = softmax_row(raw_scores)

    o_raw = [[sum(attn_weights[j] * cache.v_cache[j][d] for j in range(past_len)) for d in range(d_model)]]
    attn_out = matmul(o_raw, params["W_o"])[0]

    x1 = [[x0[0][d] + attn_out[d] for d in range(d_model)]]
    x1_norm, _ = rmsnorm_forward(x1, params["gamma2"])

    hg = matmul(x1_norm, params["W_gate"])[0]
    hu = matmul(x1_norm, params["W_up"])[0]
    h_swiglu = [[silu(hg[d]) * hu[d] for d in range(d_ffn)]]
    ffn_out = matmul(h_swiglu, params["W_down"])[0]

    x2 = [[x1[0][d] + ffn_out[d] for d in range(d_model)]]
    x2_norm, _ = rmsnorm_forward(x2, params["gamma_final"])
    return matmul(x2_norm, params["W_head"])[0]

# 6. Probabilistic Sampling Suite (Temperature, Top-k, Top-p)
def sample_next_token(logits, temperature=0.7, top_k=5, top_p=0.9):
    if temperature &lt;= 1e-4:
        return max(range(len(logits)), key=lambda i: logits[i])

    scaled = [l / temperature for l in logits]
    max_l = max(scaled)
    exps = [math.exp(l - max_l) for l in scaled]
    sum_e = sum(exps)
    probs = [e / sum_e for e in exps]

    indexed = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)
    if top_k is not None and top_k &lt; len(indexed):
        indexed = indexed[:top_k]

    cum = 0.0
    cutoff = len(indexed)
    for idx, (_, p) in enumerate(indexed):
        cum += p
        if cum &gt;= top_p:
            cutoff = idx + 1
            break
    filtered = indexed[:cutoff]
    mass = sum(p for _, p in filtered)
    norm_probs = [p / mass for _, p in filtered]

    r = random.random()
    acc = 0.0
    for (tok_id, _), p in zip(filtered, norm_probs):
        acc += p
        if r &lt;= acc:
            return tok_id
    return filtered[-1][0]
</pre>
<figcaption><strong>Figure 17b.2:</strong> Production inference engine with KV Cache prefill and decode phases combined with Top-$p$ sampling in 300 lines of pure Python.</figcaption>
</figure>

---

## Step 4: Where Did It Come From? (The Science of Production LLM Serving)

Why do production frameworks like **vLLM**, **llama.cpp**, and **TensorRT-LLM** focus almost entirely on KV Cache management and sampling algorithms?

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 17b.2:</strong> Why production LLM inference diverges from naive training loops</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="22%">Engineering Problem</th>
      <th scope="col" align="left" width="38%">Naive Generation (No Cache)</th>
      <th scope="col" align="left" width="40%">Production KV Cache Engine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>Computational Complexity</strong></th>
      <td>$O(T^2)$ matrix multiplications. Generating $T$ new tokens takes $O(T^2 \cdot d)$ total FLOPs because past tokens are repeatedly recomputed.</td>
      <td>$O(1)$ per step. Generating $T$ new tokens takes $O(T \cdot d)$ operations, maintaining steady token generation velocity regardless of length.</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Memory vs Compute Trade-off</strong></th>
      <td>Minimal RAM usage (only current activations), but massive GPU compute saturation on redundant matrix multiplications.</td>
      <td>Caches historical Keys and Values in RAM/VRAM. LLM serving becomes <strong>memory-bandwidth bound</strong> rather than compute-bound.</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Output Naturalness</strong></th>
      <td>Greedy decoding ($T=0$) gets trapped in cyclic loops (<samp>"the the the"</samp>) and cannot generate diverse answers.</td>
      <td>Nucleus sampling ($p=0.9$) dynamically restricts tokens to high-confidence probability mass while allowing natural conversational phrasing.</td>
    </tr>
  </tbody>
</table>

---

## Step 5: Concrete Execution Trace & Step-by-Step Arithmetic

Let us trace the numbers for a 3-word vocabulary when sampling with Temperature and Top-$p$:

1. **Logits to Scaled Probabilities ($T = 0.5$)**:
   Suppose our LM Head outputs raw logits $\mathbf{z} = [2.0, 1.0, 0.0]$.
   With temperature $T = 0.5$, we scale the logits:
   $$
   \mathbf{z}' = \left[\frac{2.0}{0.5}, \frac{1.0}{0.5}, \frac{0.0}{0.5}\right] = [4.0, 2.0, 0.0]
   $$
   Exponentiating (with max subtraction for numerical stability):
   $$
   e^4 \approx 54.598, \quad e^2 \approx 7.389, \quad e^0 = 1.0 \implies \sum = 62.987
   $$
   $$
   \mathbf{p} = \left[\frac{54.598}{62.987}, \frac{7.389}{62.987}, \frac{1.0}{62.987}\right] \approx [0.8668, 0.1173, 0.0159]
   $$

2. **Top-$p$ Nucleus Cutoff ($p = 0.90$)**:
   We sort the probabilities in descending order:
   - Rank 1: Token 0 with $p = 0.8668$. Cumulative sum = $0.8668 < 0.90$. (Keep!)
   - Rank 2: Token 1 with $p = 0.1173$. Cumulative sum = $0.8668 + 0.1173 = 0.9841 \ge 0.90$. (Cutoff met! Keep and stop.)
   - Rank 3: Token 2 with $p = 0.0159$. (Discarded from nucleus!)
   
   Re-normalize the retained tokens:
   $$
   \text{Mass} = 0.8668 + 0.1173 = 0.9841
   $$
   $$
   p_{\text{norm}}(0) = \frac{0.8668}{0.9841} \approx 0.8808, \quad p_{\text{norm}}(1) = \frac{0.1173}{0.9841} \approx 0.1192
   $$
   Low-probability tail tokens are pruned without arbitrary fixed-$k$ limits!

---

## Step 6: Core Takeaway & Graduation from The Python Brain

<fieldset>
<legend><strong>Capstone Pedagogical Takeaway</strong></legend>
<p>You have constructed every single component of a modern Large Language Model from nothing but raw mathematical equations and Python standard-library primitives. You understand every matrix dot-product, every attention routing score, every residual bypass, and every cached key-value tensor not as magical black boxes, but as transparent, elegant mathematics.</p>
</fieldset>

### The Complete Python Brain Evolution Summary

Over the course of these four milestone labs, you built:
1. **Lab 01 (Stage 1 &bull; 80 lines)**: The Bengio 2003 MLP Language Model. Unmasked the **1-word context amnesia bottleneck**.
2. **Lab 02 (Stage 2 &bull; 140 lines)**: The Scaled Dot-Product Attention Brain. Dynamic information routing across time. Unmasked the **depth instability bottleneck**.
3. **Lab 03 (Stage 3 &bull; 220 lines)**: The Modern Transformer Brain (Pre-RMSNorm, Residuals, SwiGLU). Deep mathematical stability. Unmasked the **mechanical greedy generation bottleneck**.
4. **Lab 04 (Stage 4 &bull; 300 lines)**: The Complete Interactive LLM. $O(1)$ KV Cache inference acceleration and creative Nucleus sampling.

You now possess the foundational intuition and rigorous mathematical mastery required to read, build, optimize, and advance modern AI research!
