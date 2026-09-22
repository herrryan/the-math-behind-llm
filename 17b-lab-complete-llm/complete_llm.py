# =====================================================================
# Stage 4: The Complete LLM (300 Lines of Pure Python)
# Dependencies: Zero external libraries (Python built-in standard library only)
#
# Architectural Milestone:
# - Full Production Transformer Architecture
# - Key-Value (KV) Cache Acceleration (Prefill vs. Decoding Phase)
# - Temperature, Top-k, and Top-p (Nucleus) Sampling Suite (Chapter 18)
# - Interactive Terminal Generation Engine
# =====================================================================
import math
import random
import time

# =====================================================================
# 1. Corpus, Vocabulary & Training Dataset Setup
# =====================================================================
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

V = len(vocab)          # Vocabulary size |V|
d_model = 12            # Residual stream dimension
d_ffn = 24              # SwiGLU expansion dimension
lr = 0.15               # Learning rate
scale = 1.0 / math.sqrt(d_model)

dataset = []
for s in corpus:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

# =====================================================================
# 2. Linear Algebra & Activation Primitives
# =====================================================================
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

# =====================================================================
# 3. Parameter Initialization
# =====================================================================
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

# =====================================================================
# 4. Training Loop: 250 Epochs
# =====================================================================
print(f"Vocabulary size |V|: {V}")
print(f"Sample sentences: {len(corpus)}")
print("\nTraining Complete LLM for 250 Epochs...")

for epoch in range(251):
    total_loss = 0.0

    for inputs, targets in dataset:
        T = len(inputs)
        X0 = [params["E"][idx][:] for idx in inputs]

        # 1. Pre-Attention RMSNorm
        X0_norm, rms1 = rmsnorm_forward(X0, params["gamma1"])

        # 2. Causal Self-Attention
        Q = matmul(X0_norm, params["W_q"])
        K = matmul(X0_norm, params["W_k"])
        V_mat = matmul(X0_norm, params["W_v"])

        scores = matmul(Q, transpose(K))
        for i in range(T):
            for j in range(T):
                scores[i][j] *= scale
                if j > i:
                    scores[i][j] = -1e9

        A = [softmax_row(scores[i]) for i in range(T)]
        O_raw = matmul(A, V_mat)
        Attn_out = matmul(O_raw, params["W_o"])

        # 3. Residual 1
        X1 = [[X0[i][j] + Attn_out[i][j] for j in range(d_model)] for i in range(T)]

        # 4. Pre-FFN RMSNorm
        X1_norm, rms2 = rmsnorm_forward(X1, params["gamma2"])

        # 5. SwiGLU FFN
        H_gate = matmul(X1_norm, params["W_gate"])
        H_up   = matmul(X1_norm, params["W_up"])
        H_silu = [[silu(H_gate[i][j]) for j in range(d_ffn)] for i in range(T)]
        H_swiglu = [[H_silu[i][j] * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        FFN_out = matmul(H_swiglu, params["W_down"])

        # 6. Residual 2
        X2 = [[X1[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

        # 7. Final RMSNorm
        X2_norm, rms_f = rmsnorm_forward(X2, params["gamma_final"])

        # 8. LM Head & Loss
        Z = matmul(X2_norm, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]
        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # Backward Pass
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
                if j <= i:
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

        # Parameter updates
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

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Average Loss: {total_loss / len(dataset):.4f}")

# =====================================================================
# 5. Production KV Cache Inference Engine
# =====================================================================
class KVCache:
    """Stores historical Key and Value vectors to enable O(1) step latency."""
    def __init__(self):
        self.k_cache = []  # List of vectors [d_model]
        self.v_cache = []  # List of vectors [d_model]

    def reset(self):
        self.k_cache = []
        self.v_cache = []

def prefill(prompt_tokens, cache):
    """Prefills the KV Cache across all prompt tokens in parallel."""
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
            if j > i:
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
    Z = matmul(X2_norm, params["W_head"])
    return Z[-1]  # Return logits of the very last prompt token

def decode_step(tok_id, cache):
    """Executes a single token decoding step in O(1) time using cached K and V."""
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
    z = matmul(x2_norm, params["W_head"])[0]
    return z

# =====================================================================
# 6. Probabilistic Sampling Suite (Chapter 18)
# =====================================================================
def sample_next_token(logits, temperature=0.7, top_k=5, top_p=0.9):
    """Samples next token using Temperature scaling, Top-k, and Top-p (Nucleus) filtering."""
    if temperature <= 1e-4:
        return max(range(len(logits)), key=lambda i: logits[i])

    # 1. Temperature Scaling
    scaled = [l / temperature for l in logits]
    max_l = max(scaled)
    exps = [math.exp(l - max_l) for l in scaled]
    sum_e = sum(exps)
    probs = [e / sum_e for e in exps]

    # 2. Top-k Truncation
    indexed = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)
    if top_k is not None and top_k < len(indexed):
        indexed = indexed[:top_k]

    # 3. Top-p (Nucleus) Truncation
    cum = 0.0
    cutoff = len(indexed)
    for idx, (_, p) in enumerate(indexed):
        cum += p
        if cum >= top_p:
            cutoff = idx + 1
            break
    filtered = indexed[:cutoff]

    # Re-normalize remaining probability mass
    mass = sum(p for _, p in filtered)
    norm_probs = [p / mass for _, p in filtered]

    # Multinomial Draw
    r = random.random()
    acc = 0.0
    for (tok_id, _), p in zip(filtered, norm_probs):
        acc += p
        if r <= acc:
            return tok_id
    return filtered[-1][0]

# =====================================================================
# 7. Autoregressive Text Generation Demonstrations
# =====================================================================
print("\n--- Autoregressive Generation with KV Cache Acceleration ---")

cache = KVCache()

def generate_sentence(prompt, max_tokens=10, temperature=0.7, top_k=5, top_p=0.9):
    prompt_tokens = [word2id[w] for w in prompt.split()]
    tokens = prompt_tokens[:]

    # Step 1: Prefill phase
    last_logit = prefill(prompt_tokens, cache)
    next_tok = sample_next_token(last_logit, temperature, top_k, top_p)
    tokens.append(next_tok)

    # Step 2: Decoding loop
    for _ in range(max_tokens - 1):
        if id2word[tokens[-1]] == ".":
            break
        decode_logit = decode_step(tokens[-1], cache)
        next_tok = sample_next_token(decode_logit, temperature, top_k, top_p)
        tokens.append(next_tok)

    return " ".join(id2word[t] for t in tokens)

for prompt in [
    "why does the sun shine ?",
    "why is the sky blue ?",
    "what do plants eat ?",
    "where do birds fly ?",
    "who made the world ?"
]:
    print(f"\nPrompt: '{prompt}'")
    # Greedy (T=0)
    greedy_out = generate_sentence(prompt, max_tokens=10, temperature=0.0)
    print(f"  [Greedy T=0.0] -> {greedy_out}")
    # Nucleus Sampling (T=0.7, Top-p=0.9)
    sample_out = generate_sentence(prompt, max_tokens=10, temperature=0.7, top_k=5, top_p=0.9)
    print(f"  [Sample T=0.7] -> {sample_out}")

print("\n" + "=" * 60)
print("Complete LLM Engine Built Successfully!")
print("=" * 60)
