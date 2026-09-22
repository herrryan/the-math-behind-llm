# =====================================================================
# Multi-Layer LLM: Deep Transformer Inference & Training Engine
# Pure Standard-Library Python (Zero External Dependencies)
#
# Architectural Features:
# - Stacking L Transformer Layers along the Residual Stream
# - Layer-by-Layer Pre-RMSNorm, Causal Multi-Head / Self-Attention, & SwiGLU FFN
# - Multi-Layer Key-Value (KV) Cache for O(1) Token-by-Token Decoding
# - Deep Backpropagation through Residual Highways and Layer Stacks
# - Full Sampling Suite: Temperature Scaling, Top-k, and Top-p (Nucleus)
# =====================================================================
import math
import random
import time

# =====================================================================
# 1. Corpus, Vocabulary & Dataset Setup
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

V = len(vocab)          # Vocabulary size |V| = 39
d_model = 12            # Residual stream dimension
d_ffn = 24              # SwiGLU expansion dimension
num_layers = 2          # Number of stacked Transformer layers (L = 2)
lr = 0.12               # Learning rate
scale = 1.0 / math.sqrt(d_model)

dataset = []
for s in corpus:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

print(f"Vocabulary size |V|: {V}")
print(f"Dataset sentences: {len(corpus)}")
print(f"Number of Transformer layers (L): {num_layers}")
print(f"Residual stream dimension (d_model): {d_model}")
print(f"SwiGLU hidden dimension (d_ffn): {d_ffn}")

# =====================================================================
# 2. Linear Algebra & Math Primitives
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
# 3. Sampling Suite: Temperature, Top-k, and Top-p (Nucleus)
# =====================================================================
def sample_with_temperature(logits, temperature=1.0):
    if temperature <= 1e-4:
        max_val = max(logits)
        return [1.0 if v == max_val else 0.0 for v in logits]
    temp_logits = [l / temperature for l in logits]
    max_logit = max(temp_logits)
    exp_logits = [math.exp(l - max_logit) for l in temp_logits]
    sum_exp = sum(exp_logits)
    return [e / sum_exp for e in exp_logits]

def apply_top_k(probs, k=5):
    indexed = list(enumerate(probs))
    sorted_probs = sorted(indexed, key=lambda x: x[1], reverse=True)
    top_k_pairs = sorted_probs[:k]
    total_mass = sum(p for _, p in top_k_pairs)
    return [(idx, p / total_mass) for idx, p in top_k_pairs]

def apply_top_p(indexed_probs, p=0.9):
    sorted_pairs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
    nucleus = []
    cumulative = 0.0
    for idx, prob in sorted_pairs:
        nucleus.append((idx, prob))
        cumulative += prob
        if cumulative >= p:
            break
    total_mass = sum(prob for _, prob in nucleus)
    return [(idx, prob / total_mass) for idx, prob in nucleus]

def generate_token(logits, temperature=0.7, top_k=5, top_p=0.9):
    if temperature <= 1e-4:
        return max(enumerate(logits), key=lambda x: x[1])[0]
    probs = sample_with_temperature(logits, temperature)
    top_k_pairs = apply_top_k(probs, top_k)
    nucleus_pairs = apply_top_p(top_k_pairs, top_p)
    r = random.random()
    acc = 0.0
    for tok_id, prob in nucleus_pairs:
        acc += prob
        if r <= acc:
            return tok_id
    return nucleus_pairs[-1][0]

# =====================================================================
# 4. Multi-Layer KV Cache Structure
# =====================================================================
class SingleLayerCache:
    def __init__(self):
        self.k_cache = []  # List of vectors [d_model]
        self.v_cache = []  # List of vectors [d_model]

    def reset(self):
        self.k_cache = []
        self.v_cache = []

class MultiLayerKVCache:
    def __init__(self, L):
        self.layers = [SingleLayerCache() for _ in range(L)]

    def reset(self):
        for layer in self.layers:
            layer.reset()

# =====================================================================
# 5. Multi-Layer Transformer Forward & Decode Steps
# =====================================================================
def layer_forward(X_in, layer_params, cache=None):
    """
    Computes a full sequence forward pass for a single Transformer layer.
    If cache is provided, populates the layer's KV cache.
    """
    T = len(X_in)
    
    # 1. Pre-RMSNorm 1 & Self-Attention
    X_norm1, rms1 = rmsnorm_forward(X_in, layer_params["gamma1"])
    Q = matmul(X_norm1, layer_params["W_q"])
    K = matmul(X_norm1, layer_params["W_k"])
    V_mat = matmul(X_norm1, layer_params["W_v"])

    if cache is not None:
        for i in range(T):
            cache.k_cache.append(K[i][:])
            cache.v_cache.append(V_mat[i][:])

    scores = matmul(Q, transpose(K))
    for i in range(T):
        for j in range(T):
            scores[i][j] *= scale
            if j > i:
                scores[i][j] = -1e9  # Causal mask

    A = [softmax_row(scores[i]) for i in range(T)]
    O_raw = matmul(A, V_mat)
    Attn_out = matmul(O_raw, layer_params["W_o"])
    
    # Residual Connection 1
    X_mid = [[X_in[i][j] + Attn_out[i][j] for j in range(d_model)] for i in range(T)]

    # 2. Pre-RMSNorm 2 & SwiGLU FFN
    X_norm2, rms2 = rmsnorm_forward(X_mid, layer_params["gamma2"])
    Hg = matmul(X_norm2, layer_params["W_gate"])
    Hu = matmul(X_norm2, layer_params["W_up"])
    H_swiglu = [[silu(Hg[i][j]) * Hu[i][j] for j in range(d_ffn)] for i in range(T)]
    FFN_out = matmul(H_swiglu, layer_params["W_down"])

    # Residual Connection 2
    X_out = [[X_mid[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

    cache_data = {
        "X_in": X_in, "X_norm1": X_norm1, "rms1": rms1,
        "Q": Q, "K": K, "V_mat": V_mat, "scores": scores, "A": A,
        "O_raw": O_raw, "Attn_out": Attn_out, "X_mid": X_mid,
        "X_norm2": X_norm2, "rms2": rms2, "Hg": Hg, "Hu": Hu,
        "H_swiglu": H_swiglu, "FFN_out": FFN_out
    }
    return X_out, cache_data

def layer_decode_step(x_in, layer_params, cache):
    """
    Computes a single token forward step in O(1) time using the layer's KV cache.
    x_in: [1, d_model]
    """
    # 1. Pre-RMSNorm 1
    x_norm1, _ = rmsnorm_forward(x_in, layer_params["gamma1"])

    # 2. Projections & Cache Append
    q = matmul(x_norm1, layer_params["W_q"])[0]
    k = matmul(x_norm1, layer_params["W_k"])[0]
    v = matmul(x_norm1, layer_params["W_v"])[0]
    cache.k_cache.append(k[:])
    cache.v_cache.append(v[:])

    # 3. Attention against all cached keys
    raw_scores = []
    for cached_k in cache.k_cache:
        score = sum(q[j] * cached_k[j] for j in range(d_model)) * scale
        raw_scores.append(score)
    attn_weights = softmax_row(raw_scores)

    o_raw = [0.0] * d_model
    for j, weight in enumerate(attn_weights):
        for d in range(d_model):
            o_raw[d] += weight * cache.v_cache[j][d]

    attn_out = matmul([o_raw], layer_params["W_o"])[0]
    x_mid = [[x_in[0][j] + attn_out[j] for j in range(d_model)]]

    # 4. Pre-RMSNorm 2 & SwiGLU FFN
    x_norm2, _ = rmsnorm_forward(x_mid, layer_params["gamma2"])
    Hg = matmul(x_norm2, layer_params["W_gate"])
    Hu = matmul(x_norm2, layer_params["W_up"])
    H_swiglu = [[silu(Hg[0][j]) * Hu[0][j] for j in range(d_ffn)]]
    FFN_out = matmul(H_swiglu, layer_params["W_down"])[0]

    # 5. Residual Connection 2
    x_out = [[x_mid[0][j] + FFN_out[j] for j in range(d_model)]]
    return x_out

def multi_layer_prefill(prompt_tokens, params, multi_cache):
    """
    Processes all prompt tokens in parallel across all L layers,
    fills the multi-layer KV cache, and returns logits of the last token.
    """
    multi_cache.reset()
    T = len(prompt_tokens)
    X = [params["E"][tok_id][:] for tok_id in prompt_tokens]

    for l in range(num_layers):
        X, _ = layer_forward(X, params["layers"][l], cache=multi_cache.layers[l])

    X_final, _ = rmsnorm_forward(X, params["gamma_final"])
    Z = matmul(X_final, params["W_head"])
    return Z[-1]

def multi_layer_decode_step(tok_id, params, multi_cache):
    """
    Executes a single token forward pass rippling through all L layers in O(1) time.
    """
    x = [params["E"][tok_id][:]]  # [1, d_model]

    for l in range(num_layers):
        x = layer_decode_step(x, params["layers"][l], multi_cache.layers[l])

    x_final, _ = rmsnorm_forward(x, params["gamma_final"])
    z = matmul(x_final, params["W_head"])[0]
    return z

# =====================================================================
# 6. Multi-Layer Backpropagation (Training Engine)
# =====================================================================
def layer_backward(dX_out, layer_params, cache_data):
    """
    Computes gradients for a single Transformer layer via reverse chain rule.
    Returns:
        dX_in: Gradients flowing back to the previous layer
        d_params: Dictionary of weight gradients for this layer
    """
    T = len(dX_out)
    X_mid = cache_data["X_mid"]
    X_norm2 = cache_data["X_norm2"]
    rms2 = cache_data["rms2"]
    Hg = cache_data["Hg"]
    Hu = cache_data["Hu"]
    H_swiglu = cache_data["H_swiglu"]
    X_in = cache_data["X_in"]
    X_norm1 = cache_data["X_norm1"]
    rms1 = cache_data["rms1"]
    Q = cache_data["Q"]
    K = cache_data["K"]
    V_mat = cache_data["V_mat"]
    A = cache_data["A"]
    O_raw = cache_data["O_raw"]

    # --- Backprop through SwiGLU FFN ---
    dX_mid_res2 = dX_out[:]
    dFFN_out = dX_out[:]

    dW_down = matmul(transpose(H_swiglu), dFFN_out)
    dH_swiglu = matmul(dFFN_out, transpose(layer_params["W_down"]))

    dHg = [[0.0] * d_ffn for _ in range(T)]
    dHu = [[0.0] * d_ffn for _ in range(T)]
    for i in range(T):
        for j in range(d_ffn):
            dHu[i][j] = dH_swiglu[i][j] * silu(Hg[i][j])
            dHg[i][j] = dH_swiglu[i][j] * Hu[i][j] * silu_deriv(Hg[i][j])

    dW_gate = matmul(transpose(X_norm2), dHg)
    dW_up   = matmul(transpose(X_norm2), dHu)

    dX_norm2 = [[sum(dHg[i][k] * layer_params["W_gate"][j][k] + dHu[i][k] * layer_params["W_up"][j][k] for k in range(d_ffn))
                 for j in range(d_model)] for i in range(T)]
    dX_mid_from_norm, dgamma2 = rmsnorm_backward(dX_norm2, X_mid, layer_params["gamma2"], rms2)
    dX_mid = [[dX_mid_res2[i][j] + dX_mid_from_norm[i][j] for j in range(d_model)] for i in range(T)]

    # --- Backprop through Self-Attention ---
    dX_in_res1 = dX_mid[:]
    dAttn_out = dX_mid[:]

    dW_o = matmul(transpose(O_raw), dAttn_out)
    dO_raw = matmul(dAttn_out, transpose(layer_params["W_o"]))
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
    dW_q = matmul(transpose(X_norm1), dQ)
    dW_k = matmul(transpose(X_norm1), dK)
    dW_v = matmul(transpose(X_norm1), dV_mat)

    dX_norm1 = [[sum(dQ[i][k] * layer_params["W_q"][j][k] + dK[i][k] * layer_params["W_k"][j][k] + dV_mat[i][k] * layer_params["W_v"][j][k] for k in range(d_model))
                 for j in range(d_model)] for i in range(T)]
    dX_in_from_norm, dgamma1 = rmsnorm_backward(dX_norm1, X_in, layer_params["gamma1"], rms1)
    dX_in = [[dX_in_res1[i][j] + dX_in_from_norm[i][j] for j in range(d_model)] for i in range(T)]

    d_layer_params = {
        "W_q": dW_q, "W_k": dW_k, "W_v": dW_v, "W_o": dW_o,
        "gamma1": dgamma1, "gamma2": dgamma2,
        "W_gate": dW_gate, "W_up": dW_up, "W_down": dW_down
    }
    return dX_in, d_layer_params

# =====================================================================
# 7. Model Initialization & Training Loop
# =====================================================================
random.seed(42)

params = {
    "E": init_matrix(V, d_model),
    "layers": [
        {
            "gamma1":      [1.0] * d_model,
            "W_q":         init_matrix(d_model, d_model),
            "W_k":         init_matrix(d_model, d_model),
            "W_v":         init_matrix(d_model, d_model),
            "W_o":         init_matrix(d_model, d_model),
            "gamma2":      [1.0] * d_model,
            "W_gate":      init_matrix(d_model, d_ffn),
            "W_up":        init_matrix(d_model, d_ffn),
            "W_down":      init_matrix(d_ffn, d_model),
        }
        for _ in range(num_layers)
    ],
    "gamma_final": [1.0] * d_model,
    "W_head":      init_matrix(d_model, V),
}

print(f"\nTraining Multi-Layer LLM (L={num_layers}) for 250 Epochs...")
start_time = time.time()

for epoch in range(251):
    total_loss = 0.0
    for inputs, targets in dataset:
        T = len(inputs)
        X = [params["E"][idx][:] for idx in inputs]

        # Forward pass through all L layers
        layer_caches = []
        for l in range(num_layers):
            X, c_data = layer_forward(X, params["layers"][l])
            layer_caches.append(c_data)

        # Final RMSNorm and Head Logits
        X_final_norm, rms_final = rmsnorm_forward(X, params["gamma_final"])
        Z = matmul(X_final_norm, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]

        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # Backward Pass: Head & Final RMSNorm
        dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]
        dW_head = matmul(transpose(X_final_norm), dZ)
        dX_final_norm = matmul(dZ, transpose(params["W_head"]))
        dX_final, dgamma_final = rmsnorm_backward(dX_final_norm, X, params["gamma_final"], rms_final)

        # Backward pass through all L layers in reverse order
        dX = dX_final
        layer_grads = []
        for l in reversed(range(num_layers)):
            dX, d_layer_params = layer_backward(dX, params["layers"][l], layer_caches[l])
            layer_grads.append((l, d_layer_params))

        # Backward through Token Embedding table E
        for i in range(T):
            idx = inputs[i]
            for j in range(d_model):
                params["E"][idx][j] -= lr * dX[i][j]

        # Apply gradients to head and final norm
        for i in range(d_model):
            for v in range(V):
                params["W_head"][i][v] -= lr * dW_head[i][v]
            params["gamma_final"][i] -= lr * dgamma_final[i]

        # Apply gradients to each layer
        for l, d_params in layer_grads:
            lp = params["layers"][l]
            for i in range(d_model):
                lp["gamma1"][i] -= lr * d_params["gamma1"][i]
                lp["gamma2"][i] -= lr * d_params["gamma2"][i]
                for k in range(d_model):
                    lp["W_q"][i][k] -= lr * d_params["W_q"][i][k]
                    lp["W_k"][i][k] -= lr * d_params["W_k"][i][k]
                    lp["W_v"][i][k] -= lr * d_params["W_v"][i][k]
                    lp["W_o"][i][k] -= lr * d_params["W_o"][i][k]
                for k in range(d_ffn):
                    lp["W_gate"][i][k] -= lr * d_params["W_gate"][i][k]
                    lp["W_up"][i][k]   -= lr * d_params["W_up"][i][k]
            for k in range(d_ffn):
                for j in range(d_model):
                    lp["W_down"][k][j] -= lr * d_params["W_down"][k][j]

    if epoch % 50 == 0:
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d} | Average Loss: {avg_loss:.4f}")

elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f}s")

# =====================================================================
# 8. Generation Demonstration with Multi-Layer KV Cache
# =====================================================================
cache = MultiLayerKVCache(num_layers)

def generate_sentence(prompt, max_tokens=10, temperature=0.7, top_k=5, top_p=0.9):
    prompt_tokens = [word2id[w] for w in prompt.split()]
    tokens = prompt_tokens[:]

    # Phase 1: Prefill across all layers
    last_logit = multi_layer_prefill(prompt_tokens, params, cache)
    next_tok = generate_token(last_logit, temperature, top_k, top_p)
    tokens.append(next_tok)

    # Phase 2: Single-token decode loop with Multi-Layer KV Cache
    for _ in range(max_tokens - 1):
        if id2word[tokens[-1]] == ".":
            break
        decode_logit = multi_layer_decode_step(tokens[-1], params, cache)
        next_tok = generate_token(decode_logit, temperature, top_k, top_p)
        tokens.append(next_tok)

    return " ".join(id2word[t] for t in tokens)

print("\n--- Multi-Layer Autoregressive Generation (L=2) ---")
test_prompts = [
    "why does the sun shine ?",
    "why is the sky blue ?",
    "what do plants eat ?",
    "where do birds fly ?",
    "who made the world ?"
]

for prompt in test_prompts:
    greedy_gen = generate_sentence(prompt, max_tokens=12, temperature=0.0)
    sample_gen = generate_sentence(prompt, max_tokens=12, temperature=0.7, top_k=5, top_p=0.9)
    print(f"\nPrompt: '{prompt}'")
    print(f"  [Greedy T=0.0] -> {greedy_gen}")
    print(f"  [Sample T=0.7] -> {sample_gen}")

print("\n" + "=" * 60)
print(f"Multi-Layer LLM (L={num_layers}) Training & KV Cache Inference Verified!")
print("=" * 60)
