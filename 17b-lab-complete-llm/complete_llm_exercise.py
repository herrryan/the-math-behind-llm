# =====================================================================
# Guided Exercise: The Complete LLM (Inference Engine & Sampling Suite)
# Pure Standard-Library Python (Zero External Dependencies)
#
# YOUR MISSION:
# In Lab 03, we built the Modern Transformer Brain. But raw greedy decoding
# suffers from severe mechanical repetition and $O(T^2)$ inference latency.
#
# In this capstone exercise, you will build the two foundational pillars of
# industrial LLM deployment (vLLM, Ollama, HuggingFace, llama.cpp):
# 1. Key-Value (KV) Cache Acceleration (Chapter 17)
#    - Reduces decoding latency from O(T^2) to O(1) step complexity.
# 2. Temperature, Top-k, and Top-p (Nucleus) Sampling Suite (Chapter 18)
#    - Transforms rigid repetitive outputs into natural, creative dialogue.
#
# Complete the 4 TODO blocks below. An automated unit test suite will verify
# your implementation at every step before starting generation!
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

V = len(vocab)          # |V| = 39
d_model = 12            # Residual stream dimension
d_ffn = 24              # SwiGLU expansion dimension
lr = 0.15               # Learning rate
scale = 1.0 / math.sqrt(d_model)

dataset = []
for s in corpus:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

print(f"Vocabulary size |V|: {V}")
print(f"Dataset sentences: {len(corpus)}")

# =====================================================================
# 2. Linear Algebra Primitives
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
# TODO 1: Temperature Scaling (Chapter 18)
# =====================================================================
def sample_with_temperature(logits, temperature=1.0):
    """
    Applies temperature scaling to raw logits and returns normalized probabilities.

    Mathematical Formula:
        z'_i = z_i / T
        p_i = exp(z'_i - max(z')) / sum_j exp(z'_j - max(z'))

    Physical Metaphor:
        The Thermal Agitator. High temperature (T > 1.0) boils the water,
        flattening probabilities so rare words can bubble up. Low temperature
        (T < 1.0) freezes the water into crystal ice, amplifying the peak word.

    Arguments:
        logits: List of float logit scores of length |V|
        temperature: Positive float scalar T (default 1.0)

    Returns:
        probs: List of probabilities summing to 1.0
    """
    # YOUR CODE HERE
    temp_logits = [l / temperature for l in logits]
    max_logit = max(temp_logits)
    exp_logits = [math.exp(l - max_logit) for l in temp_logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    return probs


# =====================================================================
# TODO 2: Top-k Truncation (Chapter 18)
# =====================================================================
def apply_top_k(probs, k=5):
    """
    Keeps only the k tokens with the highest probabilities, zeroing out all others.

    Mathematical Definition:
        Select subset K containing top-k tokens:
        p'_i = p_i if i in K else 0.0
        Re-normalize: p''_i = p'_i / sum_{j in K} p'_j

    Physical Metaphor:
        The VIP Bouncer. Only the top-k contenders on the guest list
        are allowed into the room; all others are barred at the door.

    Arguments:
        probs: List of probabilities of length |V|
        k: Integer number of candidate tokens to retain

    Returns:
        List of (token_id, normalized_prob) pairs for the top-k tokens
    """
    # YOUR CODE HERE
    indexed_probs = list(enumerate(probs))
    sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
    top_k = sorted_probs[:k]
    sum_top_k = sum(p for _, p in top_k)
    normalized_top_k = [(idx, p / sum_top_k) for idx, p in top_k]
    return normalized_top_k


# =====================================================================
# TODO 3: Top-p (Nucleus) Truncation (Chapter 18)
# =====================================================================
def apply_top_p(indexed_probs, p=0.9):
    """
    Retains the smallest set of tokens whose cumulative probability exceeds p.

    Mathematical Definition:
        Sort indexed pairs in descending order: p_1 >= p_2 >= ... >= p_V
        Find smallest cutoff index M such that sum_{i=1}^M p_i >= p
        Keep only indices {1, ..., M} and re-normalize probabilities.

    Physical Metaphor:
        The Dynamic Bubble. When confident (e.g. 95% on one word), the bubble
        shrinks to just 1 token. When uncertain, the bubble expands to include
        many reasonable possibilities.

    Arguments:
        indexed_probs: List of (token_id, prob) pairs sorted in descending order
        p: Float probability mass threshold in (0.0, 1.0]

    Returns:
        List of (token_id, normalized_prob) pairs retained in the nucleus
    """
    # YOUR CODE HERE
    indexed_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
    cumulative_prob = 0.0
    nucleus_tokens = []
    for idx, prob in indexed_probs:
        nucleus_tokens.append((idx, prob))
        cumulative_prob += prob
        if cumulative_prob >= p:
            break
    sum_nucleus = sum(p for _, p in nucleus_tokens)
    normalized_nucleus = [(idx, p / sum_nucleus) for idx, p in nucleus_tokens]
    return normalized_nucleus


# =====================================================================
# KV Cache Structure & Prefill
# =====================================================================
class KVCache:
    def __init__(self):
        self.k_cache = []  # List of vectors [d_model]
        self.v_cache = []  # List of vectors [d_model]

    def reset(self):
        self.k_cache = []
        self.v_cache = []

def prefill(prompt_tokens, params, cache):
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
    return Z[-1]


# =====================================================================
# TODO 4: KV Cache Single-Token Decoding Step (Chapter 17)
# =====================================================================
def decode_step_with_cache(tok_id, params, cache):
    """
    Executes a single token forward step in O(1) time using cached Keys and Values.

    Computational Steps:
        1. Embed the single token: x0 = [params["E"][tok_id]]  # Shape [1, d_model]
        2. Apply Pre-RMSNorm 1 with params["gamma1"]
        3. Project single q, k, v vectors:
           q = (x0_norm * W_q)[0]
           k = (x0_norm * W_k)[0]
           v = (x0_norm * W_v)[0]
        4. Append k to cache.k_cache, v to cache.v_cache
        5. Compute attention against ALL cached keys:
           raw_score[j] = (q (dot) cache.k_cache[j]) * scale
           attn_weights = softmax(raw_scores)
        6. Aggregate cached values:
           o_raw = sum_j (attn_weights[j] * cache.v_cache[j])
        7. Project o_raw through W_o and add residual highway: x1 = x0 + attn_out
        8. Apply Pre-RMSNorm 2 with params["gamma2"]
        9. Compute SwiGLU FFN: (SiLU(x1_norm * W_gate) * (x1_norm * W_up)) * W_down
        10. Add residual highway: x2 = x1 + ffn_out
        11. Apply Final RMSNorm with params["gamma_final"]
        12. Project through W_head to produce logits: z = x2_norm * W_head

    Arguments:
        tok_id: Integer token ID
        params: Model parameter dictionary
        cache: KVCache instance holding past keys and values

    Returns:
        z: 1D list of length |V| containing output logits for next token
    """
    # YOUR CODE HERE
    x0 = [params["E"][tok_id]]
    # 2. Apply Pre-RMSNorm 1 with params["gamma1"]
    x0_norm, _ = rmsnorm_forward(x0, params["gamma1"])
    # 3. Project single q, k, v vectors:
    q = [v * params["W_q"][0][j] for j, v in enumerate(x0_norm[0])]
    k = [v * params["W_k"][0][j] for j, v in enumerate(x0_norm[0])]
    v = [v * params["W_v"][0][j] for j, v in enumerate(x0_norm[0])]
    
    # 4. Append k to cache.k_cache, v to cache.v_cache
    cache.k_cache.append(k)
    cache.v_cache.append(v)
    
    # 5. Compute attention against ALL cached keys:
    raw_scores = []
    for cached_k in cache.k_cache:
        score = sum(q[j] * cached_k[j] for j in range(d_model)) * scale
        raw_scores.append(score)
        
    attn_weights = softmax_row(raw_scores)
    
    # 6. Aggregate cached values:
    o_raw = [0.0] * d_model
    for j, weight in enumerate(attn_weights):
        for d in range(d_model):
            o_raw[d] += weight * cache.v_cache[j][d]
            
    # 7. Project o_raw through W_o and add residual highway: x1 = x0 + attn_out
    attn_out = matmul([o_raw], params["W_o"])[0]
    x1 = [x0_norm[0][j] + attn_out[j] for j in range(d_model)]
    
    # 8. Apply Pre-RMSNorm 2 with params["gamma2"]
    x1_norm, _ = rmsnorm_forward([x1], params["gamma2"])
    
    # 9. Compute SwiGLU FFN
    # H_gate = x1_norm * W_gate
    H_gate = matmul(x1_norm, params["W_gate"])
    # H_up = x1_norm * W_up
    H_up = matmul(x1_norm, params["W_up"])
    
    # SiLU(H_gate)
    H_silu = [[silu(val) for val in row] for row in H_gate]
    
    # H_swiglu = SiLU(H_gate) * H_up
    H_swiglu = [[H_silu[0][j] * H_up[0][j] for j in range(d_model)]]
    
    # FFN_out = H_swiglu * W_down
    FFN_out = matmul(H_swiglu, params["W_down"])
    
    # 10. Add residual highway: x2 = x1 + ffn_out
    x2 = [x1_norm[0][j] + FFN_out[0][j] for j in range(d_model)]
    
    # 11. Apply Final RMSNorm with params["gamma_final"]
    x2_norm, _ = rmsnorm_forward([x2], params["gamma_final"])
    
    # 12. Project through W_head to produce logits
    z = matmul(x2_norm, params["W_head"])[0]
    
    return z


# =====================================================================
# Automated Unit Test Suite
# =====================================================================
def run_unit_tests():
    print("=" * 60)
    print("Running Automated Unit Tests for Complete LLM Engine...")
    print("=" * 60)

    # Test 1: Temperature Scaling
    try:
        logits = [2.0, 4.0]
        p_t1 = sample_with_temperature(logits, temperature=1.0)
        p_t2 = sample_with_temperature(logits, temperature=2.0)
        assert abs(sum(p_t1) - 1.0) < 1e-5, f"Probabilities must sum to 1, got {sum(p_t1)}"
        assert p_t2[1] < p_t1[1], f"Higher temperature should flatten distribution: {p_t2[1]} >= {p_t1[1]}"
        print("[PASS] Step 1: sample_with_temperature is correct.")
    except NotImplementedError:
        print("[TODO] Step 1: sample_with_temperature not implemented yet.")
        return False

    # Test 2: Top-k Truncation
    try:
        probs = [0.1, 0.5, 0.05, 0.25, 0.1]
        top_pairs = apply_top_k(probs, k=2)
        assert len(top_pairs) == 2, f"Should return 2 pairs, got {len(top_pairs)}"
        assert top_pairs[0][0] == 1, f"Top token should be index 1, got {top_pairs[0][0]}"
        assert abs(sum(p for _, p in top_pairs) - 1.0) < 1e-5, f"Top-k mass must renormalize to 1.0"
        print("[PASS] Step 2: apply_top_k is correct.")
    except NotImplementedError:
        print("[TODO] Step 2: apply_top_k not implemented yet.")
        return False

    # Test 3: Top-p Truncation
    try:
        indexed_pairs = [(1, 0.6), (3, 0.25), (0, 0.1), (4, 0.05)]
        nucleus = apply_top_p(indexed_pairs, p=0.8)
        assert len(nucleus) == 2, f"Top-p 0.8 should keep top 2 items (0.6 + 0.25 = 0.85 >= 0.8), got {len(nucleus)}"
        assert abs(sum(p for _, p in nucleus) - 1.0) < 1e-5, f"Top-p mass must renormalize to 1.0"
        print("[PASS] Step 3: apply_top_p is correct.")
    except NotImplementedError:
        print("[TODO] Step 3: apply_top_p not implemented yet.")
        return False

    # Test 4: KV Cache Decode Step
    try:
        test_cache = KVCache()
        test_cache.k_cache.append([1.0] * d_model)
        test_cache.v_cache.append([1.0] * d_model)
        dummy_params = {
            "E": [[0.5] * d_model for _ in range(V)],
            "gamma1": [1.0] * d_model,
            "W_q": [[0.1] * d_model for _ in range(d_model)],
            "W_k": [[0.1] * d_model for _ in range(d_model)],
            "W_v": [[0.1] * d_model for _ in range(d_model)],
            "W_o": [[0.1] * d_model for _ in range(d_model)],
            "gamma2": [1.0] * d_model,
            "W_gate": [[0.1] * d_ffn for _ in range(d_model)],
            "W_up":   [[0.1] * d_ffn for _ in range(d_model)],
            "W_down": [[0.1] * d_model for _ in range(d_ffn)],
            "gamma_final": [1.0] * d_model,
            "W_head": [[0.1] * V for _ in range(d_model)],
        }
        test_logits = decode_step_with_cache(0, dummy_params, test_cache)
        assert len(test_logits) == V, f"Expected output logits of size {V}, got {len(test_logits)}"
        assert len(test_cache.k_cache) == 2, f"Cache length should have grown to 2, got {len(test_cache.k_cache)}"
        print("[PASS] Step 4: decode_step_with_cache is correct.")
    except NotImplementedError:
        print("[TODO] Step 4: decode_step_with_cache not implemented yet.")
        return False

    print("=" * 60)
    print("All unit tests PASSED! Ready for full training and inference.")
    print("=" * 60)
    return True


# =====================================================================
# Main Execution, Training & Generation Demo
# =====================================================================
if __name__ == "__main__":
    if not run_unit_tests():
        print("\nPlease complete the TODO blocks above and rerun this script!\n")
        exit(0)

    # Initialize model parameters
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

    print("\nTraining Complete LLM for 250 Epochs...")
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
                    if j > i:
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

    # Text Generation Function combining KV Cache and Sampling Suite
    cache = KVCache()

    def generate_token(logits, temperature=0.7, top_k=5, top_p=0.9):
        if temperature <= 1e-4:
            return max(range(len(logits)), key=lambda i: logits[i])
        probs = sample_with_temperature(logits, temperature)
        top_k_pairs = apply_top_k(probs, top_k)
        nucleus_pairs = apply_top_p(top_k_pairs, top_p)
        r = random.random()
        acc = 0.0
        for tok_id, p in nucleus_pairs:
            acc += p
            if r <= acc:
                return tok_id
        return nucleus_pairs[-1][0]

    def generate_sentence(prompt, max_tokens=10, temperature=0.7, top_k=5, top_p=0.9):
        prompt_tokens = [word2id[w] for w in prompt.split()]
        tokens = prompt_tokens[:]

        last_logit = prefill(prompt_tokens, params, cache)
        next_tok = generate_token(last_logit, temperature, top_k, top_p)
        tokens.append(next_tok)

        for _ in range(max_tokens - 1):
            if id2word[tokens[-1]] == ".":
                break
            decode_logit = decode_step_with_cache(tokens[-1], params, cache)
            next_tok = generate_token(decode_logit, temperature, top_k, top_p)
            tokens.append(next_tok)

        return " ".join(id2word[t] for t in tokens)

    print("\n--- Autoregressive Generation with KV Cache Acceleration ---")
    for prompt in [
        "why does the sun shine ?",
        "why is the sky blue ?",
        "what do plants eat ?",
        "where do birds fly ?",
        "who made the world ?"
    ]:
        print(f"\nPrompt: '{prompt}'")
        greedy_out = generate_sentence(prompt, max_tokens=10, temperature=0.0)
        print(f"  [Greedy T=0.0] -> {greedy_out}")
        sample_out = generate_sentence(prompt, max_tokens=10, temperature=0.7, top_k=5, top_p=0.9)
        print(f"  [Sample T=0.7] -> {sample_out}")

    print("\n" + "=" * 60)
    print("Congratulations! Stage 4 Complete LLM is fully operational!")
    print("=" * 60)
