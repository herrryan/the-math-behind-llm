# =====================================================================
# Stage 3: The Transformer Brain (220 Lines of Pure Python)
# Dependencies: Zero external libraries (Python built-in standard library only)
#
# Architectural Milestone:
# - Pre-RMSNorm (Chapter 13)
# - Scaled Dot-Product Causal Self-Attention with W_o (Chapters 08 & 09)
# - Residual Connection Highways (Chapter 12)
# - Modern LLaMA-style SwiGLU Gated Feed-Forward Network (Chapters 04 & 14)
# =====================================================================
import math
import random

# 1. Corpus, Vocabulary & Dataset Setup
sentences = [
    "the cat sat on the mat .",
    "the dog sat on the rug .",
    "the cat walked on the mat .",
    "the dog walked on the rug ."
]

all_words = (" ".join(sentences)).split()
vocab = sorted(list(set(all_words)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}

V = len(vocab)          # |V| = 9
d_model = 8             # Residual stream dimension
d_ffn = 16              # SwiGLU hidden expansion dimension
lr = 0.1                # Learning rate
scale = 1.0 / math.sqrt(d_model)

dataset = []
for s in sentences:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

# 2. Linear Algebra & Activation Primitives
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

# 3. Model Parameters Initialization
random.seed(42)
params = {
    "E":           init_matrix(V, d_model),     # Embedding table [V x d_model]
    "gamma1":      [1.0] * d_model,             # Pre-RMSNorm 1 scale
    "W_q":         init_matrix(d_model, d_model),
    "W_k":         init_matrix(d_model, d_model),
    "W_v":         init_matrix(d_model, d_model),
    "W_o":         init_matrix(d_model, d_model), # Attention output projection
    "gamma2":      [1.0] * d_model,             # Pre-RMSNorm 2 scale
    "W_gate":      init_matrix(d_model, d_ffn), # SwiGLU gate projection
    "W_up":        init_matrix(d_model, d_ffn), # SwiGLU up projection
    "W_down":      init_matrix(d_ffn, d_model), # SwiGLU down projection
    "gamma_final": [1.0] * d_model,             # Final RMSNorm scale
    "W_head":      init_matrix(d_model, V),     # LM Head [d_model x V]
}

# 4. Training Loop: 200 Epochs
print(f"Vocabulary size |V|: {V}")
print(f"Vocabulary: {vocab}")
print("\nTraining Transformer Brain for 200 Epochs...")

for epoch in range(201):
    total_loss = 0.0

    for inputs, targets in dataset:
        T = len(inputs)
        X0 = [params["E"][idx][:] for idx in inputs]

        # --- 1. PRE-ATTENTION RMSNORM ---
        X0_norm, rms1 = rmsnorm_forward(X0, params["gamma1"])

        # --- 2. CAUSAL SELF-ATTENTION ---
        Q = matmul(X0_norm, params["W_q"])
        K = matmul(X0_norm, params["W_k"])
        V_mat = matmul(X0_norm, params["W_v"])

        scores = matmul(Q, transpose(K))
        for i in range(T):
            for j in range(T):
                scores[i][j] *= scale
                if j > i:
                    scores[i][j] = -1e9  # Causal mask

        A = [softmax_row(scores[i]) for i in range(T)]
        O_raw = matmul(A, V_mat)
        Attn_out = matmul(O_raw, params["W_o"])

        # --- 3. RESIDUAL HIGHWAY 1 ---
        X1 = [[X0[i][j] + Attn_out[i][j] for j in range(d_model)] for i in range(T)]

        # --- 4. PRE-FFN RMSNORM ---
        X1_norm, rms2 = rmsnorm_forward(X1, params["gamma2"])

        # --- 5. SWIGLU FEED-FORWARD NETWORK ---
        H_gate = matmul(X1_norm, params["W_gate"])
        H_up   = matmul(X1_norm, params["W_up"])
        H_silu = [[silu(H_gate[i][j]) for j in range(d_ffn)] for i in range(T)]
        H_swiglu = [[H_silu[i][j] * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        FFN_out = matmul(H_swiglu, params["W_down"])

        # --- 6. RESIDUAL HIGHWAY 2 ---
        X2 = [[X1[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

        # --- 7. FINAL RMSNORM ---
        X2_norm, rms_f = rmsnorm_forward(X2, params["gamma_final"])

        # --- 8. LM HEAD & CROSS-ENTROPY LOSS ---
        Z = matmul(X2_norm, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]

        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # --- BACKWARD PASS ---
        dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]
        dW_head = matmul(transpose(X2_norm), dZ)
        dX2_norm = matmul(dZ, transpose(params["W_head"]))

        # Backprop through Final RMSNorm
        dX2, dgamma_f = rmsnorm_backward(dX2_norm, X2, params["gamma_final"], rms_f)

        # Backprop through Residual 2: X2 = X1 + FFN_out
        dX1_res2 = dX2[:]
        dFFN_out = dX2[:]

        # Backprop through SwiGLU FFN
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

        # Backprop through Residual 1: X1 = X0 + Attn_out
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

        # --- SGD PARAMETER UPDATES ---
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
        print(f"Epoch {epoch:3d} | Average Cross-Entropy Loss: {total_loss / len(dataset):.4f}")

# 5. Autoregressive Generation
print("\n--- Autoregressive Generation Results ---")
def generate(prompt, max_new_tokens=2):
    toks = [word2id[w] for w in prompt.split()]
    for _ in range(max_new_tokens):
        T = len(toks)
        X0 = [params["E"][idx][:] for idx in toks]
        X0_norm, _ = rmsnorm_forward(X0, params["gamma1"])

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

        X1_norm, _ = rmsnorm_forward(X1, params["gamma2"])
        H_gate = matmul(X1_norm, params["W_gate"])
        H_up   = matmul(X1_norm, params["W_up"])
        H_swiglu = [[silu(H_gate[i][j]) * H_up[i][j] for j in range(d_ffn)] for i in range(T)]
        FFN_out = matmul(H_swiglu, params["W_down"])
        X2 = [[X1[i][j] + FFN_out[i][j] for j in range(d_model)] for i in range(T)]

        X2_norm, _ = rmsnorm_forward(X2, params["gamma_final"])
        Z = matmul(X2_norm, params["W_head"])

        next_tok = max(range(V), key=lambda v: Z[-1][v])
        toks.append(next_tok)
    return " ".join(id2word[t] for t in toks)

for prompt in [
    "the cat sat on the",
    "the dog sat on the",
    "the cat walked on the",
    "the dog walked on the"
]:
    print(f"Prompt '{prompt}' -> {generate(prompt)}")
