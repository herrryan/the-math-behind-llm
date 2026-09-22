# =====================================================================
# Guided Exercise: The Transformer Brain (Modern LLaMA-style Architecture)
# Pure Standard-Library Python (Zero External Dependencies)
#
# YOUR MISSION:
# In Lab 02, we discovered the fatal bottleneck of raw attention:
# "Depth Instability". When stacking deep attention blocks, signals explode
# or vanish, and token identities blur away.
#
# In this exercise, you will implement the two foundational pillars of modern
# LLMs (LLaMA 3, Gemma, Mistral, DeepSeek) that solve depth instability:
# 1. Root Mean Square Normalization (RMSNorm) - Chapter 13
# 2. Residual Connection Highways (X + Sublayer(X)) - Chapter 12
# 3. Modern Gated SwiGLU Feed-Forward Networks - Chapters 04 & 14
#
# Complete the 5 TODO blocks below. An automated unit test suite at the bottom
# will verify your arithmetic at each step before beginning training!
# =====================================================================
import math
import random

# =====================================================================
# 1. Corpus, Vocabulary & Dataset Setup (Chapter 00)
# =====================================================================
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

print(f"Vocabulary size |V|: {V}")
print(f"Vocabulary: {vocab}")

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

def silu_deriv(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


# =====================================================================
# TODO 1: Root Mean Square Normalization (Chapter 13)
# =====================================================================
def compute_rmsnorm(X, gamma, eps=1e-5):
    """
    Computes Root Mean Square Normalization along the feature dimension d.

    Mathematical Formula:
        RMS(x_i) = sqrt( (1 / d) * sum_{j=1}^d (x_{ij}^2) + eps )
        x_{i, norm}[j] = (x_{ij} / RMS(x_i)) * gamma[j]

    Physical Metaphor:
        The Volume Limiter in a concert hall. It scales down loud signals
        and boosts quiet signals so the speaker never pops or clips.

    Arguments:
        X: List of T vectors, each of dimension d. Shape [T, d]
        gamma: Learnable gain vector of dimension d. Shape [d]
        eps: Small numerical stability constant (default 1e-5)

    Returns:
        X_norm: Normalized 2D tensor of shape [T, d]
        rms_list: List of length T containing the scalar RMS value for each row
    """
    # YOUR CODE HERE
    raise NotImplementedError("TODO 1: Implement compute_rmsnorm(X, gamma, eps)")


# =====================================================================
# TODO 2: Residual Connection Highway (Chapter 12)
# =====================================================================
def compute_residual(X_in, Sublayer_out):
    """
    Adds the input representation directly to the sublayer output.

    Mathematical Formula:
        X_out = X_in + Sublayer_out
        x_{out}[i][j] = x_{in}[i][j] + sublayer_{out}[i][j]

    Physical Metaphor:
        The Express Highway running alongside a twisty mountain path.
        Gradients can travel straight back along the highway with zero
        decay or bottleneck.

    Arguments:
        X_in: 2D list of shape [T, d]
        Sublayer_out: 2D list of shape [T, d]

    Returns:
        X_out: Element-wise sum of shape [T, d]
    """
    # YOUR CODE HERE
    raise NotImplementedError("TODO 2: Implement compute_residual(X_in, Sublayer_out)")


# =====================================================================
# TODO 3: SiLU (Swish) Activation Function (Chapter 04)
# =====================================================================
def compute_silu(x):
    """
    Computes the Sigmoid Linear Unit (SiLU / Swish) activation.

    Mathematical Formula:
        SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Physical Metaphor:
        A smooth physical one-way valve that allows positive signals
        to pass proportional to their strength, but softly pinches
        negative signals near zero without a harsh angular cutoff.

    Arguments:
        x: A scalar float

    Returns:
        The scalar float result of SiLU(x)
    """
    # YOUR CODE HERE
    raise NotImplementedError("TODO 3: Implement compute_silu(x)")


# =====================================================================
# TODO 4: SwiGLU Gated Feed-Forward Network (Chapters 04 & 14)
# =====================================================================
def compute_swiglu_ffn(X_norm, W_gate, W_up, W_down):
    """
    Computes the modern LLaMA-style SwiGLU feed-forward network.

    Mathematical Formula:
        H_gate = X_norm * W_gate         # Shape [T, d_ffn]
        H_up   = X_norm * W_up           # Shape [T, d_ffn]
        H_silu = SiLU(H_gate)            # Element-wise SiLU
        H_swiglu = H_silu (dot) H_up     # Element-wise multiplication (Hadamard)
        FFN_out = H_swiglu * W_down      # Shape [T, d_model]

    Physical Metaphor:
        The Memory Storage Locker with an active security gate.
        H_up retrieves candidate factual knowledge, while H_gate
        acts as an intelligent filter deciding which facts to release.

    Arguments:
        X_norm: Normalized activations of shape [T, d_model]
        W_gate: Weight matrix of shape [d_model, d_ffn]
        W_up: Weight matrix of shape [d_model, d_ffn]
        W_down: Weight matrix of shape [d_ffn, d_model]

    Returns:
        FFN_out: 2D list of shape [T, d_model]
        cache: (H_gate, H_up, H_silu, H_swiglu) needed for backprop
    """
    # YOUR CODE HERE
    raise NotImplementedError("TODO 4: Implement compute_swiglu_ffn(X_norm, W_gate, W_up, W_down)")


# =====================================================================
# Backward Helpers (Provided)
# =====================================================================
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
# Automated Unit Test Suite
# =====================================================================
def run_unit_tests():
    print("=" * 60)
    print("Running Automated Unit Tests for Transformer Brain...")
    print("=" * 60)

    # Test 1: compute_rmsnorm
    try:
        test_x = [[2.0, 2.0, 2.0, 2.0]]
        test_gamma = [1.0, 1.0, 1.0, 1.0]
        x_norm, rms = compute_rmsnorm(test_x, test_gamma, eps=0.0)
        assert abs(rms[0] - 2.0) < 1e-4, f"RMS should be 2.0, got {rms[0]}"
        assert all(abs(v - 1.0) < 1e-4 for v in x_norm[0]), f"x_norm should be [1, 1, 1, 1], got {x_norm[0]}"
        print("[PASS] Step 1: compute_rmsnorm is correct.")
    except NotImplementedError:
        print("[TODO] Step 1: compute_rmsnorm not implemented yet.")
        return False

    # Test 2: compute_residual
    try:
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[0.5, 0.5], [1.0, 1.0]]
        res = compute_residual(a, b)
        assert abs(res[0][0] - 1.5) < 1e-5 and abs(res[1][1] - 5.0) < 1e-5, f"Residual addition failed: got {res}"
        print("[PASS] Step 2: compute_residual is correct.")
    except NotImplementedError:
        print("[TODO] Step 2: compute_residual not implemented yet.")
        return False

    # Test 3: compute_silu
    try:
        s0 = compute_silu(0.0)
        s2 = compute_silu(2.0)
        assert abs(s0 - 0.0) < 1e-5, f"SiLU(0) should be 0, got {s0}"
        expected_s2 = 2.0 / (1.0 + math.exp(-2.0))
        assert abs(s2 - expected_s2) < 1e-4, f"SiLU(2) mismatch: expected {expected_s2}, got {s2}"
        print("[PASS] Step 3: compute_silu is correct.")
    except NotImplementedError:
        print("[TODO] Step 3: compute_silu not implemented yet.")
        return False

    # Test 4: compute_swiglu_ffn
    try:
        x_in = [[1.0, 1.0]]
        w_g = [[1.0, 0.0], [0.0, 1.0]]
        w_u = [[2.0, 0.0], [0.0, 2.0]]
        w_d = [[1.0, 0.0], [0.0, 1.0]]
        ffn_out, _ = compute_swiglu_ffn(x_in, w_g, w_u, w_d)
        expected_val = compute_silu(1.0) * 2.0
        assert abs(ffn_out[0][0] - expected_val) < 1e-4, f"SwiGLU output mismatch: got {ffn_out[0][0]}, expected {expected_val}"
        print("[PASS] Step 4: compute_swiglu_ffn is correct.")
    except NotImplementedError:
        print("[TODO] Step 4: compute_swiglu_ffn not implemented yet.")
        return False

    print("=" * 60)
    print("All core unit tests PASSED! Ready for training loop.")
    print("=" * 60)
    return True


# =====================================================================
# Main Execution & Training Loop
# =====================================================================
if __name__ == "__main__":
    if not run_unit_tests():
        print("\nPlease complete the TODO blocks above and rerun this script!\n")
        exit(0)

    # Parameter initialization
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

    print("\nTraining Transformer Brain for 200 Epochs...")
    for epoch in range(201):
        total_loss = 0.0

        for inputs, targets in dataset:
            T = len(inputs)
            X0 = [params["E"][idx][:] for idx in inputs]

            # 1. Pre-RMSNorm 1
            X0_norm, rms1 = compute_rmsnorm(X0, params["gamma1"])

            # 2. Self-Attention
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

            # 3. Residual Connection 1
            X1 = compute_residual(X0, Attn_out)

            # 4. Pre-RMSNorm 2
            X1_norm, rms2 = compute_rmsnorm(X1, params["gamma2"])

            # 5. SwiGLU FFN
            FFN_out, (H_gate, H_up, H_silu, H_swiglu) = compute_swiglu_ffn(
                X1_norm, params["W_gate"], params["W_up"], params["W_down"]
            )

            # 6. Residual Connection 2
            X2 = compute_residual(X1, FFN_out)

            # 7. Final RMSNorm
            X2_norm, rms_f = compute_rmsnorm(X2, params["gamma_final"])

            # 8. LM Head & Loss
            Z = matmul(X2_norm, params["W_head"])
            P = [softmax_row(Z[i]) for i in range(T)]

            loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
            total_loss += loss

            # --- Backward Pass ---
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

            # SGD Updates
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

    print("\n--- Autoregressive Generation Results ---")
    def generate(prompt, max_new_tokens=2):
        toks = [word2id[w] for w in prompt.split()]
        for _ in range(max_new_tokens):
            T = len(toks)
            X0 = [params["E"][idx][:] for idx in toks]
            X0_norm, _ = compute_rmsnorm(X0, params["gamma1"])

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
            X1 = compute_residual(X0, Attn_out)

            X1_norm, _ = compute_rmsnorm(X1, params["gamma2"])
            FFN_out, _ = compute_swiglu_ffn(X1_norm, params["W_gate"], params["W_up"], params["W_down"])
            X2 = compute_residual(X1, FFN_out)

            X2_norm, _ = compute_rmsnorm(X2, params["gamma_final"])
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
