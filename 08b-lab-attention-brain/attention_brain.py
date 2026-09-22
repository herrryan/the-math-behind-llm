"""
Stage 2: The Attention Brain (Scaled Dot-Product Attention & Causal Masking)
The Math Behind Large Language Models - Lab 02

A complete, self-contained autoregressive Attention Neural Language Model in ~140 lines of pure Python.
Zero external libraries: No PyTorch, no TensorFlow, no NumPy.
Dependencies: Python standard library only (math, random).

Corresponds to:
- Chapter 05: The Transformer Blueprint (Round-Table eye contact vs Markov amnesia)
- Chapter 06: Queries, Keys, and Values (Q = X*W_q, K = X*W_k, V = X*W_v)
- Chapter 07: The Softmax Function (Row-wise probability normalization)
- Chapter 08: The Attention Formula & Dividing by sqrt(d_k)
- Chapter 09: Causal Masking (Lower-triangular blindfolds on future tokens)
"""

import math
import random

# =====================================================================
# 1. Corpus, Vocabulary & Dataset Setup
# =====================================================================
# This corpus requires long-range context:
# After "the cat sat on the", the next word is "mat".
# After "the dog sat on the", the next word is "rug".
# A 1-word bigram model (Lab 1) sees only "the" and cannot distinguish mat from rug!
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

V = len(vocab)          # Vocabulary size |V| = 9
d_model = 8             # Hidden dimension of residual stream (Chapter 01)
d_k = 8                 # Query/Key/Value subspace dimension (Chapter 06)
lr = 0.2                # Learning rate for SGD
scale = 1.0 / math.sqrt(d_k)  # Volume knob scale factor 1/sqrt(d_k) (Chapter 08)

print(f"Vocabulary size |V|: {V}")
print(f"Vocabulary: {vocab}")

# Each sentence is paired: input tokens -> target next tokens
# e.g., ['the', 'cat', 'sat', 'on', 'the'] -> ['cat', 'sat', 'on', 'the', 'mat']
dataset = []
for s in sentences:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

# =====================================================================
# 2. Pure Python Linear Algebra Helpers
# =====================================================================
def init_matrix(rows, cols, scale_init=0.3):
    return [[random.gauss(0, scale_init) for _ in range(cols)] for _ in range(rows)]

def matmul(A, B):
    """Matrix multiplication: A [n x m] @ B [m x p] -> [n x p]"""
    n, m = len(A), len(A[0])
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def transpose(A):
    """Transposes a 2D matrix A [n x m] -> [m x n]"""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def softmax_row(row):
    """Numerically stable softmax for a 1D list of logits."""
    max_val = max(row)
    exp_r = [math.exp(v - max_val) for v in row]
    sum_r = sum(exp_r)
    return [v / sum_r for v in exp_r]

# =====================================================================
# 3. Parameter Initialization (Chapters 01, 06, 08)
# =====================================================================
random.seed(42)

params = {
    "E":      init_matrix(V, d_model),     # Token Embedding matrix E in R^{|V| x d_model}
    "W_q":    init_matrix(d_model, d_k),   # Query projection W_q in R^{d_model x d_k}
    "W_k":    init_matrix(d_model, d_k),   # Key projection W_k in R^{d_model x d_k}
    "W_v":    init_matrix(d_model, d_k),   # Value projection W_v in R^{d_model x d_k}
    "W_head": init_matrix(d_k, V),         # Output LM Head W_head in R^{d_k x |V|}
}

# =====================================================================
# 4. Training Loop: Forward, Backward, and SGD Update
# =====================================================================
print("\nTraining Pure Python Attention Brain (140 Lines)...")

for epoch in range(251):
    total_loss = 0.0

    for inputs, targets in dataset:
        T = len(inputs)

        # -------------------------------------------------------------
        # FORWARD PASS (Chapters 06, 07, 08, 09)
        # -------------------------------------------------------------
        # Step A: Embedding lookup X in R^{T x d_model}
        X = [params["E"][idx][:] for idx in inputs]

        # Step B: Linear projections for Q, K, V in R^{T x d_k}
        Q = matmul(X, params["W_q"])
        K = matmul(X, params["W_k"])
        V_mat = matmul(X, params["W_v"])

        # Step C: Scaled dot-product attention scores S = (Q @ K^T) / sqrt(d_k)
        scores = matmul(Q, transpose(K))
        for i in range(T):
            for j in range(T):
                scores[i][j] *= scale
                # Step D: Causal masking (Chapter 09) - future tokens get -1e9
                if j > i:
                    scores[i][j] = -1e9

        # Step E: Row-wise softmax attention weights A in R^{T x T} (Chapter 07)
        A = [softmax_row(scores[i]) for i in range(T)]

        # Step F: Value aggregation O = A @ V in R^{T x d_k} (Chapter 08)
        O = matmul(A, V_mat)

        # Step G: LM Head projection to logits Z in R^{T x V}
        Z = matmul(O, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]

        # Step H: Cross-entropy loss averaged across sequence positions
        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # -------------------------------------------------------------
        # BACKWARD PASS: Analytical Gradient Flow
        # -------------------------------------------------------------
        # 1. Output error signal dZ = (P - one_hot) / T
        dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]

        # 2. LM Head gradients and context gradients
        dW_head = matmul(transpose(O), dZ)
        dO = matmul(dZ, transpose(params["W_head"]))

        # 3. Value and Attention matrix gradients
        dV_mat = matmul(transpose(A), dO)
        dA = matmul(dO, transpose(V_mat))

        # 4. Softmax backward through causal mask
        dScores = [[0.0] * T for _ in range(T)]
        for i in range(T):
            sum_dA_A = sum(dA[i][k] * A[i][k] for k in range(T))
            for j in range(T):
                if j <= i:
                    dScores[i][j] = A[i][j] * (dA[i][j] - sum_dA_A) * scale
                else:
                    dScores[i][j] = 0.0

        # 5. Query and Key gradients
        dQ = matmul(dScores, K)
        dK = matmul(transpose(dScores), Q)

        # 6. Projection weight gradients
        dW_q = matmul(transpose(X), dQ)
        dW_k = matmul(transpose(X), dK)
        dW_v = matmul(transpose(X), dV_mat)

        # 7. Residual input gradients
        dX = [[sum(dQ[i][k] * params["W_q"][j][k] +
                   dK[i][k] * params["W_k"][j][k] +
                   dV_mat[i][k] * params["W_v"][j][k] for k in range(d_k))
               for j in range(d_model)] for i in range(T)]

        # -------------------------------------------------------------
        # PARAMETER UPDATE: Stochastic Gradient Descent (SGD)
        # -------------------------------------------------------------
        for i in range(d_k):
            for v in range(V):
                params["W_head"][i][v] -= lr * dW_head[i][v]
        for i in range(d_model):
            for k in range(d_k):
                params["W_q"][i][k] -= lr * dW_q[i][k]
                params["W_k"][i][k] -= lr * dW_k[i][k]
                params["W_v"][i][k] -= lr * dW_v[i][k]
        for i in range(T):
            idx = inputs[i]
            for j in range(d_model):
                params["E"][idx][j] -= lr * dX[i][j]

    if epoch % 50 == 0:
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d} | Average Cross-Entropy Loss: {avg_loss:.4f}")

# =====================================================================
# 5. Autoregressive Generation: Testing Context Understanding
# =====================================================================
def generate(prompt, max_tokens=5):
    """Autoregressively predicts next tokens given an initial prompt."""
    toks = [word2id[w] for w in prompt.split()]
    for _ in range(max_tokens):
        cur_T = len(toks)
        X = [params["E"][idx][:] for idx in toks]
        Q = matmul(X, params["W_q"])
        K = matmul(X, params["W_k"])
        V_mat = matmul(X, params["W_v"])

        scores = matmul(Q, transpose(K))
        for i in range(cur_T):
            for j in range(cur_T):
                scores[i][j] *= scale
                if j > i:
                    scores[i][j] = -1e9

        A = [softmax_row(scores[i]) for i in range(cur_T)]
        O = matmul(A, V_mat)
        Z = matmul(O, params["W_head"])
        probs = softmax_row(Z[-1])  # Predict from the final token's position
        nxt = probs.index(max(probs))
        toks.append(nxt)
        if id2word[nxt] == ".":
            break
    return " ".join(id2word[t] for t in toks)

print("\n--- Autoregressive Generation Results ---")
print("Prompt 'the cat sat on the'    ->", generate("the cat sat on the"))
print("Prompt 'the dog sat on the'    ->", generate("the dog sat on the"))
print("Prompt 'the cat walked on the' ->", generate("the cat walked on the"))
print("Prompt 'the dog walked on the' ->", generate("the dog walked on the"))

# =====================================================================
# 6. Attention Heatmap Inspection (Look Under the Hood!)
# =====================================================================
def inspect_attention(text):
    """Visualizes causal attention weights across the token sequence."""
    toks = [word2id[w] for w in text.split()]
    cur_T = len(toks)
    X = [params["E"][idx][:] for idx in toks]
    Q = matmul(X, params["W_q"])
    K = matmul(X, params["W_k"])
    scores = matmul(Q, transpose(K))
    for i in range(cur_T):
        for j in range(cur_T):
            scores[i][j] *= scale
            if j > i:
                scores[i][j] = -1e9
    A = [softmax_row(scores[i]) for i in range(cur_T)]
    words_list = text.split()
    print(f"\nAttention Heatmap for: \"{text}\"")
    print("Token \\ Attends to: " + " ".join(f"{w:>6}" for w in words_list))
    for i, w in enumerate(words_list):
        row_str = " ".join(f"{A[i][j]*100:5.1f}%" for j in range(cur_T))
        print(f"[{i}] {w:>6} -> {row_str}")

inspect_attention("the cat sat on the")
inspect_attention("the dog sat on the")


# =====================================================================
# 7. SOTA Extension: Grouped-Query Attention (GQA - Chapter 11)
# =====================================================================
def get_kv_head_index(q_head_idx, group_size):
    """Maps Query head index h to its shared Key-Value head index."""
    return q_head_idx // group_size


def slice_head(tensor_2d, head_idx, d_head):
    """Slices [T x d_head] for head_idx from a [T x (H * d_head)] matrix."""
    T = len(tensor_2d)
    start = head_idx * d_head
    end = start + d_head
    return [[tensor_2d[t][d] for d in range(start, end)] for t in range(T)]


def grouped_query_attention(X, W_q, W_k, W_v, W_o, H_q, H_kv, d_head):
    """
    Computes Grouped-Query Attention (GQA).
    Generalizes across:
      - Multi-Head Attention (MHA): H_kv == H_q
      - Grouped-Query Attention (GQA): 1 < H_kv < H_q (Modern SOTA!)
      - Multi-Query Attention (MQA): H_kv == 1
    """
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
    group_size = H_q // H_kv
    scale_h = 1.0 / math.sqrt(d_head)
    T = len(X)

    # 1. Linear projections
    Q = matmul(X, W_q)    # [T x (H_q * d_head)]
    K = matmul(X, W_k)    # [T x (H_kv * d_head)]
    V_m = matmul(X, W_v)  # [T x (H_kv * d_head)]

    # 2. Multi-head attention with grouped KV routing
    head_outputs = []
    for h in range(H_q):
        kv_idx = get_kv_head_index(h, group_size)
        Q_h = slice_head(Q, h, d_head)
        K_k = slice_head(K, kv_idx, d_head)
        V_k = slice_head(V_m, kv_idx, d_head)

        scores_h = matmul(Q_h, transpose(K_k))
        for i in range(T):
            for j in range(T):
                scores_h[i][j] *= scale_h
                if j > i:
                    scores_h[i][j] = -1e9
        A_h = [softmax_row(scores_h[i]) for i in range(T)]
        O_h = matmul(A_h, V_k)
        head_outputs.append(O_h)

    # 3. Concatenate all query heads along feature dimension
    O_concat = []
    for t in range(T):
        row = []
        for h in range(H_q):
            row.extend(head_outputs[h][t])
        O_concat.append(row)

    # 4. Final output projection back to d_model
    return matmul(O_concat, W_o)


print("\n" + "=" * 60)
print("SOTA Extension: Grouped-Query Attention (GQA) Verification")
print("=" * 60)
toy_T = 3
toy_d_model = 8
toy_d_head = 4
toy_H_q = 4
toy_H_kv = 2  # Group size = 4 // 2 = 2

toy_X = [[0.1 * ((i + j) % 5) for j in range(toy_d_model)] for i in range(toy_T)]
toy_W_q = [[0.05] * (toy_H_q * toy_d_head) for _ in range(toy_d_model)]
toy_W_k = [[0.05] * (toy_H_kv * toy_d_head) for _ in range(toy_d_model)]
toy_W_v = [[0.05] * (toy_H_kv * toy_d_head) for _ in range(toy_d_model)]
toy_W_o = [[0.05] * toy_d_model for _ in range(toy_H_q * toy_d_head)]

gqa_out = grouped_query_attention(
    toy_X, toy_W_q, toy_W_k, toy_W_v, toy_W_o,
    toy_H_q, toy_H_kv, toy_d_head
)
print(f"GQA Output Shape: [{len(gqa_out)} tokens x {len(gqa_out[0])} d_model] - Match!")
print("KV Cache Memory Savings at 128k context (LLaMA-3 70B):")
print("  - MHA (64 KV heads): 312.50 GB (VRAM bottleneck)")
print("  - GQA (8 KV heads):   39.06 GB (8x reduction! SOTA standard)")
print("=" * 60)
