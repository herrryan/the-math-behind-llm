"""
Stage 2: The Attention Brain (Scaled Dot-Product Attention & Causal Masking) - Guided Exercise
The Math Behind Large Language Models - Lab 02

Instructions:
-------------
In this hands-on lab, you will build a complete autoregressive Attention Brain from scratch
using 100% pure Python (no PyTorch, no TensorFlow, no NumPy).

Scaffolding (corpus setup, parameter initialization, matrix multiplication helpers,
training loop harness) has been provided for you. Your mission is to fill in the core
mathematical operations marked with TODO.

To guide your implementation, every TODO block includes:
  1. Mathematical Formula (LaTeX style)
  2. Physical Intuition & Tensor Shapes
  3. Heuristic Guidance & Python Hints

When you run this script:
  $ python3 02_attention_brain_exercise.py
It will validate each step with automated unit tests before running training and generation!
"""

import math
import random

# =====================================================================
# 1. Corpus, Vocabulary & Dataset Setup
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

V = len(vocab)          # Vocabulary size |V| = 9
d_model = 8             # Hidden dimension of residual stream (Chapter 01)
d_k = 8                 # Query/Key/Value subspace dimension (Chapter 06)
lr = 0.2                # Learning rate for SGD
scale = 1.0 / math.sqrt(d_k)  # Volume knob scale factor 1/sqrt(d_k) (Chapter 08)

print(f"Vocabulary size |V|: {V}")
print(f"Vocabulary: {vocab}")

# Each sentence is paired: input tokens -> target next tokens
dataset = []
for s in sentences:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

# =====================================================================
# 2. Linear Algebra Helpers (Provided)
# =====================================================================
def init_matrix(rows, cols, scale_init=0.3):
    """Initializes a 2D list of shape [rows x cols] with Gaussian random values."""
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

def init_parameters(seed=42):
    """Initializes all trainable parameters of the Attention Brain."""
    random.seed(seed)
    return {
        "E":      init_matrix(V, d_model),     # Token Embeddings: [V x d_model] (Chapter 01)
        "W_q":    init_matrix(d_model, d_k),   # Query projection: [d_model x d_k] (Chapter 06)
        "W_k":    init_matrix(d_model, d_k),   # Key projection:   [d_model x d_k] (Chapter 06)
        "W_v":    init_matrix(d_model, d_k),   # Value projection: [d_model x d_k] (Chapter 06)
        "W_head": init_matrix(d_k, V),         # Output LM Head:   [d_k x V]
    }

# =====================================================================
# 3. Core Attention Operations to Implement
# =====================================================================

def project_qkv(X, W_q, W_k, W_v):
    """
    Projects sequence token representations into Query, Key, and Value spaces.

    Args:
        X: list of lists [T x d_model], token embedding representations
        W_q: list of lists [d_model x d_k], Query projection matrix
        W_k: list of lists [d_model x d_k], Key projection matrix
        W_v: list of lists [d_model x d_k], Value projection matrix

    Returns:
        Q: list of lists [T x d_k]
        K: list of lists [T x d_k]
        V_mat: list of lists [T x d_k]

    Mathematical Formula:
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v

    Physical Intuition:
        Each word enters with a general identity X. Projections give it three distinct roles:
        - Query (Q): What clues am I looking for in the past?
        - Key (K): What information do I offer to future searchers?
        - Value (V): What actual content payload do I carry?
    """
    # -----------------------------------------------------------------
    # TODO 1: Implement Q, K, V Projections (Chapter 06)
    # Hint: Use the provided matmul(A, B) helper function.
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # Q = ...
    # K = ...
    # V_mat = ...
    raise NotImplementedError("TODO 1: Implement project_qkv(X, W_q, W_k, W_v)")


def compute_scaled_scores(Q, K, scale_factor):
    """
    Computes scaled dot-product attention affinity scores.

    Args:
        Q: list of lists [T x d_k], Query matrix
        K: list of lists [T x d_k], Key matrix
        scale_factor: float, 1 / sqrt(d_k) (the Master Volume Knob)

    Returns:
        scores: list of lists [T x T], where scores[i][j] = (Q[i] . K[j]) * scale_factor

    Mathematical Formula:
        S = (Q @ K^T) * (1 / sqrt(d_k))

    Physical Intuition:
        Word i asks question Q[i]. Word j shows badge K[j].
        Their dot product measures how well question and badge match.
        We multiply by 1 / sqrt(d_k) to prevent dot products from exploding
        into saturation (where softmax gradients vanish).
    """
    # -----------------------------------------------------------------
    # TODO 2: Implement Scaled Dot-Product Scores (Chapter 08)
    # Hint:
    # 1. Transpose K using transpose(K) to get [d_k x T].
    # 2. Multiply Q @ K^T using matmul(Q, K_T) to get [T x T].
    # 3. Multiply every entry in the resulting matrix by scale_factor.
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    raise NotImplementedError("TODO 2: Implement compute_scaled_scores(Q, K, scale_factor)")


def apply_causal_mask(scores):
    """
    Applies lower-triangular causal mask to prevent peeking at future tokens.

    Args:
        scores: list of lists [T x T], raw scaled scores

    Returns:
        masked_scores: list of lists [T x T], where entries with j > i are replaced by -1e9

    Mathematical Formula:
        S_masked[i, j] = S[i, j] if j <= i else -1e9

    Physical Intuition:
        During autoregressive generation, word i must never see word j if j > i
        (future tokens). By setting future scores to a huge negative value (-1e9),
        e^(-1e9) will equal exactly 0.0 in softmax, completely blindfolding the past!
    """
    # -----------------------------------------------------------------
    # TODO 3: Implement Causal Masking (Chapter 09)
    # Hint:
    # Iterate over row i in range(T) and column j in range(T).
    # If j > i, set masked_scores[i][j] = -1e9.
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    raise NotImplementedError("TODO 3: Implement apply_causal_mask(scores)")


def compute_attention_weights(masked_scores):
    """
    Applies row-wise softmax to convert scores into attention percentage weights.

    Args:
        masked_scores: list of lists [T x T], causally masked attention scores

    Returns:
        A: list of lists [T x T], attention weights where sum(A[i]) == 1.0 for each row i

    Mathematical Formula:
        A[i, :] = softmax(S_masked[i, :])

    Physical Intuition:
        The voting booth: each token has a 100% attention budget (a whole pizza)
        to distribute among itself and all preceding visible tokens.
    """
    # -----------------------------------------------------------------
    # TODO 4: Implement Row-wise Softmax Attention Weights (Chapter 07)
    # Hint:
    # Use the provided softmax_row(row) helper on each row of masked_scores.
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    raise NotImplementedError("TODO 4: Implement compute_attention_weights(masked_scores)")


def aggregate_values(A, V_mat):
    """
    Blends value vectors according to attention weights.

    Args:
        A: list of lists [T x T], attention weight matrix
        V_mat: list of lists [T x d_k], Value matrix

    Returns:
        O: list of lists [T x d_k], context-aware output vectors

    Mathematical Formula:
        O = A @ V_mat

    Physical Intuition:
        If word i paid 60% attention to word 1 and 40% to word 3, its final
        context representation O[i] is 0.6 * V[1] + 0.4 * V[3].
    """
    # -----------------------------------------------------------------
    # TODO 5: Implement Value Aggregation (Chapter 08)
    # Hint:
    # Multiply A @ V_mat using matmul(A, V_mat).
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    raise NotImplementedError("TODO 5: Implement aggregate_values(A, V_mat)")


def forward_pass(inputs, targets, params):
    """
    Full forward pass of the Attention Brain:
    Embeddings -> QKV Projections -> Scaled Dot-Product -> Causal Mask
    -> Softmax -> Value Aggregation -> LM Head -> Cross-Entropy Loss
    """
    T = len(inputs)
    E = params["E"]
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]
    W_head = params["W_head"]

    # 1. Embedding lookup: X in R^{T x d_model}
    X = [E[idx][:] for idx in inputs]

    # 2. Q, K, V projections
    Q, K, V_mat = project_qkv(X, W_q, W_k, W_v)

    # 3. Scaled dot-product scores
    scores = compute_scaled_scores(Q, K, scale)

    # 4. Causal masking
    masked_scores = apply_causal_mask(scores)

    # 5. Row-wise Softmax
    A = compute_attention_weights(masked_scores)

    # 6. Aggregate values
    O = aggregate_values(A, V_mat)

    # 7. LM Head projection to vocabulary logits: Z in R^{T x V}
    # -----------------------------------------------------------------
    # TODO 6: LM Head Projection (Chapter 03)
    # Math: Z = O @ W_head
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # Z = ...
    Z = matmul(O, W_head)

    # 8. Softmax over vocabulary logits for each position
    P = [softmax_row(Z[i]) for i in range(T)]

    # 9. Cross-entropy loss averaged across all sequence positions
    # -----------------------------------------------------------------
    # TODO 7: Cross-Entropy Loss (Chapter 00)
    # Math: loss = (1 / T) * sum(-log(P[i][targets[i]]))
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T

    cache = (X, Q, K, V_mat, scores, masked_scores, A, O, Z, P)
    return loss, P, cache


def backward_pass(inputs, targets, params, cache):
    """
    Backpropagation: Traces sensitivities backwards through LM Head,
    Values, Attention Softmax, Scaled Dot-Product, and QKV projections.
    """
    T = len(inputs)
    X, Q, K, V_mat, scores, masked_scores, A, O, Z, P = cache
    W_head = params["W_head"]
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]

    # 1. Output error signal: dZ = (P - one_hot) / T
    dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]

    # 2. Gradients for LM Head and context vectors
    dW_head = matmul(transpose(O), dZ)
    dO = matmul(dZ, transpose(W_head))

    # 3. Gradients for Value matrix and Attention matrix
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

    # 5. Gradients for Query and Key matrices
    dQ = matmul(dScores, K)
    dK = matmul(transpose(dScores), Q)

    # 6. Gradients for projection weights
    dW_q = matmul(transpose(X), dQ)
    dW_k = matmul(transpose(X), dK)
    dW_v = matmul(transpose(X), dV_mat)

    # 7. Gradients for input embeddings
    dX = [[sum(dQ[i][k] * W_q[j][k] +
               dK[i][k] * W_k[j][k] +
               dV_mat[i][k] * W_v[j][k] for k in range(d_k))
           for j in range(d_model)] for i in range(T)]

    grads = {
        "dW_head": dW_head,
        "dW_q": dW_q,
        "dW_k": dW_k,
        "dW_v": dW_v,
        "dX": dX
    }
    return grads


def update_parameters(params, grads, inputs, lr):
    """Performs SGD update: param = param - lr * grad."""
    for i in range(d_k):
        for v in range(V):
            params["W_head"][i][v] -= lr * grads["dW_head"][i][v]
    for i in range(d_model):
        for k in range(d_k):
            params["W_q"][i][k] -= lr * grads["dW_q"][i][k]
            params["W_k"][i][k] -= lr * grads["dW_k"][i][k]
            params["W_v"][i][k] -= lr * grads["dW_v"][i][k]
    T = len(inputs)
    for i in range(T):
        idx = inputs[i]
        for j in range(d_model):
            params["E"][idx][j] -= lr * grads["dX"][i][j]


# =====================================================================
# 4. Self-Test Suite: Automated Sanity Checks
# =====================================================================
def run_unit_tests():
    """Validates each student implementation step with deterministic inputs."""
    print("=" * 60)
    print("Running Automated Unit Tests...")
    print("=" * 60)

    # Test 1: Q, K, V Projections
    try:
        X_test = [[1.0, 0.0], [0.0, 1.0]]
        W_test = [[1.0, 2.0], [3.0, 4.0]]
        Q, K, V_m = project_qkv(X_test, W_test, W_test, W_test)
        assert len(Q) == 2 and len(Q[0]) == 2, f"Expected shape [2 x 2], got [{len(Q)} x {len(Q[0])}]"
        assert abs(Q[0][0] - 1.0) < 1e-5 and abs(Q[0][1] - 2.0) < 1e-5
        print("[PASS] Step 1: project_qkv implementation is correct.")
    except NotImplementedError:
        print("[TODO] Step 1: project_qkv not implemented yet.")
        return False
    except Exception as e:
        print(f"[FAIL] Step 1: project_qkv error: {e}")
        return False

    # Test 2: Scaled scores
    try:
        Q_test = [[1.0, 0.0], [0.0, 1.0]]
        K_test = [[1.0, 0.0], [0.0, 1.0]]
        scores = compute_scaled_scores(Q_test, K_test, scale_factor=0.5)
        assert abs(scores[0][0] - 0.5) < 1e-5, f"Expected 0.5, got {scores[0][0]}"
        assert abs(scores[0][1] - 0.0) < 1e-5, f"Expected 0.0, got {scores[0][1]}"
        print("[PASS] Step 2: compute_scaled_scores implementation is correct.")
    except NotImplementedError:
        print("[TODO] Step 2: compute_scaled_scores not implemented yet.")
        return False
    except Exception as e:
        print(f"[FAIL] Step 2: compute_scaled_scores error: {e}")
        return False

    # Test 3: Causal Mask
    try:
        scores_test = [[1.0, 2.0], [3.0, 4.0]]
        masked = apply_causal_mask(scores_test)
        assert masked[0][0] == 1.0, f"Expected 1.0, got {masked[0][0]}"
        assert masked[0][1] <= -1e8, f"Expected masked value <= -1e8, got {masked[0][1]}"
        assert masked[1][0] == 3.0 and masked[1][1] == 4.0
        print("[PASS] Step 3: apply_causal_mask implementation is correct.")
    except NotImplementedError:
        print("[TODO] Step 3: apply_causal_mask not implemented yet.")
        return False
    except Exception as e:
        print(f"[FAIL] Step 3: apply_causal_mask error: {e}")
        return False

    # Test 4: Attention Weights (Softmax)
    try:
        masked_test = [[2.0, -1e9], [1.0, 1.0]]
        A = compute_attention_weights(masked_test)
        assert abs(A[0][0] - 1.0) < 1e-5 and abs(A[0][1] - 0.0) < 1e-5
        assert abs(A[1][0] - 0.5) < 1e-5 and abs(A[1][1] - 0.5) < 1e-5
        print("[PASS] Step 4: compute_attention_weights implementation is correct.")
    except NotImplementedError:
        print("[TODO] Step 4: compute_attention_weights not implemented yet.")
        return False
    except Exception as e:
        print(f"[FAIL] Step 4: compute_attention_weights error: {e}")
        return False

    # Test 5: Value Aggregation
    try:
        A_test = [[1.0, 0.0], [0.5, 0.5]]
        V_test = [[10.0, 20.0], [30.0, 40.0]]
        O = aggregate_values(A_test, V_test)
        assert abs(O[0][0] - 10.0) < 1e-5 and abs(O[0][1] - 20.0) < 1e-5
        assert abs(O[1][0] - 20.0) < 1e-5 and abs(O[1][1] - 30.0) < 1e-5
        print("[PASS] Step 5: aggregate_values implementation is correct.")
    except NotImplementedError:
        print("[TODO] Step 5: aggregate_values not implemented yet.")
        return False
    except Exception as e:
        print(f"[FAIL] Step 5: aggregate_values error: {e}")
        return False

    print("=" * 60)
    print("All core unit tests PASSED! Ready for training loop.")
    print("=" * 60)
    return True


# =====================================================================
# 5. Training Loop & Interactive Generation
# =====================================================================
def main():
    if not run_unit_tests():
        print("\nPlease complete the TODO blocks above and rerun this script!")
        return

    params = init_parameters(seed=42)
    print("\nTraining Attention Brain for 250 Epochs...")

    for epoch in range(251):
        total_loss = 0.0
        for inputs, targets in dataset:
            loss, P, cache = forward_pass(inputs, targets, params)
            total_loss += loss
            grads = backward_pass(inputs, targets, params, cache)
            update_parameters(params, grads, inputs, lr)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Average Loss: {total_loss / len(dataset):.4f}")

    # Generation helper
    def generate(prompt, max_tokens=5):
        toks = [word2id[w] for w in prompt.split()]
        for _ in range(max_tokens):
            cur_T = len(toks)
            X = [params["E"][idx][:] for idx in toks]
            Q, K, V_mat = project_qkv(X, params["W_q"], params["W_k"], params["W_v"])
            scores = compute_scaled_scores(Q, K, scale)
            masked = apply_causal_mask(scores)
            A = compute_attention_weights(masked)
            O = aggregate_values(A, V_mat)
            Z = matmul(O, params["W_head"])
            probs = softmax_row(Z[-1])
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

    # Attention Heatmap
    def inspect_attention(text):
        toks = [word2id[w] for w in text.split()]
        cur_T = len(toks)
        X = [params["E"][idx][:] for idx in toks]
        Q, K, V_mat = project_qkv(X, params["W_q"], params["W_k"], params["W_v"])
        scores = compute_scaled_scores(Q, K, scale)
        masked = apply_causal_mask(scores)
        A = compute_attention_weights(masked)
        words_list = text.split()
        print(f"\nAttention Heatmap for: \"{text}\"")
        print("Token \\ Attends to: " + " ".join(f"{w:>6}" for w in words_list))
        for i, w in enumerate(words_list):
            row_str = " ".join(f"{A[i][j]*100:5.1f}%" for j in range(cur_T))
            print(f"[{i}] {w:>6} -> {row_str}")

    inspect_attention("the cat sat on the")
    inspect_attention("the dog sat on the")


if __name__ == "__main__":
    main()
