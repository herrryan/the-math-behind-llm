"""
Stage 1: The Micro-Brain (Bengio 2003 Neural Language Model) - Guided Exercise
The Math Behind Large Language Models - Lab 01

Instructions:
-------------
In this hands-on lab, you will build a complete neural language model from scratch
using 100% pure Python (no PyTorch, no TensorFlow, no NumPy).

All scaffolding (vocabulary extraction, parameter initialization, training loop harness)
has been provided for you. Your mission is to fill in the core mathematical operations
marked with TODO.

To guide your implementation, every TODO block includes:
  1. Mathematical Formula (LaTeX style)
  2. Physical Intuition & Tensor Shapes
  3. Heuristic Guidance & Python Hints

When you run this script:
  $ python3 micro_brain_exercise.py
It will test your implementation step-by-step and show you training progress.
"""

import math
import random


# =====================================================================
# 1. Corpus, Vocabulary & Dataset Setup (Chapter 00)
# =====================================================================
corpus = "the cat sat on the mat the dog sat on the rug"
words = corpus.split()
vocab = sorted(list(set(words)))
word2id = {w: i for i, w in enumerate(vocab)}
print(f"word2id: {word2id}")
id2word = {i: w for i, w in enumerate(vocab)}

V = len(vocab)          # Vocabulary size |V| = 7
d_embed = 4             # Embedding dimension d = 4 (Chapter 01)
d_hidden = 8            # Hidden layer dimension = 8 (Chapter 03)
lr = 0.1                # Learning rate eta (Chapter 04)

# Construct consecutive bigram training pairs: (current_word_id -> next_word_id)
dataset = [(word2id[words[i]], word2id[words[i+1]]) for i in range(len(words) - 1)]
print(f"dataset: {dataset}")
 
# =====================================================================
# 2. Parameter Initialization (Chapters 01 & 03)
# =====================================================================
def init_matrix(rows, cols, scale=0.1):
    """Initializes a 2D list of shape [rows x cols] with Gaussian random values."""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def init_parameters(seed=42):
    """Initializes all trainable parameters of the micro-brain."""
    random.seed(seed)
    params = {
        "E":  init_matrix(V, d_embed),         # Embedding table: [V x d_embed] (Chapter 01)
        "W1": init_matrix(d_embed, d_hidden),  # Layer 1 weights: [d_embed x d_hidden] (Chapter 03)
        "b1": [0.0] * d_hidden,                # Layer 1 biases:  [d_hidden]
        "W2": init_matrix(d_hidden, V),        # Layer 2 weights: [d_hidden x V] (Chapter 03)
        "b2": [0.0] * V,                       # Layer 2 biases:  [V]
    }
    return params


# =====================================================================
# 3. Core Math Functions to Implement
# =====================================================================

def forward_pass(x_id, y_target, params):
    """
    Executes the left-to-right forward pass of the neural language model.

    Args:
        x_id: int, vocabulary ID of the input word (e.g., 0 for 'cat')
        y_target: int, vocabulary ID of the true next word (e.g., 5 for 'sat')
        params: dict, containing E, W1, b1, W2, b2

    Returns:
        loss: float, cross-entropy loss scalar
        probs: list of float, length V, predicted probability distribution
        cache: tuple, intermediate tensors (x_vec, z1, a1, z2) needed for backprop
    """
    E = params["E"]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # -----------------------------------------------------------------
    # TODO 1: Embedding Lookup (Chapter 01)
    # -----------------------------------------------------------------
    # Math:
    #   x_vec = e_{x_id}^T * E  which extracts row x_id of matrix E
    # Shapes:
    #   E is [V x d_embed], x_id is int -> x_vec must be list of float of length d_embed
    # Heuristic Hint:
    #   In pure Python, you can directly slice row x_id from table E.
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # x_vec = ...
    x_vec = E[x_id]
    # raise NotImplementedError("TODO 1: Implement embedding lookup x_vec from table E.")

    # -----------------------------------------------------------------
    # TODO 2: Layer 1 Linear Transformation (Chapter 03)
    # -----------------------------------------------------------------
    # Math:
    #   z_1 = x_vec * W1 + b1
    #   z_1[j] = sum_{k=0}^{d_embed-1} (x_vec[k] * W1[k][j]) + b1[j]  for j in range(d_hidden)
    # Shapes:
    #   x_vec: [d_embed]
    #   W1:    [d_embed x d_hidden]
    #   b1:    [d_hidden]
    #   z1:    [d_hidden]
    # Heuristic Hint:
    #   Use a list comprehension over j in range(d_hidden) with an inner sum over k.
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    z1 = [0.0] * d_hidden
    for j in range(d_hidden):
        sum = 0.0
        for k in range(d_embed):
            sum += x_vec[k] * W1[k][j]
        z1.append(sum + b1[j])
    # raise NotImplementedError("TODO 2: Implement linear projection z1 = x_vec * W1 + b1.")

    # -----------------------------------------------------------------
    # TODO 3: Non-Linear ReLU Activation (Chapter 04)
    # -----------------------------------------------------------------
    # Math:
    #   a_1 = ReLU(z_1) = max(0, z_1)
    # Physical Intuition:
    #   A one-way ratchet valve: allows positive signals to flow through untouched,
    #   while clamping negative signals to strictly 0.0.
    # Shapes:
    #   z1: [d_hidden] -> a1: [d_hidden]
    # Heuristic Hint:
    #   [max(0.0, val) for val in z1]
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    a1 = [max(0.0, val) for val in z1]
    # raise NotImplementedError("TODO 3: Implement ReLU activation a1 = max(0, z1).")

    # -----------------------------------------------------------------
    # TODO 4: Layer 2 Output Logits Projection (Chapter 03)
    # -----------------------------------------------------------------
    # Math:
    #   z_2 = a_1 * W2 + b2
    #   z_2[j] = sum_{k=0}^{d_hidden-1} (a_1[k] * W2[k][j]) + b2[j]  for j in range(V)
    # Shapes:
    #   a1: [d_hidden]
    #   W2: [d_hidden x V]
    #   b2: [V]
    #   z2: [V] (raw unnormalized logits across all V vocabulary words)
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    z2 = [0.0] * V
    for j in range(V):
        sum = 0.0
        for k in range(d_hidden):
            sum += a1[k] * W2[k][j]
        z2.append(sum + b2[j])
    # raise NotImplementedError("TODO 4: Implement output logits projection z2 = a1 * W2 + b2.")

    # -----------------------------------------------------------------
    # TODO 5: Softmax Probability Distribution (Chapters 00 & 07)
    # -----------------------------------------------------------------
    # Math:
    #   probs[j] = exp(z_2[j] - max(z_2)) / sum_{k=0}^{V-1} exp(z_2[k] - max(z_2))
    # Numerical Stability Heuristic:
    #   Always subtract max(z_2) before math.exp()!
    #   This prevents math range overflow errors (e.g. exp(1000) -> Inf) while
    #   leaving the final mathematical probabilities completely unchanged.
    # Shapes:
    #   z2: [V] -> probs: [V], where all elements in (0, 1) and sum(probs) == 1.0
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    max_z2 = max(z2)
    exp_z2 = [math.exp(val - max_z2) for val in z2]
    sum_exp = sum(exp_z2)
    probs = [val / sum_exp for val in exp_z2]
    # raise NotImplementedError("TODO 5: Implement numerically-stable Softmax.")

    # -----------------------------------------------------------------
    # TODO 6: Cross-Entropy Loss (Chapter 00)
    # -----------------------------------------------------------------
    # Math:
    #   L = -log(probs[y_target])
    # Physical Intuition:
    #   Measures 'how surprised' the model is by the correct word.
    #   If probs[y_target] == 1.0, loss == 0.0.
    #   If probs[y_target] -> 0.0, loss -> +Inf.
    # Safety Hint:
    #   Guard against log(0) using max(probs[y_target], 1e-12).
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    loss = -math.log(max(probs[y_target], 1e-12))
    # raise NotImplementedError("TODO 6: Implement cross-entropy loss L = -log(probs[y_target]).")

    cache = (x_vec, z1, a1, z2)
    print(f"loss: {loss}, probs: {probs}")
    return loss, probs, cache


def backward_pass(x_id, y_target, probs, cache, params):
    """
    Executes the right-to-left backward pass (backpropagation) using the chain rule.

    Args:
        x_id: int, input word ID
        y_target: int, target word ID
        probs: list of float, length V, model's predicted probabilities
        cache: tuple (x_vec, z1, a1, z2) from forward_pass
        params: dict containing E, W1, b1, W2, b2

    Returns:
        grads: dict containing dE_row, dW1, db1, dW2, db2
    """
    x_vec, z1, a1, z2 = cache
    W1 = params["W1"]
    W2 = params["W2"]

    # -----------------------------------------------------------------
    # TODO 7: Output Error Signal dz2 (Chapter 04 & Lab 01)
    # -----------------------------------------------------------------
    # Math:
    #   dz_2 = dL / dz_2 = probs - y_one_hot
    #   That is:
    #     dz_2[j] = probs[j] - 1.0  (if j == y_target)
    #     dz_2[j] = probs[j]        (if j != y_target)
    # Physical Intuition:
    #   The raw error vector: how much probability mass was mistakenly
    #   allocated away from the true target word.
    # Shape: [V]
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # dz2 = probs[:]
    # dz2[y_target] -= 1.0
    dz2 = probs[:]
    dz2[y_target] -= 1.0
    # raise NotImplementedError("TODO 7: Compute output error signal dz2.")

    # -----------------------------------------------------------------
    # TODO 8: Gradients for Layer 2 (W2, b2) and Hidden Signal da1 (Chapter 04)
    # -----------------------------------------------------------------
    # Math:
    #   dL / dW2[k][j] = a1[k] * dz2[j]       (outer product of a1 and dz2)
    #   dL / db2[j]    = dz2[j]
    #   dL / da1[k]    = sum_{j=0}^{V-1} dz2[j] * W2[k][j]  (error propagated back through W2)
    # Shapes:
    #   dW2: [d_hidden x V]
    #   db2: [V]
    #   da1: [d_hidden]
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # dW2 = [[a1[k] * dz2[j] for j in range(V)] for k in range(d_hidden)]
    # db2 = dz2[:]
    # da1 = [sum(dz2[j] * W2[k][j] for j in range(V)) for k in range(d_hidden)]
    dW2 = [[a1[k] * dz2[j] for j in range(V)] for k in range(d_hidden)]
    db2 = dz2[:]  # Copy dz2 into db2

    da1 = [0.0] * d_hidden
    for k in range(d_hidden):
        sum_val = 0.0
        for j in range(V):
            sum_val += dz2[j] * W2[k][j]
        da1.append(sum_val)
    raise NotImplementedError("TODO 8: Compute dW2, db2, and da1.")

    # -----------------------------------------------------------------
    # TODO 9: Backprop Through the ReLU Valve dz1 (Chapter 04)
    # -----------------------------------------------------------------
    # Math:
    #   dz1[j] = da1[j] * ReLU'(z1[j])
    #   where ReLU'(z1[j]) = 1.0 if z1[j] > 0 else 0.0
    # Physical Intuition:
    #   If the valve was open during forward pass (z1 > 0), the error flows
    #   backward unchanged. If it was closed (z1 <= 0), the gradient is blocked!
    # Shape: [d_hidden]
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # dz1 = [da1[j] if z1[j] > 0 else 0.0 for j in range(d_hidden)]
    dz1 = [da1[j] if z1[j] > 0 else 0.0 for j in range(d_hidden)]
    # raise NotImplementedError("TODO 9: Backpropagate through ReLU valve to obtain dz1.")

    # -----------------------------------------------------------------
    # TODO 10: Gradients for Layer 1 (W1, b1) and Embedding dx_vec (Chapter 04)
    # -----------------------------------------------------------------
    # Math:
    #   dL / dW1[k][j] = x_vec[k] * dz1[j]
    #   dL / db1[j]    = dz1[j]
    #   dL / dx_vec[k] = sum_{j=0}^{d_hidden-1} dz1[j] * W1[k][j]
    # Shapes:
    #   dW1:    [d_embed x d_hidden]
    #   db1:    [d_hidden]
    #   dx_vec: [d_embed] (gradient for the specific row E[x_id])
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    # dW1 = [[x_vec[k] * dz1[j] for j in range(d_hidden)] for k in range(d_embed)]
    # db1 = dz1[:]
    # dx_vec = [sum(dz1[j] * W1[k][j] for j in range(d_hidden)) for k in range(d_embed)]
    dW1 = [[x_vec[k] * dz1[j] for j in range(d_hidden)] for k in range(d_embed)]
    db1 = dz1[:]

    dx_vec = [0.0] * d_embed
    for k in range(d_embed):
        sum_val = 0.0
        for j in range(d_hidden):
            sum_val += dz1[j] * W1[k][j]
        dx_vec.append(sum_val)
    raise NotImplementedError("TODO 10: Compute dW1, db1, and dx_vec.")

    grads = {
        "dW2": dW2,
        "db2": db2,
        "dW1": dW1,
        "db1": db1,
        "dx_vec": dx_vec,
    }
    return grads


def sgd_step(params, grads, x_id, lr):
    """
    Executes a Stochastic Gradient Descent (SGD) parameter update step.

    Math:
        theta <- theta - lr * dL / dtheta
    """
    W2 = params["W2"]
    b2 = params["b2"]
    W1 = params["W1"]
    b1 = params["b1"]
    E = params["E"]

    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dx_vec = grads["dx_vec"]

    # -----------------------------------------------------------------
    # TODO 11: Update All Model Parameters Using Learning Rate lr
    # -----------------------------------------------------------------
    # Update:
    #   1. W2[k][j] -= lr * dW2[k][j]
    #   2. b2[j]    -= lr * db2[j]
    #   3. W1[k][j] -= lr * dW1[k][j]
    #   4. b1[j]    -= lr * db1[j]
    #   5. E[x_id][k] -= lr * dx_vec[k]  (only the active embedding row!)
    # -----------------------------------------------------------------
    # YOUR CODE HERE:
    for k in range(d_hidden):
        for j in range(V):
            W2[k][j] -= lr * dW2[k][j]
    for j in range(V):
        b2[j] -= lr * db2[j]
    for k in range(d_embed):
        for j in range(d_hidden):
            W1[k][j] -= lr * dW1[k][j]
    for j in range(d_hidden):
        b1[j] -= lr * db1[j]
    # Update embedding row E[x_id]
    for k in range(d_embed):
        E[x_id][k] -= lr * dx_vec[k]
    # raise NotImplementedError("TODO 11: Implement SGD parameter updates.")


def generate_text(prompt_word, n_tokens, params):
    """
    Autoregressively generates next tokens using greedy decoding (argmax).

    Args:
        prompt_word: str, initial seed word (e.g. 'cat')
        n_tokens: int, number of words to generate
        params: dict of trained parameters

    Returns:
        str, generated text sequence
    """
    curr_word = prompt_word
    output = [curr_word]

    for _ in range(n_tokens):
        # -------------------------------------------------------------
        # TODO 12: Autoregressive Step
        # -------------------------------------------------------------
        # 1. Look up x_id = word2id[curr_word]
        # 2. Run forward pass (you can pass dummy y_target=0 since we only need probs)
        # 3. Find index of maximum probability: best_id = probs.index(max(probs))
        # 4. Map back to word string: next_word = id2word[best_id]
        # 5. Append to output, set curr_word = next_word
        # -------------------------------------------------------------
        # YOUR CODE HERE:
        x_id = word2id[curr_word]
        _, probs, _ = forward_pass(x_id, 0, params)  # y_target is dummy here
        best_id = probs.index(max(probs))
        next_word = id2word[best_id]
        output.append(next_word)
        curr_word = next_word
        # raise NotImplementedError("TODO 12: Implement autoregressive token generation.")

    return " ".join(output)


# =====================================================================
# 4. Interactive Test Harness & Training Loop
# =====================================================================
def run_unit_tests():
    """Runs deterministic sanity checks to verify each TODO before training."""
    print("=" * 70)
    print("Running Sanity Verification on Initial Parameters...")
    print("=" * 70)

    params = init_parameters(seed=42)
    x_id, y_target = dataset[0]  # First pair: ('the' -> 'cat')

    # Test Forward Pass
    try:
        loss, probs, cache = forward_pass(x_id, y_target, params)
        print("[PASS] Forward pass executed successfully.")
        print(f"       Initial step loss: {loss:.4f} (Expected: ~1.9444)")
        assert abs(loss - 1.9444) < 0.05, f"Loss mismatch: got {loss:.4f}, expected ~1.9444"
        assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities do not sum to 1.0!"
    except NotImplementedError as e:
        print(f"[TODO] Forward pass incomplete: {e}")
        return False

    # Test Backward Pass
    try:
        grads = backward_pass(x_id, y_target, probs, cache, params)
        print("[PASS] Backward pass executed successfully.")
        assert len(grads["dW2"]) == d_hidden and len(grads["dW2"][0]) == V
        assert len(grads["dW1"]) == d_embed and len(grads["dW1"][0]) == d_hidden
        assert len(grads["dx_vec"]) == d_embed
    except NotImplementedError as e:
        print(f"[TODO] Backward pass incomplete: {e}")
        return False

    # Test SGD Update
    try:
        sgd_step(params, grads, x_id, lr=0.1)
        print("[PASS] SGD parameter update executed successfully.")
    except NotImplementedError as e:
        print(f"[TODO] SGD update incomplete: {e}")
        return False

    print("\nAll unit tests PASSED! Proceeding to full training loop...\n")
    return True


def train_micro_brain(epochs=121):
    """Executes the full training loop over dataset for specified epochs."""
    params = init_parameters(seed=42)

    print("=" * 70)
    print(f"Training Pure Python Micro-Brain for {epochs} Epochs...")
    print(f"Corpus: '{corpus}'")
    print(f"Vocabulary ({V} tokens): {vocab}")
    print("=" * 70)

    for epoch in range(epochs):
        total_loss = 0.0

        for x_id, y_target in dataset:
            loss, probs, cache = forward_pass(x_id, y_target, params)
            total_loss += loss

            grads = backward_pass(x_id, y_target, probs, cache, params)
            sgd_step(params, grads, x_id, lr)

        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch:3d} | Average Cross-Entropy Loss: {avg_loss:.4f}")

    print("\nTraining completed!")
    print("-" * 70)

    # Test Generation
    try:
        sample_prompt = "cat"
        generated_sentence = generate_text(sample_prompt, n_tokens=6, params=params)
        print(f"Prompt:    '{sample_prompt}'")
        print(f"Generated: '{generated_sentence}'")
        print("-" * 70)
        print("Congratulations! You have implemented a neural language model from first principles!")
    except NotImplementedError as e:
        print(f"[TODO] Generation incomplete: {e}")


if __name__ == "__main__":
    if run_unit_tests():
        train_micro_brain(epochs=121)
