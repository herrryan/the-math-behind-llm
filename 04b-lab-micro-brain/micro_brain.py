"""
Stage 1: The Micro-Brain (Bengio 2003 Neural Language Model)
The Math Behind Large Language Models - Lab 01

A complete, self-contained neural next-word predictor in ~80 lines of pure Python.
Zero external libraries: No PyTorch, no TensorFlow, no NumPy.
Dependencies: Python standard library only (math, random).

Corresponds to:
- Chapter 00: Next-Word Prediction & Cross-Entropy Loss
- Chapter 01: Vector Embeddings & Lookups (E matrix)
- Chapter 02: Dot Product (vector similarity)
- Chapter 03: Matrix Multiplication (linear transformation W1, W2)
- Chapter 04: ReLU Non-linear Activation & Backpropagation
"""

import math
import random

# =====================================================================
# 1. Corpus & Vocabulary (Chapter 00)
# =====================================================================
corpus = "the cat sat on the mat the dog sat on the rug"
words = corpus.split()
vocab = sorted(list(set(words)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}
V = len(vocab)          # Vocabulary size |V| = 7
d_embed = 4             # Embedding dimension d = 4 (Chapter 01)
d_hidden = 8            # Hidden layer dimension = 8 (Chapter 03)
lr = 0.1                # Learning rate eta (Chapter 04)

print(f"Vocabulary size |V|: {V}, Words: {vocab}")

# Construct training pairs: (current_word -> next_word)
dataset = [(word2id[words[i]], word2id[words[i+1]]) for i in range(len(words)-1)]

# =====================================================================
# 2. Parameter Initialization (Chapters 01 & 03)
# =====================================================================
random.seed(42)
def init_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

E  = init_matrix(V, d_embed)         # Embedding matrix E in R^{|V| x d} (Chapter 01)
W1 = init_matrix(d_embed, d_hidden)  # Layer 1 projection W1 (Chapter 03)
b1 = [0.0] * d_hidden                # Layer 1 bias b1
W2 = init_matrix(d_hidden, V)        # Layer 2 projection to vocab W2 (Chapter 03)
b2 = [0.0] * V                       # Layer 2 bias b2

# =====================================================================
# 3. Training Loop: Forward, Backward, and SGD Update (Chapter 04)
# =====================================================================
print("\nTraining Pure Python Neural Network...")

for epoch in range(121):
    total_loss = 0.0
    
    for x_id, y_target in dataset:
        # -------------------------------------------------------------
        # FORWARD PASS: Left-to-Right Prediction
        # -------------------------------------------------------------
        # Step A: Embedding lookup (Chapter 01: x = e_i^T * E)
        x_vec = E[x_id]
        
        # Step B: Hidden linear transformation z1 = x * W1 + b1 (Chapter 03)
        z1 = [sum(x_vec[k] * W1[k][j] for k in range(d_embed)) + b1[j] for j in range(d_hidden)]
        
        # Step C: Non-linear ReLU activation a1 = max(0, z1) (Chapter 04)
        a1 = [max(0.0, val) for val in z1]
        
        # Step D: Output logits z2 = a1 * W2 + b2 (Chapter 03)
        z2 = [sum(a1[k] * W2[k][j] for k in range(d_hidden)) + b2[j] for j in range(V)]
        
        # Step E: Softmax probability distribution (Chapter 00)
        max_z2 = max(z2)  # Numerical stability trick
        exp_z2 = [math.exp(val - max_z2) for val in z2]
        sum_exp = sum(exp_z2)
        probs = [val / sum_exp for val in exp_z2]
        
        # Step F: Cross-entropy loss L = -log(probs[target]) (Chapter 00)
        loss = -math.log(max(probs[y_target], 1e-12))
        total_loss += loss
        
        # -------------------------------------------------------------
        # BACKWARD PASS: Right-to-Left Credit Assignment (Chapter 04)
        # -------------------------------------------------------------
        # 1. Output error signal: dz2 = probs - one_hot
        dz2 = probs[:]
        dz2[y_target] -= 1.0
        
        # 2. Local sensitivity for W2 and b2: dW2 = a1^T * dz2
        dW2 = [[a1[k] * dz2[j] for j in range(V)] for k in range(d_hidden)]
        db2 = dz2[:]
        
        # 3. Propagate error back through W2: da1 = dz2 * W2^T
        da1 = [sum(dz2[j] * W2[k][j] for j in range(V)) for k in range(d_hidden)]
        
        # 4. Propagate through ReLU valve (Chapter 04): dz1 = da1 * (1 if z1 > 0 else 0)
        dz1 = [da1[j] if z1[j] > 0 else 0.0 for j in range(d_hidden)]
        
        # 5. Local sensitivity for W1 and b1: dW1 = x^T * dz1
        dW1 = [[x_vec[k] * dz1[j] for j in range(d_hidden)] for k in range(d_embed)]
        db1 = dz1[:]
        
        # 6. Propagate error back to embedding vector: dx_vec = dz1 * W1^T
        dx_vec = [sum(dz1[j] * W1[k][j] for j in range(d_hidden)) for k in range(d_embed)]
        
        # -------------------------------------------------------------
        # PARAMETER UPDATE: Gradient Descent w <- w - lr * grad (Chapter 04)
        # -------------------------------------------------------------
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
            
        for k in range(d_embed):
            E[x_id][k] -= lr * dx_vec[k]

    if epoch % 20 == 0:
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:3d} | Average Loss: {avg_loss:.4f}")

# =====================================================================
# 4. Autoregressive Generation: Testing the Trained Brain (Chapter 00)
# =====================================================================
print("\nAutoregressive Sequence Generation:")
curr_word = "cat"
generated = [curr_word]

for _ in range(6):
    x_id = word2id[curr_word]
    x_vec = E[x_id]
    z1 = [sum(x_vec[k] * W1[k][j] for k in range(d_embed)) + b1[j] for j in range(d_hidden)]
    a1 = [max(0.0, val) for val in z1]
    z2 = [sum(a1[k] * W2[k][j] for k in range(d_hidden)) + b2[j] for j in range(V)]
    exp_z2 = [math.exp(val - max(z2)) for val in z2]
    probs = [val / sum(exp_z2) for val in exp_z2]
    
    best_next_id = probs.index(max(probs))
    curr_word = id2word[best_next_id]
    generated.append(curr_word)

print("Prompt: 'cat'")
print("Generated:", " ".join(generated))
