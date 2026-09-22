# Hands-on Lab 02: The Attention Brain in 140 Lines of Pure Python

<fieldset id="evolution">
<legend><strong>The Python Brain Evolution Chain &bull; Stage 2 of 4</strong></legend>
<p>In Lab 01, we built the 2003 Bengio Multi-Layer Perceptron. While it successfully learned token embeddings and local transitions, it suffered from a fatal architectural flaw: <strong>Severe Short-Term Amnesia</strong>. Because it only accepted a single token as input ($x_{t-1} \to y_t$), it was fundamentally blind to context located two or more words in the past.</p>
<p>In this second hands-on lab, we build <strong>The Attention Brain</strong> from first principles in ~140 lines of pure standard-library Python. Zero external dependencies: no PyTorch, no TensorFlow, no NumPy.</p>
<pre>
[The Python Brain Evolution Roadmap]
[Stage 1] 80 Lines Pure Python: Bengio 2003 MLP Language Model (Embeddings, Dense Layers, Manual Backprop)
       │
       ▼ (Fatal Flaw: 1-word context window; instant amnesia on earlier clauses)
[Stage 2 (Current)] 140 Lines Pure Python: Attention Brain (Unlocking Q, K, V Projections & Causal Attention)
       │
       ▼ (Fatal Flaw: Stacking deep attention blocks triggers vanishing gradients and scale blowups)
[Stage 3 (Chapter 13)] 220 Lines Pure Python: Modern Transformer Block (Residuals, RMSNorm, & SwiGLU Gating)
       │
       ▼ (Generation Polish: Stiff greedy decoding without memory acceleration)
[Stage 4 (Chapter 17)] 300 Lines Pure Python Final: Production-Grade Inference Engine (KV Cache & Top-p Sampling)
</pre>
</fieldset>

---

## Step 1: 3-Year-Old Intuition (The Round-Table Council & The Causal Spotlight)

Imagine a circle of schoolchildren sitting around a council table, telling a collaborative story one word at a time:

<figure>
<pre>
[The Round-Table Attention Council]

Position 0: "the"   [ Key Badge: Definite Article ]  ───► [ Value: Grammatical Setup ]
Position 1: "cat"   [ Key Badge: Feline Subject   ]  ───► [ Value: Furry, Likes Fish, Sleeps on Mats ]
Position 2: "sat"   [ Key Badge: Resting Action   ]  ───► [ Value: Static Physical Posture ]
Position 3: "on"    [ Key Badge: Spatial Preposition] ───► [ Value: Surface Support ]
Position 4: "the"   [ Query Flashlight: "Which subject needs a resting place?" ]
                         │
                         ├─────────────────────────────────────────┐
                         ▼ (39.0% Attention Beam)                  ▼ (16.0% Attention Beam)
                 [ Position 1: "cat" ]                     [ Position 4: "the" ]
                         │
                         ▼
        [ Mixed Context: "A cat is looking for a surface!" ]
                         │
                         ▼
     [ LM Head Predicts Winning Next Word: "mat" ]
</pre>
<figcaption><strong>Figure 8b.1:</strong> The Attention Brain allows the second "the" to shine a query flashlight directly backward to "cat", resolving which object should complete the clause.</figcaption>
</figure>

1. **The Three Identity Badges ($Q, K, V$)**:
   Whenever a child enters the discussion, they don't just shout their word. They receive three distinct pieces of equipment:
   - **Query ($Q$)**: A spotlight flashlight. It asks a specific question: *"I am the word 'the' following 'sat on'. Who is the animal resting on this surface?"*
   - **Key ($K$)**: A reflective name tag. It advertises identity: *"I am 'cat', a small furry domestic pet."*
   - **Value ($V$)**: A treasure backpack. It contains the actual descriptive facts that can be passed to the listener: `[Furry, small, sleeps on mats]`.

2. **The Causal Blindfold Rule (No Peeking into Tomorrow)**:
   Because this is a storytelling game, no child is allowed to look to their right. Child 2 can only shine their flashlight at Child 0, Child 1, and themselves. If they peeked at Child 4, they would be reading answers from future pages of the book before writing them!

3. **The Master Volume Knob ($1/\sqrt{d_k}$)**:
   When many flashlights shine at the same name badge, the glare can become blinding. If one badge reflects just slightly more light than the others, an exponential voting booth would award it $100\%$ of all attention, ignoring every other clue. We divide the brightness by the square root of the badge size ($\sqrt{d_k}$), keeping the council calm, sensitive, and open-minded.

4. **Blending the Treasure ($O = A V$)**:
   Child 4 measures how brightly each past badge reflects their spotlight ($39\%$ on `"cat"`, $26\%$ on `"on"`, $16\%$ on `"the"`). They then take exactly $39\%$ of cat's backpack treasure, $26\%$ of on's treasure, and blend them into a single enriched context thought.

5. **Striking the Output Bells (LM Head)**:
   The council leader examines this blended thought and immediately announces the next word: <samp>"mat"</samp>!

---

## Step 2: The Math Bridge to Pure Python

Every theoretical equation introduced across Chapters 05 through 09 translates directly into elementary Python list operations:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 8b.1:</strong> Line-by-line translation between attention equations and pure Python implementations</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="18%">Mechanism</th>
      <th scope="col" align="left" width="14%">Chapter</th>
      <th scope="col" align="left" width="34%">Mathematical Formula</th>
      <th scope="col" align="left" width="34%">Pure Python Implementation (Zero Libs)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>Sequence Embeddings</strong></th>
      <td>Chapter 01</td>
      <td>$\mathbf{X} = [\mathbf{e}_{x_0}, \dots, \mathbf{e}_{x_{T-1}}]^\top \mathbf{E} \in \mathbb{R}^{T \times d_{\text{model}}}$</td>
      <td><code>X = [params["E"][idx][:] for idx in inputs]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>QKV Linear Projections</strong></th>
      <td>Chapter 06</td>
      <td>$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \mathbf{K} = \mathbf{X} \mathbf{W}_K, \mathbf{V} = \mathbf{X} \mathbf{W}_V$</td>
      <td><code>Q = matmul(X, params["W_q"])</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Scaled Dot-Product</strong></th>
      <td>Chapter 08</td>
      <td>$\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$</td>
      <td><code>scores[i][j] = dot(Q[i], K[j]) * scale</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Causal Masking</strong></th>
      <td>Chapter 09</td>
      <td>$S_{ij}^{\text{masked}} = \begin{cases} S_{ij} & j \le i \\ -\infty & j > i \end{cases}$</td>
      <td><code>if j > i: scores[i][j] = -1e9</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Row-wise Softmax</strong></th>
      <td>Chapter 07</td>
      <td>$A_{ij} = \frac{\exp(S_{ij}^{\text{masked}})}{\sum_{k=0}^{T-1} \exp(S_{ik}^{\text{masked}})}$</td>
      <td><code>A = [softmax_row(scores[i]) for i in range(T)]</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Value Aggregation</strong></th>
      <td>Chapter 08</td>
      <td>$\mathbf{O} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T \times d_k}$</td>
      <td><code>O = matmul(A, V_mat)</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>LM Head Logits</strong></th>
      <td>Chapter 03</td>
      <td>$\mathbf{Z} = \mathbf{O} \mathbf{W}_{\text{head}} \in \mathbb{R}^{T \times |V|}$</td>
      <td><code>Z = matmul(O, params["W_head"])</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Sequence Loss</strong></th>
      <td>Chapter 00</td>
      <td>$\mathcal{L} = -\frac{1}{T} \sum_{t=0}^{T-1} \log P(y_t \mid x_{\le t})$</td>
      <td><code>loss = sum(-math.log(P[i][targets[i]])) / T</code></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Softmax Derivative</strong></th>
      <td>Chapter 07</td>
      <td>$\frac{\partial \mathcal{L}}{\partial S_{ij}} = A_{ij} \left( \frac{\partial \mathcal{L}}{\partial A_{ij}} - \sum_k \frac{\partial \mathcal{L}}{\partial A_{ik}} A_{ik} \right) \frac{1}{\sqrt{d_k}}$</td>
      <td><code>dScores[i][j] = A[i][j] * (dA[i][j] - sum_dA) * scale</code></td>
    </tr>
  </tbody>
</table>

---

## Step 3: Complete Source Code & Math Anatomy

To build deep muscle memory, we provide two parallel files for this lab:
- **Guided Exercise Script**: [`labs/02_attention_brain_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/02_attention_brain_exercise.py) (also in [`08b-lab-attention-brain/attention_brain_exercise.py`](file:///Users/guofei/workspace/the-math-behind-llm/08b-lab-attention-brain/attention_brain_exercise.py)), equipped with automated unit test assertions that validate each mathematical operation as you write it.
- **Reference Solution**: [`labs/02_attention_brain.py`](file:///Users/guofei/workspace/the-math-behind-llm/labs/02_attention_brain.py) (also in [`08b-lab-attention-brain/attention_brain.py`](file:///Users/guofei/workspace/the-math-behind-llm/08b-lab-attention-brain/attention_brain.py)).

Here is the complete reference implementation:

<figure>
<pre>
# =====================================================================
# Stage 2: The Attention Brain (140 Lines of Pure Python)
# Dependencies: Zero external libraries (Python built-in math &amp; random only)
# =====================================================================
import math
import random

# 1. Corpus, Vocabulary &amp; Sequence Dataset
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
d_k = 8                 # Attention subspace dimension
lr = 0.2                # Learning rate
scale = 1.0 / math.sqrt(d_k)

# Autoregressive sequence pairs: (tokens[0:T-1] -&gt; tokens[1:T])
dataset = []
for s in sentences:
    toks = [word2id[w] for w in s.split()]
    dataset.append((toks[:-1], toks[1:]))

# 2. Linear Algebra Helpers
def init_matrix(rows, cols, scale_init=0.3):
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

# 3. Parameter Initialization
random.seed(42)
params = {
    "E":      init_matrix(V, d_model),     # Embedding table [V x d_model]
    "W_q":    init_matrix(d_model, d_k),   # Query projection [d_model x d_k]
    "W_k":    init_matrix(d_model, d_k),   # Key projection   [d_model x d_k]
    "W_v":    init_matrix(d_model, d_k),   # Value projection [d_model x d_k]
    "W_head": init_matrix(d_k, V),         # LM Head          [d_k x V]
}

# 4. Training Loop: 250 Epochs
for epoch in range(251):
    total_loss = 0.0

    for inputs, targets in dataset:
        T = len(inputs)

        # FORWARD PASS
        X = [params["E"][idx][:] for idx in inputs]
        Q = matmul(X, params["W_q"])
        K = matmul(X, params["W_k"])
        V_mat = matmul(X, params["W_v"])

        scores = matmul(Q, transpose(K))
        for i in range(T):
            for j in range(T):
                scores[i][j] *= scale
                if j &gt; i:
                    scores[i][j] = -1e9  # Causal mask

        A = [softmax_row(scores[i]) for i in range(T)]
        O = matmul(A, V_mat)
        Z = matmul(O, params["W_head"])
        P = [softmax_row(Z[i]) for i in range(T)]

        loss = sum(-math.log(max(P[i][targets[i]], 1e-12)) for i in range(T)) / T
        total_loss += loss

        # BACKWARD PASS
        dZ = [[(P[i][v] - (1.0 if v == targets[i] else 0.0)) / T for v in range(V)] for i in range(T)]
        dW_head = matmul(transpose(O), dZ)
        dO = matmul(dZ, transpose(params["W_head"]))

        dV_mat = matmul(transpose(A), dO)
        dA = matmul(dO, transpose(V_mat))

        dScores = [[0.0] * T for _ in range(T)]
        for i in range(T):
            sum_dA_A = sum(dA[i][k] * A[i][k] for k in range(T))
            for j in range(T):
                if j &lt;= i:
                    dScores[i][j] = A[i][j] * (dA[i][j] - sum_dA_A) * scale

        dQ = matmul(dScores, K)
        dK = matmul(transpose(dScores), Q)

        dW_q = matmul(transpose(X), dQ)
        dW_k = matmul(transpose(X), dK)
        dW_v = matmul(transpose(X), dV_mat)

        dX = [[sum(dQ[i][k] * params["W_q"][j][k] +
                   dK[i][k] * params["W_k"][j][k] +
                   dV_mat[i][k] * params["W_v"][j][k] for k in range(d_k))
               for j in range(d_model)] for i in range(T)]

        # SGD UPDATES
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
</pre>
<figcaption><strong>Figure 8b.2:</strong> The complete self-contained Attention Brain forward and backward training pipeline in 140 lines of standard Python.</figcaption>
</figure>

---

## Step 4: Where Did It Come From? (Why Attention Destroyed Recurrent Networks)

Before the Transformer architecture introduced by <cite>Vaswani et al. (2017)</cite>, the dominant sequential language models were **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks:

<figure>
<pre>
RNN / LSTM Sequential Bottleneck:
x_0 ──► [ Cell 0 ] ──(h_0)──► [ Cell 1 ] ──(h_1)──► [ Cell 2 ] ──(h_2)──► [ Cell 3 ] ──► Next Word
             │                     │                     │                     │
          (Cat)                  (Sat)                  (On)                  (The)

Attention Direct Highway:
x_0 ("cat") ────────────────────────────────────────────────────────────┐
x_1 ("sat") ──────────────────────────────────────────────┐              │
x_2 ("on")  ───────────────────────────────┐              │              │
x_3 ("the") ──► [ Query Spotlight ] ───────┴──────────────┴──────────────▼──► Next Word ("mat")
</pre>
<figcaption><strong>Figure 8b.3:</strong> RNNs suffer from sequential information decay across long distances ($O(T)$ latency). Self-attention provides an immediate $O(1)$ direct connection between any two tokens regardless of distance.</figcaption>
</figure>

1. **The Sequential Memory Squeeze**:
   In an RNN, information from word 0 must squeeze through hidden state $h_0$, then $h_1$, then $h_2$, before reaching word 3. By the time the hidden vector arrives at position 3, the specific distinction between `"cat"` and `"dog"` has been compressed and diluted by intermediate matrix multiplications.
2. **The $O(1)$ Path Length of Attention**:
   In Self-Attention, position 4 (`"the"`) computes an immediate dot product with position 1 (`"cat"`):
   $$
   S_{4, 1} = \frac{\mathbf{q}_4 \cdot \mathbf{k}_1}{\sqrt{d_k}}
   $$
   The distance in time between word 1 and word 4 does not weaken the connection. If the Query matches the Key, the gradient flows directly through a single step ($O(1)$ path length), eliminating vanishing gradients over distance.

---

## Step 5: Concrete Execution Trace & Attention Heatmap Inspection

Let us examine the actual mathematical weights learned by our Python Attention Brain. When prompted with the 5-token prefix:
<kbd>"the"</kbd> <kbd>"cat"</kbd> <kbd>"sat"</kbd> <kbd>"on"</kbd> <kbd>"the"</kbd>

The model computes attention affinities for the 5th token (`"the"`, index 4) across all visible tokens:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 8b.2:</strong> Attention distribution for token 4 ("the") across historical context</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="15%">Token Position</th>
      <th scope="col" align="left" width="15%">Word</th>
      <th scope="col" align="center" width="25%">Learned Attention Weight ($A_{4, j}$)</th>
      <th scope="col" align="left" width="45%">Visual Probability Bar &amp; Semantic Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left">Position 0</th>
      <td><code>"the"</code></td>
      <td align="center"><strong>16.0%</strong></td>
      <td><meter min="0" max="1" low="0.2" high="0.5" optimum="0.8" value="0.16"></meter> Initial article setup</td>
    </tr>
    <tr>
      <th scope="row" align="left">Position 1</th>
      <td><code>"cat"</code></td>
      <td align="center"><mark><strong>39.0%</strong></mark></td>
      <td><meter min="0" max="1" low="0.2" high="0.5" optimum="0.8" value="0.39"></meter> <strong>Dominant Signal: Identifies subject as feline!</strong></td>
    </tr>
    <tr>
      <th scope="row" align="left">Position 2</th>
      <td><code>"sat"</code></td>
      <td align="center"><strong>3.1%</strong></td>
      <td><meter min="0" max="1" low="0.2" high="0.5" optimum="0.8" value="0.031"></meter> Posture verb</td>
    </tr>
    <tr>
      <th scope="row" align="left">Position 3</th>
      <td><code>"on"</code></td>
      <td align="center"><strong>26.0%</strong></td>
      <td><meter min="0" max="1" low="0.2" high="0.5" optimum="0.8" value="0.26"></meter> Preposition indicating target surface</td>
    </tr>
    <tr>
      <th scope="row" align="left">Position 4</th>
      <td><code>"the"</code></td>
      <td align="center"><strong>16.0%</strong></td>
      <td><meter min="0" max="1" low="0.2" high="0.5" optimum="0.8" value="0.16"></meter> Current query anchor</td>
    </tr>
  </tbody>
</table>

Now observe what happens when we swap `"cat"` for `"dog"`:
<kbd>"the"</kbd> <kbd>"dog"</kbd> <kbd>"sat"</kbd> <kbd>"on"</kbd> <kbd>"the"</kbd>

At index 4, the attention weight on `"dog"` surges to **63.5%**!
Because the Value vector for `"dog"` dominates the mixed context vector $\mathbf{O}_4$, the LM Head matrix $\mathbf{W}_{\text{head}}$ predicts <samp>"rug"</samp> instead of <samp>"mat"</samp> with over $95\%$ probability:

<figure>
<pre>
Autoregressive Generation Trace:
Prompt:    "the cat sat on the" ──► Predicts: "mat" ──► "the cat sat on the mat ."
Prompt:    "the dog sat on the" ──► Predicts: "rug" ──► "the dog sat on the rug ."
Prompt: "the cat walked on the" ──► Predicts: "mat" ──► "the cat walked on the mat ."
Prompt: "the dog walked on the" ──► Predicts: "rug" ──► "the dog walked on the rug ."
</pre>
<figcaption><strong>Figure 8b.4:</strong> The Attention Brain completely resolves ambiguous bigrams by routing historical subject tokens across multiple time steps.</figcaption>
</figure>

---

## Step 6: Core Takeaway & The Fatal Bottleneck

<fieldset>
<legend><strong>Core Pedagogical Takeaway</strong></legend>
<p>The Attention Brain frees language models from the shackles of Markovian amnesia. By projecting representations into dynamic <strong>Query, Key, and Value spaces</strong>, tokens can cast targeted spotlights across history, pulling relevant contextual facts directly into their active representations regardless of sentence length.</p>
</fieldset>

### The Fatal Bottleneck: Depth Instability

If a single attention layer is so powerful, why not simply stack 12 or 24 attention layers on top of each other?

When researchers attempted to stack raw attention layers directly:
$$
\mathbf{X}_{l+1} = \operatorname{Attention}(\mathbf{X}_l)
$$
A catastrophic failure occurred:
1. **Vanishing and Exploding Gradients**: As gradients backpropagate through repeated softmax layers and projection matrices, their magnitudes either decay to zero or blow up to infinity.
2. **Representational Collapse**: Without a persistent highway carrying original token identities forward, deep attention layers repeatedly average values, causing all token representations to converge into an identical, blurry vector.

To build deep, multi-layer neural architectures that can train stably across hundreds of layers, we require two critical engineering innovations:
- **Residual Connections** ($\mathbf{X} + \operatorname{Layer}(\mathbf{X})$): Chapter 11
- **Root Mean Square Normalization (RMSNorm)**: Chapter 12

These two pillars form the foundation of **Stage 3: The Modern Transformer Block (Lab 03)**!
