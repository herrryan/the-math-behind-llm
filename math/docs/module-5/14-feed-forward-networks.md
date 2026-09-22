# Chapter 14: The Thinking Chamber (Feed-Forward Networks)


## Step 1: 3-Year-Old Intuition (The Private Study Desk & Fact Library) {: #step-1 }

!!! note "3-Year-Old Intuition: The Round Table Gossip vs. The Private Study Desk"
    Imagine a classroom of curious detective children trying to solve a big mystery puzzle.

    In the first part of their day, they all sit together around a big circular table (the **Attention Round Table** we met in Chapter 05):
    - The child holding the clue card `"capital"` looks over at the child holding the card `"France"`.
    - They whisper, compare notes, and exchange hints.
    - By the time they finish talking, the `"France"` card has shared its context with the `"capital"` card.

    But exchanging hints is only half the battle! Talking to your neighbors doesn't magically create new knowledge if nobody actually looks up the facts.

    So, the teacher rings a bell. All thirty children stand up from the round table, walk over to their own **private study desks**, and close their doors:

    1. Each child sits alone at their desk. There is no talking, no whispering, and zero looking at other children.
    2. On each desk sits a giant **encyclopedia bookshelf** containing thousands of factual cards.
    3. The child reads the blended clues they gathered at the round table (e.g., `"capital"` + `"France"`).
    4. They flip through their private encyclopedia, find the matching knowledge card, and shout: <samp>"Paris!"</samp>
    5. Having retrieved this factual memory, they pack it into their backpack and return to the classroom.

    In Large Language Models, this private study desk is the **Feed-Forward Network (<dfn id="def-ffn">FFN</dfn>)**, also known as the **Multi-Layer Perceptron (<dfn id="def-mlp">MLP</dfn>)**.

    While Attention is where tokens *talk to each other*, the Feed-Forward Network is where each token *thinks in private* and pulls stored factual memories out of the model's vast synaptic library!

<figure>
<pre>
Attention Layer (Communication Phase):
  [ Token 1: "The" ] ◄──► [ Token 2: "capital" ] ◄──► [ Token 3: "of France" ]
  (Tokens look at each other and mix contextual clues)
                           │
                           ▼
Feed-Forward Network (Thinking & Knowledge Retrieval Phase):
  [ Token 1 ] ──► [ Private Encyclopedia ] ──► [ Updated Meaning 1 ]
  [ Token 2 ] ──► [ Private Encyclopedia ] ──► [ Updated Meaning 2 ]
  [ Token 3 ] ──► [ Private Encyclopedia ] ──► [ "Paris" Fact Retrieved! ]
  (Zero token interaction! Each token processes its vector in isolation)
</pre>
<figcaption><strong>Figure 14.1:</strong> Attention mixes information across time steps (horizontal communication), whereas the Feed-Forward Network transforms each token vector individually (vertical memory retrieval).</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: Why Can't Attention Do Everything Alone?"
    Self-Attention is fundamentally a **linear weighted blending mechanism**:



    $$
    \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
    $$



    Even though the Softmax weights are non-linear, the output for each token is strictly a convex combination (a weighted average) of the existing value vectors $\mathbf{v}_1, \dots, \mathbf{v}_n$.

    Weighted averages can redistribute existing features, but they cannot perform complex non-linear feature synthesis, logic gating, or store billions of factual associations (like *"What year was Ada Lovelace born?"* or *"What is the capital of Peru?"*).

    Furthermore, in a standard Transformer, approximately **two-thirds of all trainable parameters** reside inside the Feed-Forward Networks, not the Attention heads!

    *"How does a two-layer neural network take a single token's vector representation, project it into a higher-dimensional thinking space, query its stored memories through non-linear activation gates, and compress it back down into the model's communication stream?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Classic Transformer FFN (Vaswani et al., 2017)

In the original Transformer architecture, the Feed-Forward sub-layer is applied to each token vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ separately and identically (point-wise):



$$
\operatorname{FFN}(\mathbf{x}) = \sigma\left(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1\right)\mathbf{W}_2 + \mathbf{b}_2
$$



where:
- $\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{model}}}$ is the row vector of a single token.
- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ expands the vector into a wide intermediate thinking space. In the 2017 Transformer, $d_{\text{ff}} = 4 \times d_{\text{model}}$ (e.g., from 512 up to 2048, or from 4096 up to 16384).
- $\mathbf{b}_1 \in \mathbb{R}^{1 \times d_{\text{ff}}}$ is an intermediate bias vector.
- $\sigma(\cdot)$ is a non-linear activation function (historically $\operatorname{ReLU}$, later $\operatorname{GELU}$).
- $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ projects the expanded representation back down to the model dimension.
- $\mathbf{b}_2 \in \mathbb{R}^{1 \times d_{\text{model}}}$ is the output bias vector.

---

### 2. The Modern Standard: The SwiGLU Gated MLP (Shazeer, 2020)

Modern LLMs (including LLaMA 1/2/3, Mistral, Gemma, and DeepSeek) have replaced the simple two-matrix FFN with the **SwiGLU Gated Multi-Layer Perceptron** and eliminated all bias terms ($\mathbf{b} = \mathbf{0}$).

Instead of one expansion matrix $\mathbf{W}_1$, SwiGLU uses **two parallel projection matrices**:
1. $\mathbf{W}_{\text{gate}}$: Computes a dynamic gating vector via the $\operatorname{SiLU}$ (Swish) activation function.
2. $\mathbf{W}_{\text{up}}$: Projects the vector linearly into the thinking space.

The two branches are multiplied element-wise (Hadamard product $\odot$) before being projected back down by $\mathbf{W}_{\text{down}}$:



$$
\operatorname{SwiGLU}(\mathbf{x}) = \left(\operatorname{SiLU}\left(\mathbf{x}\mathbf{W}_{\text{gate}}\right) \odot \left(\mathbf{x}\mathbf{W}_{\text{up}}\right)\right) \mathbf{W}_{\text{down}}
$$



where:
- $\operatorname{SiLU}(z) = z \cdot \operatorname{sigmoid}(z) = \frac{z}{1 + e^{-z}}$.
- $\odot$ denotes element-wise multiplication.
- To keep the total parameter count identical to a standard $4d$ FFN (which has $2 \times d \times 4d = 8d^2$ parameters), SwiGLU sets the intermediate dimension to approximately:



$$
d_{\text{ff}} \approx \frac{8}{3} d_{\text{model}}
$$



Because $3 \times d \times \left(\frac{8}{3}d\right) = 8d^2$, the three matrices ($\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$) contain exactly the same total weight budget while delivering substantially superior empirical reasoning and perplexity scores!

<figure>
<pre>
The SwiGLU Gated Feed-Forward Architecture:

                     Token Vector x  [ 1 x d ]
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       x * W_gate                  x * W_up
             │                           │
             ▼                           │
        SiLU( ... )                      │
             │                           │
             └───────────► ( ⊙ ) ◄───────┘
                       Element-wise
                         Product
                            │
                            ▼
                     [ 1 x d_ff ]
                            │
                            ▼
                         * W_down
                            │
                            ▼
                   Output y  [ 1 x d ]
</pre>
<figcaption><strong>Figure 14.2:</strong> SwiGLU splits the expansion into a continuous gate branch and a value branch, enabling multiplicative feature filtering before down-projection.</figcaption>
</figure>

---

## Step 4: Where Did It Come From? (FFNs as Key-Value Memories) {: #step-4 }

Why did researchers write the Feed-Forward layer as an expansion followed by a contraction ($\mathbb{R}^d \to \mathbb{R}^{4d} \to \mathbb{R}^d$)?

In a groundbreaking 2021 research paper titled *"Transformer Feed-Forward Layers Are Key-Value Memories"*, Mor Geva and colleagues at Tel Aviv University revealed the true mathematical nature of the FFN:

<figure>
<pre>
FFN as an Associative Key-Value Memory Bank:

Input Vector x
      │
      ▼
   W_1 (Memory Keys):  Each column k_i is a pattern detector.
                       "Does x mention French royalty?"
                       "Is x followed by a year?"
                       Activations a_i = sigma(x * k_i) are match scores!
      │
      ▼
   W_2 (Memory Values): Each row v_i is a factual concept update!
                        If key k_i fires strongly (a_i ≈ 1),
                        its corresponding value v_i ("Paris", "1789")
                        is added to the output stream:
                        y = sum_i  a_i * v_i
</pre>
<figcaption><strong>Figure 14.3:</strong> The first matrix acts as a collection of memory keys that detect input patterns; the second matrix acts as memory values that inject factual knowledge.</figcaption>
</figure>

Mathematically, the output of the FFN can be written as a sum of memory vectors:



$$
\operatorname{FFN}(\mathbf{x}) = \sum_{m=1}^{d_{\text{ff}}} \underbrace{\sigma\left(\mathbf{x}\mathbf{k}_m\right)}_{\text{Pattern Match Score } a_m} \cdot \underbrace{\mathbf{v}_m}_{\text{Memory Value Vector}}
$$



1. $\mathbf{k}_m$ (the $m$-th column of $\mathbf{W}_1$): An encyclopedia **Key** that detects a specific linguistic or factual feature.
2. $a_m \in [0, \infty)$: The **Match Intensity** (activation level) of that memory key.
3. $\mathbf{v}_m$ (the $m$-th row of $\mathbf{W}_2$): The factual **Value** injected into the token vector.

This explains why scaling up $d_{\text{ff}}$ (and building Mixture-of-Experts like Mixtral or DeepSeek-V3 with dozens of FFNs) directly increases the model's factual knowledge base!

---

## Step 5: Concrete Toy Example (Step-by-Step Hand Arithmetic) {: #step-5 }

Let us compute a complete forward pass through a miniature Feed-Forward Network by hand.

### 1. Miniature Dimensions & Parameters
- Input dimension: $d_{\text{model}} = 2$
- Intermediate expansion dimension: $d_{\text{ff}} = 4$
- Activation function: $\operatorname{ReLU}(z) = \max(0, z)$
- No bias terms ($\mathbf{b}_1 = \mathbf{0}, \mathbf{b}_2 = \mathbf{0}$).

Let our incoming token vector be:



$$
\mathbf{x} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
$$



Let the first layer weights (Memory Keys $\mathbf{W}_1 \in \mathbb{R}^{2 \times 4}$) be:



$$
\mathbf{W}_1 = \begin{bmatrix}
2.0 & -1.0 &  0.0 &  1.0 \\
1.0 &  3.0 & -2.0 & -1.0
\end{bmatrix}
$$



Let the second layer weights (Memory Values $\mathbf{W}_2 \in \mathbb{R}^{4 \times 2}$) be:



$$
\mathbf{W}_2 = \begin{bmatrix}
1.0 &  0.0 \\
0.0 &  1.0 \\
2.0 & -1.0 \\
-1.0 &  1.0
\end{bmatrix}
$$



---

### 2. Step-by-Step Forward Execution

<fieldset>
<legend><strong>Execution Checklist</strong></legend>
<p><input type="checkbox" checked disabled> <strong>Step A:</strong> Project into expansion space $\mathbf{z} = \mathbf{x}\mathbf{W}_1$.</p>
<p><input type="checkbox" checked disabled> <strong>Step B:</strong> Apply non-linear activation $\mathbf{a} = \operatorname{ReLU}(\mathbf{z})$.</p>
<p><input type="checkbox" checked disabled> <strong>Step C:</strong> Project back to model dimension $\mathbf{y} = \mathbf{a}\mathbf{W}_2$.</p>
</fieldset>

#### Step A: Matrix Multiplication $\mathbf{z} = \mathbf{x}\mathbf{W}_1$
Multiply the $1 \times 2$ vector by each of the 4 columns of $\mathbf{W}_1$:

- $z_1 = (1.0 \times 2.0) + (2.0 \times 1.0) = 2.0 + 2.0 = 4.0$
- $z_2 = (1.0 \times -1.0) + (2.0 \times 3.0) = -1.0 + 6.0 = 5.0$
- $z_3 = (1.0 \times 0.0) + (2.0 \times -2.0) = 0.0 - 4.0 = -4.0$
- $z_4 = (1.0 \times 1.0) + (2.0 \times -1.0) = 1.0 - 2.0 = -1.0$



$$
\mathbf{z} = \begin{bmatrix} 4.0 & 5.0 & -4.0 & -1.0 \end{bmatrix}
$$



#### Step B: Non-Linear Activation $\mathbf{a} = \operatorname{ReLU}(\mathbf{z})$
Apply $\max(0, z)$ element-wise:
- $a_1 = \max(0, 4.0) = 4.0$ (Key 1 matches strongly!)
- $a_2 = \max(0, 5.0) = 5.0$ (Key 2 matches strongly!)
- $a_3 = \max(0, -4.0) = 0.0$ (Key 3 did not match; inhibited!)
- $a_4 = \max(0, -1.0) = 0.0$ (Key 4 did not match; inhibited!)



$$
\mathbf{a} = \begin{bmatrix} 4.0 & 5.0 & 0.0 & 0.0 \end{bmatrix}
$$



<mark>Notice the sparsity: Keys 3 and 4 were zeroed out completely!</mark>

#### Step C: Down-Projection $\mathbf{y} = \mathbf{a}\mathbf{W}_2$
Multiply the active match scores by the memory values:



$$
\mathbf{y} = 4.0 \times \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} + 5.0 \times \begin{bmatrix} 0.0 & 1.0 \end{bmatrix} + 0.0 \times \mathbf{v}_3 + 0.0 \times \mathbf{v}_4
$$





$$
y_1 = (4.0 \times 1.0) + (5.0 \times 0.0) + (0.0 \times 2.0) + (0.0 \times -1.0) = 4.0
$$





$$
y_2 = (4.0 \times 0.0) + (5.0 \times 1.0) + (0.0 \times -1.0) + (0.0 \times 1.0) = 5.0
$$





$$
\mathbf{y} = \begin{bmatrix} 4.0 & 5.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
$$



The token vector entered as $[1.0, 2.0]$ and emerged from its thinking chamber enriched with new factual coordinates $[4.0, 5.0]$, ready to be added back to the residual highway!

---

## Step 6: Core Takeaway {: #step-6 }

!!! tip "Key Insight: The Punchline of Feed-Forward Networks"
    **If Self-Attention is the telephone network where tokens share context across the sentence, the Feed-Forward Network is the private library where each token recalls stored facts and synthesizes new knowledge.**

    Modern LLMs implement this memory bank through **SwiGLU gated MLPs**, which dynamically open and close feature channels to inject precise factual memories into the residual stream.
