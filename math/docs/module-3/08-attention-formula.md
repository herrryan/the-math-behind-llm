# Chapter 08: The Attention Formula & Why We Divide by $\sqrt{d_k}$


## Step 1: 3-Year-Old Intuition (The Whispering Hall & The Volume Knob) {: #step-1 }

!!! note "3-Year-Old Intuition: The Megaphone Shouting Match & The Master Volume Knob"
    Imagine a large classroom with 64 children sitting in a circle. Each child holds a megaphone.

    When one child asks a question, every other child shouts their answer through their megaphone. But there is a catch: each megaphone is powered by **64 separate battery cells**.

    When 64 battery cells combine their electrical energy, the sound doesn't just add up a little—it multiplies into an ear-splitting roar! If Child 2 is just slightly more excited than Child 3, their 64 batteries amplify that tiny difference into a deafening sonic boom. Meanwhile, Child 1's helpful answer is completely drowned out into absolute silence.

    What happens to the teacher's ears?

    The sound is so overwhelmingly loud that the teacher's eardrums pop! Once your ears are ringing from a sonic boom, you can no longer hear any subtle changes in voice. If Child 2 whispers or screams even louder, your overloaded ears register no difference at all. Your ears have **frozen**.

    The wise audio engineer solves this with a **Master Volume Knob**:

    Before the sound signals reach the voting table, the engineer passes all audio cables through a dampener that divides the total electrical power by the square root of the battery count (dividing the volume by 8, since 8 × 8 = 64):

    1. **Dampen the Blasts**: The deafening sonic boom is turned down to a healthy, energetic speaking volume.
    2. **Protect the Ears**: The teacher's eardrums never pop. They can hear Child 2's enthusiasm clearly, while still catching Child 1's quiet advice.
    3. **Keep the Teacher Sensitive**: Because the sound stays in the comfortable hearing range, any small improvement from any child is instantly heard and praised!

    In an LLM, dividing by the square root of the battery count is that exact **Master Volume Knob**. It stops the signal from exploding, prevents the listener from going deaf, and keeps attention sensors razor-sharp!

<figure>
<pre>
Without Volume Knob (Raw Power Multiplies Across 64 Battery Cells):
[64 Battery Cells] ──► [Roar: +24, -8, +3] ──► [Ears Overloaded: 99.9999%, 0.0000%, 0.0001%] ──► Ears Ringing (Brain Deaf!)

With Volume Knob (Dividing Power by √64 = 8):
[64 Battery Cells] ──► [Volume: +3.0, -1.0, +0.38] ──► [Attentive Hearing: 88.2%, 1.6%, 10.2%] ──► Ears Sensitive (Brain Learns!)
</pre>
<figcaption><strong>Figure 8.1:</strong> The Master Volume Knob prevents signal energy from saturating the listener into an unteachable coma.</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: From Dot Products and Softmax to the Unified Attention Engine"
    In Chapter 06, we learned how each word projects into three distinct roles:

    - **Query** ($\mathbf{q}$): What clue am I searching for?
    - **Key** ($\mathbf{k}$): What label do I offer to searchers?
    - **Value** ($\mathbf{v}$): What core information do I carry?

    In Chapter 07, we discovered the **Softmax function**, which transforms unbounded real-valued scores into legitimate probability percentages that sum to strictly $100\%$.

    But when we assemble these pieces into a complete, production-grade neural network, a perilous mathematical trap appears:

    In real models, embedding dimensions are not tiny numbers like $2$; they are typically $d_k = 64$, $d_k = 128$, or even higher. The dot product sums over $d_k$ separate multiplications:

    $$
    \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i = q_1 k_1 + q_2 k_2 + \dots + q_{d_k} k_{d_k}
    $$

    As $d_k$ grows large, the variance of this sum grows linearly with $d_k$, pushing raw scores to extreme positive and negative values ($\pm 20$, $\pm 30$). Softmax on these huge inputs produces outputs of $1.0000$ and $0.0000$, driving its gradients to **absolute zero**. The model completely stops learning!

    This brings us to the bridging question of the Transformer architecture:

    *"How do we fuse Query matching, Key indexing, Softmax normalization, and Value retrieval into a single, fully parallel matrix equation, and what exact mathematical factor stabilizes the variance so gradients never vanish, regardless of vector dimension?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Master Attention Formula

The complete, iconic equation governing Scaled Dot-Product Attention in the Transformer is defined as:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

Let us break down the tensor dimensions and dataflow across this pipeline.

\lt figure>
\lt pre>
  Q Matrix              K^T Matrix                 Raw Logits                Attention Weights (A)             V Matrix                Output Context (O)
 [T × d_k]             [d_k × T]                   [T × T]                          [T × T]                   [T × d_v]                    [T × d_v]
┌─────────┐           ┌──────────────┐          ┌──────────────┐                 ┌──────────────┐            ┌─────────┐                  ┌─────────┐
│ q_1     │     ×     │ k_1  k_2  k_3│    =     │ S_11 S_12 ...│   ──÷√d_k──►    │ A_11 A_12 ...│      ×     │ v_1     │        =         │ c_1     │
│ q_2     │           │              │          │ S_21 S_22 ...│   ──Softmax─►   │ A_21 A_22 ...│            │ v_2     │                  │ c_2     │
│ q_3     │           │              │          │ S_31 S_32 ...│                 │ A_31 A_32 ...│            │ v_3     │                  │ c_3     │
└─────────┘           └──────────────┘          └──────────────┘                 └──────────────┘            └─────────┘                  └─────────┘
  Queries               Keys (Transposed)       All-to-All Scores                Probability Routing           Values                      Blended Context
</pre>
\lt figcaption>\lt strong>Figure 8.2:</strong> Linear algebra tensor dimensions for the Scaled Dot-Product Attention pipeline.</figcaption>
</figure>

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 8.1:</strong> Complete Mathematical Symbol Definitions for the Attention Formula</caption>
  \lt thead>
    \lt tr bgcolor="#eae9e1">
      \lt th scope="col" align="left" width="22%">Symbol</th>
      \lt th scope="col" align="left" width="22%">Type &amp; Dimension</th>
      \lt th scope="col" align="left" width="56%">Mathematical Meaning &amp; Computational Role</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{Q}$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times d_k}$</td>
      \lt td>Packed Query vectors for all $T$ sequence tokens, where row $i$ represents $\mathbf{q}_i^\top$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{K}$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times d_k}$</td>
      \lt td>Packed Key vectors for all $T$ sequence tokens, where row $j$ represents $\mathbf{k}_j^\top$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{K}^\top$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{d_k \times T}$</td>
      \lt td>Transpose of the Key matrix. Transposing allows parallel dot products via matrix multiplication.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{Q}\mathbf{K}^\top$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times T}$</td>
      \lt td>Raw attention affinity matrix. Entry $(i, j)$ is the unscaled dot product $\mathbf{q}_i^\top \mathbf{k}_j$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$d_k$</th>
      \lt td align="left">Scalar $\in \mathbb{N}^+$</td>
      \lt td>Projection dimension of Query and Key vectors (typically $64$ or $128$ in modern LLMs).</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\sqrt{d_k}$</th>
      \lt td align="left">Scalar $\in \mathbb{R}^+$</td>
      \lt td>The \lt strong>scaling factor</strong> (standard deviation normalizer) that preserves unit variance.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times T}$</td>
      \lt td>Scaled logit matrix. Each element has expected variance $\operatorname{Var}(S_{ij}) \approx 1$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{A} = \operatorname{softmax}(\mathbf{S})$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times T}$</td>
      \lt td>Row-stochastic attention weight matrix. Each row sums to $1.0$: $\sum_{j=1}^T A_{ij} = 1.0$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{V}$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times d_v}$</td>
      \lt td>Packed Value vectors representing the contextual content payload of each token.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$</th>
      \lt td align="left">Matrix $\in \mathbb{R}^{T \times d_v}$</td>
      \lt td>Final contextual token representations $\mathbf{O} = \mathbf{A}\mathbf{V}$, blending information across the sentence.</td>
    </tr>
  </tbody>
</table>

---

### 2. The Dependency Ladder: First-Principles Proof of Variance Explosion

Why does dividing by $\sqrt{d_k}$ work? To understand this deeply without leaps of faith, let us climb the dependency ladder from the foundational axioms of probability theory.

#### Helper Concept 1: Expected Value and Variance
For any discrete or continuous random variable $X$:
- The **Expected Value** $\mathbb{E}[X]$ measures the mean (center of mass):
  $$
  \mathbb{E}[X] = \mu
  $$
- The **Variance** $\operatorname{Var}(X)$ measures the expected squared deviation from the mean (the spread):
  $$
  \operatorname{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
  $$

#### Helper Concept 2: Independence and Products
If two random variables $X$ and $Y$ are **statistically independent**:
- $\mathbb{E}[X Y] = \mathbb{E}[X] \mathbb{E}[Y]$
- If both have zero mean ($\mathbb{E}[X] = 0, \mathbb{E}[Y] = 0$), then:
  $$
  \operatorname{Var}(X Y) = \mathbb{E}[(X Y)^2] - (\mathbb{E}[X Y])^2 = \mathbb{E}[X^2]\mathbb{E}[Y^2] - 0 = \operatorname{Var}(X) \operatorname{Var}(Y)
  $$

#### Helper Concept 3: Sum of Independent Variances
For any independent random variables $X_1, X_2, \dots, X_N$:
$$
\operatorname{Var}\left(\sum_{i=1}^N X_i\right) = \sum_{i=1}^N \operatorname{Var}(X_i)
$$

---

#### The Rigorous Step-by-Step Derivation

Assume the components of the Query vector $\mathbf{q} = [q_1, \dots, q_{d_k}]^\top$ and Key vector $\mathbf{k} = [k_1, \dots, k_{d_k}]^\top$ are independent random variables with zero mean and unit variance:

$$
\mathbb{E}[q_i] = 0, \quad \operatorname{Var}(q_i) = 1, \quad \mathbb{E}[k_i] = 0, \quad \operatorname{Var}(k_i) = 1
$$

Now, consider the unscaled dot product $Z = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$:

1. **Calculate the Expected Value of the Dot Product**:
   $$
   \mathbb{E}[Z] = \mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = \sum_{i=1}^{d_k} (0)(0) = 0
   $$
   The mean remains perfectly centered at zero.

2. **Calculate the Variance of a Single Product Term**:
   $$
   \operatorname{Var}(q_i k_i) = \operatorname{Var}(q_i)\operatorname{Var}(k_i) = (1)(1) = 1
   $$

3. **Calculate the Variance of the Sum of $d_k$ Terms**:
   $$
   \operatorname{Var}(Z) = \operatorname{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k
   $$

4. **Calculate the Standard Deviation**:
   $$
   \sigma_Z = \sqrt{\operatorname{Var}(Z)} = \sqrt{d_k}
   $$

The spread (standard deviation) of the raw dot product grows directly as $\sqrt{d_k}$!

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 8.2:</strong> How Dot Product Variance and Score Magnitude Scale with Vector Dimension</caption>
  \lt thead>
    \lt tr bgcolor="#eae9e1">
      \lt th scope="col" align="center" width="15%">Vector Dimension $d_k$</th>
      \lt th scope="col" align="center" width="20%">Variance $\operatorname{Var}(Z) = d_k$</th>
      \lt th scope="col" align="center" width="25%">Standard Deviation $\sigma = \sqrt{d_k}$</th>
      \lt th scope="col" align="left" width="40%">Typical Score Range ($\pm 3\sigma$) &amp; Effect</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt td align="center">$d_k = 4$</td>
      \lt td align="center">$4$</td>
      \lt td align="center">$2.0$</td>
      \lt td>$[-6.0, +6.0]$ &mdash; Moderate spread; Softmax remains somewhat responsive.</td>
    </tr>
    \lt tr>
      \lt td align="center">$d_k = 16$</td>
      \lt td align="center">$16$</td>
      \lt td align="center">$4.0$</td>
      \lt td>$[-12.0, +12.0]$ &mdash; Significant widening; large exponents start appearing.</td>
    </tr>
    \lt tr>
      \lt td align="center">$d_k = 64$</td>
      \lt td align="center">$64$</td>
      \lt td align="center">$8.0$</td>
      \lt td>$[-24.0, +24.0]$ &mdash; \lt mark>Extreme saturation!</mark> Exponents reach $e^{24} \approx 2.6 \times 10^{10}$.</td>
    </tr>
    \lt tr>
      \lt td align="center">$d_k = 128$</td>
      \lt td align="center">$128$</td>
      \lt td align="center">$11.31$</td>
      \lt td>$[-33.9, +33.9]$ &mdash; Catastrophic saturation! One-hot hardmax collapse.</td>
    </tr>
  </tbody>
</table>

---

### 3. The Softmax Saturation Zone & Vanishing Gradients

Why is a score range of $[-24, +24]$ fatal to an LLM?

Recall the exact Softmax derivative derived in Chapter 07:

$$
\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j) = \begin{cases} s_i(1 - s_i) & \text{if } i = j \\ -s_i s_j & \text{if } i \neq j \end{cases}
$$

Let us evaluate this derivative when inputs have standard deviation $8$ (e.g., $z_1 = +24, z_2 = 0, z_3 = -8$):

- For the winning token $i=1$: $s_1 \approx 0.9999999999$.
  $$
  \frac{\partial s_1}{\partial z_1} = s_1(1 - s_1) \approx 1.0 \times (1.0 - 1.0) = \mathbf{0.0}
  $$
- For the losing tokens $i \neq 1$: $s_j \approx 0.0000000001$.
  $$
  \frac{\partial s_1}{\partial z_j} = -s_1 s_j \approx -(1.0) \times (0.0) = \mathbf{0.0}
  $$

Every single partial derivative vanishes into machine zero!

When the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{S}}$ is backpropagated to update the projection weights $\mathbf{W}_Q$ and $\mathbf{W}_K$, the chain rule multiplies by $\frac{\partial s}{\partial z} \approx 0$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_Q} = \sum \frac{\partial \mathcal{L}}{\partial s} \cdot \mathbf{0} = \mathbf{0}
$$

The learning signal is entirely extinguished. The network falls into a computational coma.

---

### 4. The Mathematical Cure: Dividing by $\sqrt{d_k}$

By the scalar property of variance, for any constant $c$:

$$
\operatorname{Var}(c X) = c^2 \operatorname{Var}(X)
$$

Let $c = \frac{1}{\sqrt{d_k}}$. We scale the raw dot product $Z$:

$$
\operatorname{Var}\left(\frac{Z}{\sqrt{d_k}}\right) = \left(\frac{1}{\sqrt{d_k}}\right)^2 \operatorname{Var}(Z) = \frac{1}{d_k} \times d_k = \mathbf{1}
$$

The variance is restored to strictly **$1$**, and the standard deviation is restored to **$1.0$**!

No matter whether $d_k = 4$, $d_k = 64$, or $d_k = 1024$, the scaled logits $\frac{\mathbf{q}\cdot\mathbf{k}}{\sqrt{d_k}}$ comfortably reside within $[-3.0, +3.0]$.

In this range, Softmax outputs balanced, smooth probabilities, and its derivative $\frac{\partial s_i}{\partial z_j}$ maintains active, robust values (e.g. $0.15 \sim 0.25$), allowing gradient descent to train billion-parameter models rapidly and stably!

---

## Step 4: Where Did It Come From? {: #step-4 }

\lt dl>
  \lt dt>\lt time datetime="2014">2014</time> &mdash; \lt strong>Dzmitry Bahdanau, Kyunghyun Cho, &amp; Yoshua Bengio</strong></dt>
  \lt dd>
    In their seminal paper \lt cite>"Neural Machine Translation by Jointly Learning to Align and Translate"</cite>, Bahdanau et al. introduced the first soft attention mechanism to solve the fixed-length memory bottleneck in RNNs. However, their mechanism used \lt strong>additive attention</strong>:
    
$$
\operatorname{score}(\mathbf{s}, \mathbf{h}) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{s} + \mathbf{U}_a \mathbf{h})
$$

    While highly expressive, additive attention requires projecting two vectors through weight matrices and computing expensive non-linear $\tanh$ activations for every token pair, making it slow and memory-intensive to parallelize.
  </dd>

  \lt dt>\lt time datetime="2015">2015</time> &mdash; \lt strong>Minh-Thang Luong, Hieu Pham, &amp; Christopher D. Manning</strong></dt>
  \lt dd>
    In \lt cite>"Effective Approaches to Attention-based Neural Machine Translation"</cite>, Luong et al. introduced \lt strong>multiplicative (dot-product) attention</strong>:
    
$$
\operatorname{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^\top \mathbf{W} \mathbf{h}
$$

    Dot-product attention was dramatically faster on GPUs because it mapped directly to optimized matrix multiplication kernels (BLAS / cuBLAS). However, researchers noticed an unsettling bug: as hidden dimensions grew, dot-product attention struggled to converge compared to additive attention.
  </dd>

  \lt dt>\lt time datetime="2017">2017</time> &mdash; \lt strong>Ashish Vaswani et al. (Google Brain &amp; Google Research)</strong></dt>
  \lt dd>
    In the landmark paper \lt cite>"Attention Is All You Need"</cite>, Vaswani et al. identified the theoretical reason why dot-product attention failed at large dimensions: variance explosion pushing Softmax into regions with vanishing gradients.
    
    They proved that scaling by $\frac{1}{\sqrt{d_k}}$ preserves the blazing computational speed of matrix multiplication while matching the convergence stability of additive attention:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

    This single equation eliminated recurrence entirely and became the computational foundation for GPT, BERT, LLaMA, and Gemini.
  </dd>
</dl>

---

## Step 5: Concrete Toy Example (Step-by-Step Hand Arithmetic) {: #step-5 }

Let us follow the journey of a 3-word sequence:

- Token 1 ($t=1$): \lt kbd>"The"</kbd>
- Token 2 ($t=2$): \lt kbd>"river"</kbd>
- Token 3 ($t=3$): \lt kbd>"bank"</kbd> (Ambiguous noun)

We will compute the updated contextual representation for \lt kbd>"bank"</kbd> ($t=3$) by hand, using $d_k = 4$ so that $\sqrt{d_k} = \sqrt{4} = 2.0$ exactly!

### Step 5.1: The Setup Vectors

Let the Query vector for token 3 (\lt kbd>"bank"</kbd>) be:

$$
\mathbf{q}_3 = \begin{bmatrix} 1.0 \\ 2.0 \\ 1.0 \\ 0.0 \end{bmatrix} \in \mathbb{R}^4
$$

Let the Key vectors for the three tokens be:

$$
\mathbf{k}_1 (\text{"The"}) = \begin{bmatrix} 0.0 \\ 1.0 \\ 0.0 \\ 1.0 \end{bmatrix}, \quad
\mathbf{k}_2 (\text{"river"}) = \begin{bmatrix} 2.0 \\ 2.0 \\ 1.0 \\ 1.0 \end{bmatrix}, \quad
\mathbf{k}_3 (\text{"bank"}) = \begin{bmatrix} 1.0 \\ 1.0 \\ 1.0 \\ 0.0 \end{bmatrix}
$$

Let each word carry a 2-dimensional Value vector $\mathbf{v}_j \in \mathbb{R}^2$ ($d_v = 2$), where dimension 1 encodes grammatical function and dimension 2 encodes water/nature features:

$$
\mathbf{v}_1 (\text{"The"}) = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad
\mathbf{v}_2 (\text{"river"}) = \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix}, \quad
\mathbf{v}_3 (\text{"bank"}) = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}
$$

---

### Step 5.2: Raw Dot Products ($\mathbf{q}_3 \cdot \mathbf{k}_j$)

We calculate the unscaled dot products:

1. **Match with Token 1 (\lt kbd>"The"</kbd>)**:
   $$
   z_1 = \mathbf{q}_3^\top \mathbf{k}_1 = (1.0)(0.0) + (2.0)(1.0) + (1.0)(0.0) + (0.0)(1.0) = 0.0 + 2.0 + 0.0 + 0.0 = 2.0
   $$

2. **Match with Token 2 (\lt kbd>"river"</kbd>)**:
   $$
   z_2 = \mathbf{q}_3^\top \mathbf{k}_2 = (1.0)(2.0) + (2.0)(2.0) + (1.0)(1.0) + (0.0)(1.0) = 2.0 + 4.0 + 1.0 + 0.0 = \mathbf{7.0}
   $$

3. **Match with Token 3 (\lt kbd>"bank"</kbd>, Self-Attention)**:
   $$
   z_3 = \mathbf{q}_3^\top \mathbf{k}_3 = (1.0)(1.0) + (2.0)(1.0) + (1.0)(1.0) + (0.0)(0.0) = 1.0 + 2.0 + 1.0 + 0.0 = 4.0
   $$

The raw, unscaled logit vector is:

$$
\mathbf{z}_{\text{raw}} = [2.0, 7.0, 4.0]^\top
$$

---

### Step 5.3: Scaling by $\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{4}} = \frac{1}{2} = 0.5$

Now we apply the scaling damper:

$$
\tilde{z}_j = \frac{z_j}{\sqrt{d_k}} = \frac{z_j}{2.0}
$$

- $\tilde{z}_1 = \frac{2.0}{2.0} = 1.0$
- $\tilde{z}_2 = \frac{7.0}{2.0} = 3.5$
- $\tilde{z}_3 = \frac{4.0}{2.0} = 2.0$

The scaled logit vector is:

$$
\tilde{\mathbf{z}} = [1.0, 3.5, 2.0]^\top
$$

Notice what happened: the spread between the highest score ($7.0$) and lowest score ($2.0$) shrank from $5.0$ down to $2.5$!

---

### Step 5.4: Softmax Normalization (With vs. Without Scaling)

Let us compute the Softmax weights on our scaled scores $\tilde{\mathbf{z}} = [1.0, 3.5, 2.0]$:

1. **Exponentiate each scaled score**:
   - $e^{\tilde{z}_1} = e^{1.0} \approx 2.7183$
   - $e^{\tilde{z}_2} = e^{3.5} \approx 33.1155$
   - $e^{\tilde{z}_3} = e^{2.0} \approx 7.3891$

2. **Sum the exponentiated terms (Partition Function $Z$)**:
   $$
   \sum_{j=1}^3 e^{\tilde{z}_j} = 2.7183 + 33.1155 + 7.3891 = 43.2229
   $$

3. **Divide each term by the sum**:
   - $A_{31} = \frac{2.7183}{43.2229} \approx \mathbf{0.0629}$ ($6.29\%$)
   - $A_{32} = \frac{33.1155}{43.2229} \approx \mathbf{0.7662}$ ($76.62\%$)
   - $A_{33} = \frac{7.3891}{43.2229} \approx \mathbf{0.1709}$ ($17.09\%$)

Check total probability: $0.0629 + 0.7662 + 0.1709 = 1.0000$ ($100.0\%$).

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 8.3:</strong> Impact of $\sqrt{d_k}$ Scaling on Attention Distribution &amp; Gradient Responsiveness</caption>
  \lt thead>
    \lt tr bgcolor="#eae9e1">
      \lt th scope="col" align="left" width="22%">Target Word</th>
      \lt th scope="col" align="center" width="26%">Unscaled Attention ($[2.0, 7.0, 4.0]$)</th>
      \lt th scope="col" align="center" width="26%">Scaled Attention ($[1.0, 3.5, 2.0]$)</th>
      \lt th scope="col" align="left" width="26%">Visual Distribution (Scaled)</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt th scope="row" align="left">\lt kbd>"The"</kbd></th>
      \lt td align="center">0.64% ($s_1 = 0.0064$)</td>
      \lt td align="center">\lt strong>6.29%</strong> ($A_{31} = 0.0629$)</td>
      \lt td>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0629">6.29%</meter></td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">\lt kbd>"river"</kbd></th>
      \lt td align="center">94.65% ($s_2 = 0.9465$)</td>
      \lt td align="center">\lt strong>76.62%</strong> ($A_{32} = 0.7662$)</td>
      \lt td>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.7662">76.62%</meter></td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">\lt kbd>"bank"</kbd> (Self)</th>
      \lt td align="center">4.71% ($s_3 = 0.0471$)</td>
      \lt td align="center">\lt strong>17.09%</strong> ($A_{33} = 0.1709$)</td>
      \lt td>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1709">17.09%</meter></td>
    </tr>
    \lt tr bgcolor="#f0f7f0">
      \lt th scope="row" align="left">\lt strong>Gradient $\frac{\partial s_2}{\partial z_2}$</strong></th>
      \lt td align="center">\lt strong>$0.0506$</strong> (Vanishing!)</td>
      \lt td align="center">\lt strong>$0.1791$</strong> (\lt mark>3.5× stronger!</mark>)</td>
      \lt td>Healthy learning signal</td>
    </tr>
  </tbody>
</table>

Without scaling, the top candidate surged to $94.65\%$, and its local gradient shrank by $72\%$. With scaling, \lt kbd>"river"</kbd> still clearly wins at $76.62\%$, but the attention weights remain malleable, and gradients flow vigorously back through the computational graph!

---

### Step 5.5: Blending Value Vectors into the Final Context Vector

The final step of the attention formula multiplies attention weights $\mathbf{A}_{3,:}$ by the Value matrix $\mathbf{V}$:

$$
\mathbf{c}_3 = \sum_{j=1}^3 A_{3j} \mathbf{v}_j = A_{31} \mathbf{v}_1 + A_{32} \mathbf{v}_2 + A_{33} \mathbf{v}_3
$$

Substituting our exact numbers:

$$
\begin{aligned}
\mathbf{c}_3 &= 0.0629 \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + 0.7662 \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} + 0.1709 \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix} \\
&= \begin{bmatrix} 0.0629 \times 1.0 + 0.7662 \times 0.0 + 0.1709 \times 2.0 \\ 0.0629 \times 0.0 + 0.7662 \times 3.0 + 0.1709 \times 1.0 \end{bmatrix} \\
&= \begin{bmatrix} 0.0629 + 0.0 + 0.3418 \\ 0.0 + 2.2986 + 0.1709 \end{bmatrix} \\
&= \mathbf{\begin{bmatrix} 0.4047 \\ 2.4695 \end{bmatrix}}
\end{aligned}
$$

Look at the physical transformation of the word <kbd>"bank"</kbd>:
- Before attention, its Value vector was $\mathbf{v}_3 = [2.0, 1.0]^\top$. Its water/nature dimension (dimension 2) had a weak value of only $1.0$.
- After scaled attention, **$76.62\%$** of its context was drawn from <kbd>"river"</kbd> ($\mathbf{v}_2 = [0.0, 3.0]^\top$).
- Its water/nature coordinate surged from $1.0$ to **$2.4695$**!
- The ambiguous noun <kbd>"bank"</kbd> has successfully absorbed the meaning of its surroundings. The model now knows beyond any doubt that this word refers to a river shoreline, not a financial institution!

---

## Step 6: Core Takeaway {: #step-6 }

<fieldset>
<legend><strong>Core Memory Card</strong></legend>
<p>
<strong>The Scaled Dot-Product Attention formula is the dynamic router of the Transformer:</strong><br>
$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ computes how relevant every token is to every other token. The crucial divisor $\sqrt{d_k}$ acts as a mathematical thermostat: because summing over $d_k$ dimensions scales variance by $d_k$, dividing by $\sqrt{d_k}$ standardizes variance back to $1.0$. This prevents Softmax from saturating into extreme one-hot outputs, saving backpropagation from vanishing gradients and allowing transformers to scale to hundreds of layers and billions of parameters.
</p>
</fieldset>
