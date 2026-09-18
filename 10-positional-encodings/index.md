# Chapter 10: Where in the Sentence Am I? (Positional Encodings & RoPE)

---

## Step 1: 3-Year-Old Intuition (The Runner Bibs & The Spinning Dials)

> [!INTUITION] The Scrambled Photo & The Spinning Clock Hands
> Imagine you are a sports photographer taking pictures at a school running race.
>
> Ten energetic children line up on the grass track and race toward the finish line. After the race, you print out a snapshot of the children running side by side.
>
> But someone forgot to pin **numbered bibs** onto the children's shirts!
>
> When you look at the photograph, you see ten smiling children running with all their might. But because nobody is wearing a number tag, and there are no lane numbers painted on the ground, a huge problem appears:
>
> If you cut out the photograph of each child with scissors and shuffle them on the table like a deck of playing cards, the picture still looks completely plausible! You cannot tell who took the first step, who was leading in second place, or who started at the very back.
>
> In human language, word order is everything:
>
> - *"Dog bites man"* is an ordinary playground accident.
> - *"Man bites dog"* is shocking breaking news!
>
> Yet to a computer calculating word matches, both sentences contain the exact same words. Without position tags, the computer scrambles the words in a blender. It cannot tell who bit whom!
>
> To fix this, the referee gives every runner a special badge with **Spinning Clock Dials**:
>
> 1. **Dials at Different Speeds**: Each badge has several clock hands. The first hand spins very fast (ticking with every single step you take). The second hand spins moderately. The third hand spins slowly like a giant hour dial.
> 2. **A Unique Angle for Every Step**: At step 1, the hands point in one direction. At step 2, they twist slightly. At step 5, they rotate further. No two steps on the track ever produce the exact same dial angles.
> 3. **Relative Angle Difference**: Best of all, if Runner 1 is at minute 5 and Runner 2 is at minute 2, you don't even need to know what time of day it is! You only need to measure the angle between their two dials: exactly 3 minutes apart!
>
> In modern Large Language Models, this spinning dial system is called **Rotary Position Embedding (RoPE)**. Instead of adding messy numbers into word meanings, RoPE simply rotates the word's directional pointer by an angle that matches its exact place in the sentence!

<figure>
<pre>
Word Order Scramble (Vanilla Attention has Zero Sense of Direction):

Sentence A: [The] [dog]  [bites] [the]  [man]  ──► Attention bag: {The, dog, bites, man}
Sentence B: [The] [man]  [bites] [the]  [dog]  ──► Attention bag: {The, man, bites, dog}
Result: Raw dot products produce the exact same sum! The model is blind to order.

The Rotary Solution (RoPE Rotates Each Word by its Step Index m):

Step m=1: [The]   ──► Dial rotated by 1 x θ: ──► [Pointer at 28°]
Step m=2: [dog]   ──► Dial rotated by 2 x θ: ──► [Pointer at 57°]
Step m=3: [bites] ──► Dial rotated by 3 x θ: ──► [Pointer at 86°]
Step m=4: [the]   ──► Dial rotated by 4 x θ: ──► [Pointer at 115°]
Step m=5: [man]   ──► Dial rotated by 5 x θ: ──► [Pointer at 143°]

Relative Distance:
  Angle between "dog" (m=2) and "man" (m=5) = 5θ - 2θ = 3θ
  (Always 3 steps apart, regardless of absolute positions!)
</pre>
<figcaption><strong>Figure 10.1:</strong> RoPE turns positional distance into an angle difference on a spinning dial, preserving word order without corrupting semantic embeddings.</figcaption>
</figure>

---

## Step 2: The Bridging Question

> [!BRIDGING] Why Vanilla Attention is Permutation-Equivariant
> In Chapter 06 and Chapter 08, we studied how the attention mechanism computes pairwise token compatibility using dot products:
>
> $$
> S_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
> $$
>
> Now, suppose we permute the input sequence of token vectors $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_T]^\top$ using a permutation matrix $\mathbf{P} \in \{0, 1\}^{T \times T}$. Because projection matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ operate row-by-row independently:
>
> $$
> \mathbf{Q}_{\text{perm}} = \mathbf{P}\mathbf{Q}, \quad \mathbf{K}_{\text{perm}} = \mathbf{P}\mathbf{K}, \quad \mathbf{V}_{\text{perm}} = \mathbf{P}\mathbf{V}
> $$
>
> When we calculate the attention score matrix:
>
> $$
> \mathbf{Q}_{\text{perm}} \mathbf{K}_{\text{perm}}^\top = (\mathbf{P}\mathbf{Q})(\mathbf{P}\mathbf{K})^\top = \mathbf{P}\mathbf{Q}\mathbf{K}^\top \mathbf{P}^\top
> $$
>
> Passing this through Softmax and multiplying by $\mathbf{V}_{\text{perm}}$ yields:
>
> $$
> \operatorname{Attention}(\mathbf{Q}_{\text{perm}}, \mathbf{K}_{\text{perm}}, \mathbf{V}_{\text{perm}}) = \mathbf{P} \cdot \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
> $$
>
> This mathematical property is called <dfn id="def-permutation-equivariance">Permutation Equivariance</dfn>.
>
> It means that if you shuffle the input words, the output vectors are simply shuffled in the exact same way—**no interaction changes at all**! The model has zero built-in notion of which word came first, which came second, or how far apart two words are.
>
> Early Transformers (Vaswani et al., 2017) attempted to solve this by adding static position vectors directly to token embeddings:
>
> $$
> \mathbf{x}_m \leftarrow \mathbf{x}_m + \mathbf{p}_m
> $$
>
> But adding $\mathbf{p}_m$ directly into $\mathbf{x}_m$ has two severe flaws:
> 1. **Semantic Contamination**: It adds raw numbers directly into the semantic embedding space, distorting the lexical meaning of the word.
> 2. **Absolute, Not Relative**: The network learns that position $500$ has a certain signature, but it struggles to generalize the relative rule: *"Word A is 3 steps ahead of Word B, regardless of whether they appear at position 5 or position 5000."*
>
> *"How do we mathematically encode position so that the dot product between any two tokens depends strictly and purely on their relative distance $(m - n)$, without adding noisy vectors to their semantic content?"*

---

## Step 3: The Exact Math & Formula

### 1. The Classical Approach: Sinusoidal Absolute Positional Encodings

In the original Transformer (*"Attention Is All You Need"*, Vaswani et al., 2017), the authors used fixed harmonic sinusoidal functions of varying frequencies.

For a token at position $m \in \{0, 1, \dots, T-1\}$ and feature dimension index $i \in \{0, 1, \dots, \frac{d}{2}-1\}$:

$$
\begin{aligned}
\text{PE}_{(m, 2i)} &= \sin\left(\frac{m}{10000^{2i/d}}\right) \\
\text{PE}_{(m, 2i+1)} &= \cos\left(\frac{m}{10000^{2i/d}}\right)
\end{aligned}
$$

Let $\theta_i = \frac{1}{10000^{2i/d}} = 10000^{-2i/d}$. Then the positional vector for position $m$ is:

$$
\mathbf{p}_m = \begin{bmatrix}
\sin(m\theta_0) \\
\cos(m\theta_0) \\
\sin(m\theta_1) \\
\cos(m\theta_1) \\
\vdots \\
\sin(m\theta_{d/2 - 1}) \\
\cos(m\theta_{d/2 - 1})
\end{bmatrix} \in \mathbb{R}^d
$$

The wavelengths form a geometric progression from $2\pi$ to $2\pi \cdot 10000$:
- Low dimensions ($i = 0$): $\theta_0 = 1$, wavelength is $2\pi \approx 6.28$ tokens (fast-ticking clock hand).
- High dimensions ($i = \frac{d}{2}-1$): wavelength reaches tens of thousands of tokens (slow-moving calendar dial).

While elegant, this classical scheme adds $\mathbf{p}_m$ additively: $\widetilde{\mathbf{x}}_m = \mathbf{x}_m + \mathbf{p}_m$. In the dot product $(\mathbf{x}_m + \mathbf{p}_m)^\top (\mathbf{x}_n + \mathbf{p}_n)$, cross-terms $\mathbf{x}_m^\top \mathbf{p}_n$ and $\mathbf{p}_m^\top \mathbf{x}_n$ create spurious interactions between content and position.

---

### 2. The Modern Standard: Rotary Position Embedding (RoPE)

Formulated by Jianlin Su (2021), <dfn id="def-rope">Rotary Position Embedding (RoPE)</dfn> solves the relative position problem geometrically by **rotating** Query and Key vectors in 2D coordinate planes instead of adding vectors.

#### The 2D Rotation Block

Consider a 2-dimensional vector $\mathbf{v} = [v_1, v_2]^\top$. Rotating this vector by an angle $m\theta$ is represented by the 2D rotation matrix:

$$
\mathbf{R}_{\theta, m} = \begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

#### The Full $d$-Dimensional Rotation Operator

For a $d$-dimensional Query vector $\mathbf{q}_m \in \mathbb{R}^d$ and Key vector $\mathbf{k}_n \in \mathbb{R}^d$ (where $d$ is an even number, such as $d = 64$ or $d = 128$), RoPE divides the $d$ coordinates into $\frac{d}{2}$ consecutive pairs:

$$
(q^{(1)}, q^{(2)}), \quad (q^{(3)}, q^{(4)}), \quad \dots, \quad (q^{(d-1)}, q^{(d)})
$$

Each 2D pair is rotated by its own dedicated frequency angle $m\theta_i$:

$$
\mathbf{R}_{\Theta, m}^d = \begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \dots & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \dots & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \dots & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \dots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{bmatrix} \in \mathbb{R}^{d \times d}
$$

where the frequencies are defined identically to the sinusoidal base:

$$
\theta_i = 10000^{-2(i-1)/d}, \quad i \in \left\{1, 2, \dots, \frac{d}{2}\right\}
$$

The rotated Query and Key vectors at positions $m$ and $n$ are:

$$
\widetilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m}^d \mathbf{q}_m, \quad \widetilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n}^d \mathbf{k}_n
$$

Notice that the Value vectors $\mathbf{V}$ are **not rotated**. Value vectors carry semantic payload, not geometric search coordinates!

<details>
<summary><strong>Mathematical Symbol Catalog & Tensor Definitions</strong></summary>
<dl>
  <dt><strong>$m, n \in \mathbb{N}$</strong></dt>
  <dd>Absolute token sequence positions: $m$ is the Query token position, $n$ is the Key token position ($0 \le m, n \lt T$).</dd>
  <dt><strong>$d$ (Head Dimension)</strong></dt>
  <dd>The dimensionality of each attention head (must be an even integer, typically $d = 64$ or $d = 128$).</dd>
  <dt><strong>$\theta_i$ (Base Frequency for Pair $i$)</strong></dt>
  <dd>The rotational angular velocity for the $i$-th coordinate subspace: $\theta_i = b^{-2(i-1)/d}$, where base $b = 10000$ (or up to $500000$ in modern long-context models).</dd>
  <dt><strong>$\mathbf{R}_{\theta_i, m} \in \mathbb{R}^{2 \times 2}$</strong></dt>
  <dd>Standard orthogonal 2D rotation matrix: $\mathbf{R}_{\theta_i, m}^\top \mathbf{R}_{\theta_i, m} = \mathbf{I}_2$, $\det(\mathbf{R}) = 1$.</dd>
  <dt><strong>$\mathbf{R}_{\Theta, m}^d \in \mathbb{R}^{d \times d}$</strong></dt>
  <dd>Block-diagonal orthogonal rotation matrix composed of $\frac{d}{2}$ independent 2D rotation blocks.</dd>
  <dt><strong>$\widetilde{\mathbf{q}}_m, \widetilde{\mathbf{k}}_n \in \mathbb{R}^d$</strong></dt>
  <dd>Position-aware Query and Key vectors after applying RoPE rotation.</dd>
</dl>
</details>

---

### 3. The Grand Proof: Why RoPE Guarantees Relative Invariance

Why is RoPE so celebrated across modern AI architectures? Because it guarantees that the dot product between Query $m$ and Key $n$ depends **strictly and purely on their relative distance $(m - n)$**.

Let us prove this step by step.

#### Step A: Orthogonality and Transpose of 2D Rotations
The 2D rotation matrix for an angle $\alpha$ has the property that its transpose is its inverse, which equals a rotation by $-\alpha$:

$$
\mathbf{R}_\alpha^\top = \begin{bmatrix}
\cos(\alpha) & \sin(\alpha) \\
-\sin(\alpha) & \cos(\alpha)
\end{bmatrix} = \begin{bmatrix}
\cos(-\alpha) & -\sin(-\alpha) \\
\sin(-\alpha) & \cos(-\alpha)
\end{bmatrix} = \mathbf{R}_{-\alpha}
$$

#### Step B: Product of Rotation Matrices
Multiplying two 2D rotation matrices corresponds to adding their angles:

$$
\mathbf{R}_\alpha \mathbf{R}_\beta = \mathbf{R}_{\alpha + \beta}
$$

Therefore:

$$
\mathbf{R}_{\theta, m}^\top \mathbf{R}_{\theta, n} = \mathbf{R}_{\theta, -m} \mathbf{R}_{\theta, n} = \mathbf{R}_{\theta, (n - m)}
$$

#### Step C: The Attention Score Dot Product
Now evaluate the dot product between the rotated Query $\widetilde{\mathbf{q}}_m$ and rotated Key $\widetilde{\mathbf{k}}_n$:

$$
\begin{aligned}
\langle \widetilde{\mathbf{q}}_m, \widetilde{\mathbf{k}}_n \rangle &= \widetilde{\mathbf{q}}_m^\top \widetilde{\mathbf{k}}_n \\
&= (\mathbf{R}_{\Theta, m}^d \mathbf{q}_m)^\top (\mathbf{R}_{\Theta, n}^d \mathbf{k}_n) \\
&= \mathbf{q}_m^\top \left( (\mathbf{R}_{\Theta, m}^d)^\top \mathbf{R}_{\Theta, n}^d \right) \mathbf{k}_n
\end{aligned}
$$

Because $\mathbf{R}_{\Theta, m}^d$ is block-diagonal, each 2D block multiplies independently:

$$
(\mathbf{R}_{\Theta, m}^d)^\top \mathbf{R}_{\Theta, n}^d = \mathbf{R}_{\Theta, (n - m)}^d
$$

Substituting this back gives the foundational identity:

$$
\langle \widetilde{\mathbf{q}}_m, \widetilde{\mathbf{k}}_n \rangle = \mathbf{q}_m^\top \mathbf{R}_{\Theta, (n - m)}^d \mathbf{k}_n
$$

Look at the right-hand side:
- The absolute positions $m$ and $n$ have completely disappeared!
- Only the **relative distance** $(n - m)$ remains!
- If token A is at position $100$ and token B is at position $103$, their distance is $103 - 100 = 3$. If the entire sentence is shifted to positions $5000$ and $5003$, the distance is still $5003 - 5000 = 3$. Their attention score is **strictly identical**!

---

### 4. Complex Number Formulation (Euler's Formula)

In complex analysis, rotating a 2D vector $(x, y)$ by angle $\phi$ is identical to multiplying the complex number $z = x + i y$ by $e^{i\phi}$:

$$
z \cdot e^{i\phi} = (x + iy)(\cos\phi + i\sin\phi) = (x\cos\phi - y\sin\phi) + i(x\sin\phi + y\cos\phi)
$$

Under this complex lens, for the $k$-th coordinate pair $q_{(k)} \in \mathbb{C}$ and $k_{(k)} \in \mathbb{C}$:

$$
\widetilde{q}_{(k), m} = q_{(k)} e^{i m \theta_k}, \quad \widetilde{k}_{(k), n} = k_{(k)} e^{i n \theta_k}
$$

The real dot product of two 2D vectors equals the real part of the complex inner product $\operatorname{Re}(u v^*)$, where $v^*$ is the complex conjugate:

$$
\begin{aligned}
\operatorname{Re}\left( \widetilde{q}_{(k), m} \cdot \widetilde{k}_{(k), n}^* \right) &= \operatorname{Re}\left( (q_{(k)} e^{i m \theta_k}) \cdot (k_{(k)} e^{i n \theta_k})^* \right) \\
&= \operatorname{Re}\left( q_{(k)} k_{(k)}^* \cdot e^{i m \theta_k} e^{-i n \theta_k} \right) \\
&= \operatorname{Re}\left( q_{(k)} k_{(k)}^* \cdot e^{i (m - n) \theta_k} \right)
\end{aligned}
$$

This provides an immediate, 1-line proof of the relative position property!

---

### 5. Efficient GPU Implementation (Zero $d \times d$ Matrix Allocation)

In actual PyTorch or CUDA code, we **never** instantiate the $d \times d$ sparse matrix $\mathbf{R}_{\Theta, m}^d$. Doing so would waste massive amounts of memory and compute.

Instead, notice that:

$$
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
= \begin{bmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_1 \sin(m\theta) + x_2 \cos(m\theta) \end{bmatrix}
= \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \cos(m\theta) + \begin{bmatrix} -x_2 \\ x_1 \end{bmatrix} \sin(m\theta)
$$

For the entire vector $\mathbf{x} = [x_1, x_2, x_3, x_4, \dots, x_{d-1}, x_d]^\top$, define the half-rotated vector $\mathbf{x}_{\text{rot}}$:

$$
\mathbf{x}_{\text{rot}} = [-x_2, x_1, -x_4, x_3, \dots, -x_d, x_{d-1}]^\top
$$

Then RoPE rotation is computed using simple element-wise operations:

$$
\mathbf{R}_{\Theta, m}^d \mathbf{x} = \mathbf{x} \odot \cos(\mathbf{m}\Theta) + \mathbf{x}_{\text{rot}} \odot \sin(\mathbf{m}\Theta)
$$

This runs at memory bandwidth speed with **zero matrix multiplication** overhead!

---

## Step 4: Where Did It Come From?

<figure>
<pre>
The Evolution of Positional Representations in NLP:

1950s-2010s: Bag-of-Words / Naive Attention ──► P = None. Permutation invariant.
                                                 "Dog bites man" == "Man bites dog"
      │
      ▼
2017: Vaswani et al. (Attention Is All You Need) ──► Sinusoidal Absolute PE: x + PE(pos)
                                                      Fixed harmonic frequencies;
                                                      Additive noise in semantic space.
      │
      ▼
2018-2019: Learned Absolute PE (BERT, GPT-2) ──► Lookup table: W_pos in R^{T_max x d}
                                                 Cannot extrapolate beyond training length T_max!
      │
      ▼
2018-2020: Relative Attention Bias (Shaw, T5) ──► Attention score bias: S_{ij} + b_{i-j}
                                                  Slow O(T^2) memory footprint; breaks FlashAttention.
      │
      ▼
2021: Jianlin Su (RoFormer) ────────────────────► Rotary Position Embedding (RoPE)
                                                  Multiplicative 2D rotations on Q and K;
                                                  Preserves relative distance; O(1) allocation;
                                                  Adopted by LLaMA, Mistral, Qwen, DeepSeek.
</pre>
<figcaption><strong>Figure 10.2:</strong> Historical evolution from static bag-of-words to multiplicative rotary geometry.</figcaption>
</figure>

### 1. The Historical Quest for Relative Position

To understand why RoPE became universal, examine what broke in prior architectures:

<fieldset>
<legend><strong>Evolution of Positional Encodings: Strengths & Fatal Flaws</strong></legend>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 10.1:</strong> Comparison of positional encoding mechanisms in language models.</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">Mechanism</th>
      <th align="center">Mathematical Form</th>
      <th align="left">Where Used</th>
      <th align="left">Fatal Flaw / Limitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Sinusoidal Absolute PE</strong></td>
      <td align="center">$\mathbf{x}_m + \mathbf{p}_m$</td>
      <td>Transformer (2017)</td>
      <td>
        <del>Additive corruption!</del> Injects position directly into semantic embedding space. Cross-terms $\mathbf{x}^\top \mathbf{p}$ introduce unintended artifacts.
      </td>
    </tr>
    <tr>
      <td><strong>Learned Absolute PE</strong></td>
      <td align="center">$\mathbf{x}_m + \mathbf{W}_{\text{pos}}[m]$</td>
      <td>BERT (2018), GPT-2 (2019)</td>
      <td>
        <del>Hard length ceiling!</del> If table size is $T_{\max} = 2048$, position $2049$ has no embedding. Model cannot process longer sequences without retraining.
      </td>
    </tr>
    <tr>
      <td><strong>Relative Bias (T5 / Shaw)</strong></td>
      <td align="center">$S_{ij} + b_{i-j}$</td>
      <td>Shaw (2018), T5 (2020)</td>
      <td>
        <del>Memory & speed bottleneck!</del> Requires allocating and indexing an explicit $T \times T$ bias matrix in memory. Completely incompatible with fused kernels like FlashAttention.
      </td>
    </tr>
    <tr bgcolor="#fdfdf0">
      <td><strong>Rotary Position (RoPE)</strong></td>
      <td align="center">$\mathbf{R}_m \mathbf{q}_m, \, \mathbf{R}_n \mathbf{k}_n$</td>
      <td>LLaMA, Mistral, Qwen, DeepSeek</td>
      <td>
        <ins><strong>The Ideal Standard!</strong></ins> Relative distance invariance $\mathbf{R}_{n-m}$, zero memory allocation, zero interference with Values, 100% compatible with FlashAttention!
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

### 2. Su's Mathematical Formulation (2021)

In 2021, Chinese AI researcher **Jianlin Su (苏剑林)** posed a profound question:

> *"Can we find an encoding function $f_Q(\mathbf{x}_m, m)$ and $f_K(\mathbf{x}_n, n)$ such that their inner product $\langle f_Q(\mathbf{x}_m, m), f_K(\mathbf{x}_n, n) \rangle$ is a function $g(\mathbf{x}_m, \mathbf{x}_n, m - n)$ that depends solely on the tokens and their relative offset $(m - n)$?"*

Su proved that under basic smoothness, linearity, and boundary conditions, the **unique** non-trivial solution to this functional equation in real coordinate space is the block-diagonal rotation matrix $\mathbf{R}_{\Theta, m}^d$.

Initially published on his research blog (*Scientific Spaces*) and later in the peer-reviewed paper *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021), the technique was quickly adopted by Meta's LLaMA team in 2023, solidifying RoPE as the uncontested global standard for open-weights and frontier LLMs.

---

### 3. Long-Context Scaling: RoPE Interpolation & YaRN

Because RoPE is based on continuous rotation angles $m\theta$, researchers discovered that models can be extended from $4\text{k}$ tokens to $32\text{k}$, $128\text{k}$, or even $1\text{M}$ tokens without retraining from scratch:

1. **Linear Position Interpolation (PI)**:
   To expand context window by a factor $s$ (e.g., $s = 4$ for $4\text{k} \to 16\text{k}$), simply divide position by $s$:
   $$
   m' = \frac{m}{s}
   $$
   This keeps all rotated angles within the familiar $[0, 4000\theta]$ interval seen during pretraining!

2. **NTK-Aware Scaling & YaRN**:
   High frequencies (fast clock hands) need different scaling than low frequencies (slow clock hands). By adjusting the base frequency $b$ from $10000$ to $500000$ or applying dimension-dependent scale factors, modern models like LLaMA-3 and Qwen-2.5 achieve rock-solid retrieval across 128,000-token documents!

---

## Step 5: Concrete Toy Example (Hand Arithmetic on a 2D Slice)

Let us calculate a complete numerical walkthrough by hand on a 2-dimensional subspace so you can verify every single decimal.

### 1. Setup

Let head dimension slice be $d = 2$ (a single 2D plane).
Let base frequency be $\theta = 0.5$ radians ($\approx 28.65^\circ$).

Suppose we have two tokens in a sentence:

<p align="center">
  <kbd>Token A (Query token at position m = 1)</kbd> &emsp;
  <kbd>Token B (Key token at position n = 3)</kbd>
</p>

Relative distance between them:

$$
n - m = 3 - 1 = 2 \quad \text{(Key is 2 steps to the right of Query)}
$$

Let their unrotated Query and Key vectors be:

$$
\mathbf{q}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{k}_3 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}
$$

Notice that the raw unrotated dot product is:

$$
\mathbf{q}_1^\top \mathbf{k}_3 = (1.0)(0.0) + (0.0)(1.0) = 0.0
$$

---

### 2. Rotating Token A at Position $m = 1$

The rotation angle for position $m = 1$ is:

$$
\phi_1 = m \theta = 1 \times 0.5 = 0.5 \text{ rad}
$$

Trigonometric values (to 4 decimal places):
- $\cos(0.5) \approx 0.8776$
- $\sin(0.5) \approx 0.4794$

The rotation matrix $\mathbf{R}_{0.5, 1}$ is:

$$
\mathbf{R}_{0.5, 1} = \begin{bmatrix}
\cos(0.5) & -\sin(0.5) \\
\sin(0.5) & \cos(0.5)
\end{bmatrix} = \begin{bmatrix}
0.8776 & -0.4794 \\
0.4794 & 0.8776
\end{bmatrix}
$$

Rotate Query $\mathbf{q}_1$:

$$
\widetilde{\mathbf{q}}_1 = \mathbf{R}_{0.5, 1} \mathbf{q}_1 = \begin{bmatrix}
0.8776 & -0.4794 \\
0.4794 & 0.8776
\end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.8776 \\ 0.4794 \end{bmatrix}
$$

---

### 3. Rotating Token B at Position $n = 3$

The rotation angle for position $n = 3$ is:

$$
\phi_3 = n \theta = 3 \times 0.5 = 1.5 \text{ rad}
$$

Trigonometric values:
- $\cos(1.5) \approx 0.0707$
- $\sin(1.5) \approx 0.9975$

The rotation matrix $\mathbf{R}_{0.5, 3}$ is:

$$
\mathbf{R}_{0.5, 3} = \begin{bmatrix}
\cos(1.5) & -\sin(1.5) \\
\sin(1.5) & \cos(1.5)
\end{bmatrix} = \begin{bmatrix}
0.0707 & -0.9975 \\
0.9975 & 0.0707
\end{bmatrix}
$$

Rotate Key $\mathbf{k}_3$:

$$
\widetilde{\mathbf{k}}_3 = \mathbf{R}_{0.5, 3} \mathbf{k}_3 = \begin{bmatrix}
0.0707 & -0.9975 \\
0.9975 & 0.0707
\end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -0.9975 \\ 0.0707 \end{bmatrix}
$$

---

### 4. Computing the RoPE Attention Dot Product

Now compute the attention score between the rotated vectors:

$$
\begin{aligned}
\langle \widetilde{\mathbf{q}}_1, \widetilde{\mathbf{k}}_3 \rangle &= \widetilde{\mathbf{q}}_1^\top \widetilde{\mathbf{k}}_3 \\
&= (0.8776)(-0.9975) + (0.4794)(0.0707) \\
&= -0.8754 + 0.0339 \\
&= \mathbf{-0.8415}
\end{aligned}
$$

---

### 5. Verification: Direct Relative Formula ($n - m = 2$)

Now let us check whether the direct relative formula gives the exact same result!

The relative distance is $n - m = 3 - 1 = 2$.
The relative angle is:

$$
\Delta\phi = (n - m)\theta = 2 \times 0.5 = 1.0 \text{ rad}
$$

Trigonometric values for angle $1.0$ rad:
- $\cos(1.0) \approx 0.5403$
- $\sin(1.0) \approx 0.8415$

The relative rotation matrix $\mathbf{R}_{0.5, 2}$ is:

$$
\mathbf{R}_{0.5, 2} = \begin{bmatrix}
\cos(1.0) & -\sin(1.0) \\
\sin(1.0) & \cos(1.0)
\end{bmatrix} = \begin{bmatrix}
0.5403 & -0.8415 \\
0.8415 & 0.5403
\end{bmatrix}
$$

Apply the relative identity $\mathbf{q}_1^\top \mathbf{R}_{0.5, 2} \mathbf{k}_3$:

$$
\begin{aligned}
\mathbf{q}_1^\top \mathbf{R}_{0.5, 2} \mathbf{k}_3 &= \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 0.5403 & -0.8415 \\ 0.8415 & 0.5403 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} \\
&= \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} -0.8415 \\ 0.5403 \end{bmatrix} \\
&= \mathbf{-0.8415}
\end{aligned}
$$

<mark>The numbers match to the fourth decimal place!</mark>

---

### 6. The Shift Invariance Test (Translating the Entire Sentence by +10)

What if the sentence appears later in a long prompt, say at positions $m' = 11$ and $n' = 13$?

The relative distance is still:

$$
n' - m' = 13 - 11 = 2
$$

Let us calculate:
1. $\phi_{11} = 11 \times 0.5 = 5.5 \text{ rad}$. $\cos(5.5) \approx 0.7087, \sin(5.5) \approx -0.7055$.
   $\widetilde{\mathbf{q}}_{11} = [0.7087, -0.7055]^\top$.
2. $\phi_{13} = 13 \times 0.5 = 6.5 \text{ rad}$. $\cos(6.5) \approx 0.9766, \sin(6.5) \approx 0.2151$.
   $\widetilde{\mathbf{k}}_{13} = [-0.2151, 0.9766]^\top$.
3. Inner product:
   $$
   \widetilde{\mathbf{q}}_{11}^\top \widetilde{\mathbf{k}}_{13} = (0.7087)(-0.2151) + (-0.7055)(0.9766) = -0.1524 - 0.6889 = \mathbf{-0.8413} \approx \mathbf{-0.8415}
   $$

The score remains completely invariant under arbitrary spatial translations!

---

## Step 6: Core Takeaway

> [!TIP] The Punchline of RoPE
> **Rotary Position Embedding turns the abstract concept of token distance into the geometric angle between spinning dials in 2D coordinate planes.**
>
> By replacing additive vector noise with multiplicative orthogonal rotations, RoPE guarantees that attention scores depend strictly on relative token distances $(m - n)$, empowering modern LLMs to preserve word order cleanly and scale seamlessly to hundred-thousand-token contexts.
