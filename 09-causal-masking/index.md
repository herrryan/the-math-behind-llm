# Chapter 09: Blindfolds on Future Words (Causal Masking)

---

## Step 1: 3-Year-Old Intuition (The Sliding Storybook Card)

> [!INTUITION] The Storybook Guessing Game & The Cardboard Slider
> Imagine you are sitting at a little wooden desk playing a storybook guessing game.
>
> On the desk lies an exciting picture book. But you are not just reading the book—you are playing a detective game with your teacher. Your job is to guess what word comes next in the sentence before you read it out loud.
>
> The sentence printed on the page says:
>
> *"The brave little puppy chased the bouncy ball."*
>
> Now imagine you are looking at the word **"chased"**. The teacher asks you: *"What do you think happens right after chased?"*
>
> What would happen if the whole page were left wide open?
>
> You wouldn't need to think at all! You wouldn't need to remember what puppies like to do, and you wouldn't need to understand the story. You could just slide your eyes two inches to the right, peek at the word **"the"**, and shout it out.
>
> You didn't learn how stories work. You didn't exercise your brain. You just cheated by peeking at tomorrow's page!
>
> To stop you from cheating, the teacher places a thick cardboard slider over the page.
>
> As your finger slides from left to right, the cardboard slider only lets you see the words you have already touched. Everything to the right of your finger is locked behind a solid black wall.
>
> 1. **Yesterday's Words Are Open**: You can look back at *"The"*, *"brave"*, and *"puppy"* as much as you want to gather clues.
> 2. **Tomorrow's Words Are Blocked**: You cannot see even a single letter of what comes next.
> 3. **Genuine Thinking**: Because you cannot peek, your brain has to work hard, weigh the clues, and learn how words actually connect in the real world!
>
> In Large Language Models, this black cardboard slider is called the **Causal Mask**. It is a mathematical blindfold that allows the model to learn from the past while strictly forbidding it from peeking into the future!

<figure>
<pre>
Reading Timeline (Words Processed Left to Right):

Step 1: Reading "The"
Visible to Brain: [The] ───────► Hidden Behind Card: [???] [???] [???] [???]
(Brain can only attend to "The")

Step 2: Reading "brave"
Visible to Brain: [The] [brave] ─► Hidden Behind Card: [???] [???] [???]
(Brain attends to "The" and "brave")

Step 3: Reading "puppy"
Visible to Brain: [The] [brave] [puppy] ──► Hidden Behind Card: [???] [???]
(Brain attends to "The", "brave", and "puppy")

Unmasked (Cheating):   "The" looks ahead at "puppy" ──► Zero Thinking, Model Collapses!
Masked (Honest Study): Each word only sees its past ──► Deep Thinking, Genuine Learning!
</pre>
<figcaption><strong>Figure 9.1:</strong> The Cardboard Slider ensures that each word can only attend to past and present words, never future ones.</figcaption>
</figure>

---

## Step 2: The Bridging Question

> [!BRIDGING] From Full Attention to the Arrow of Time
> In Chapter 08, we assembled the complete <dfn id="def-attention">Scaled Dot-Product Attention</dfn> equation:
>
> $$
> \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
> $$
>
> Look closely at the raw score matrix $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$. For a sequence of length $T$, this matrix computes a score for every single pair of positions $(i, j)$:
>
> - Row $i$ represents the token asking the question (**Query**).
> - Column $j$ represents the token providing the clue (**Key**).
>
> In an unconstrained matrix multiplication, token $1$ computes an attention score with token $5$. When the Softmax is evaluated, token $1$ receives a positive percentage weight from token $5$, allowing information from token $5$ to flow directly into token $1$'s output representation.
>
> For bidirectional tasks (like sentence classification in <abbr title="Bidirectional Encoder Representations from Transformers">BERT</abbr>), seeing both directions is helpful. But an <abbr title="Large Language Model">LLM</abbr> is an **autoregressive generator**:
>
> $$
> P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_{\lt t})
> $$
>
> During real-world generation (inference), token $5$ does not exist yet when the model is generating token $2$. If the model is allowed to peek at token $5$ during training, it will develop a fatal dependency on future clues that will simply vanish at runtime. The model will utterly collapse.
>
> But how do we enforce this temporal restriction?
>
> - If we train one word at a time in a sequential `for` loop, we destroy GPU parallelism and training takes centuries.
> - We want to feed all $T$ tokens into the GPU simultaneously in a single, massive matrix multiplication, yet strictly forbid future tokens from receiving any attention weight.
>
> *"How do we mathematically modify the attention score matrix so that future tokens receive strictly zero attention weight ($0.0000\%$) and zero gradient, while preserving full GPU parallelism in a single unified operation?"*

---

## Step 3: The Exact Math & Formula

### 1. The Causal Masked Attention Equation

In modern autoregressive Transformers (such as GPT-4, LLaMA-3, and Gemini), the attention mechanism incorporates an additive <dfn id="def-causal-mask">Causal Mask Matrix</dfn> $\mathbf{M}$:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}
$$

The causal mask matrix $\mathbf{M} \in \mathbb{R}^{T \times T}$ is defined piecewise for every row index $i$ and column index $j$:

$$
M_{ij} = \begin{cases}
0 & \text{if } j \le i \quad \text{(past and present: allowed)} \\
-\infty & \text{if } j > i \quad \text{(future: strictly forbidden)}
\end{cases}
$$

Written out as an explicit $T \times T$ matrix:

$$
\mathbf{M} = \begin{bmatrix}
0 & -\infty & -\infty & \dots & -\infty \\
0 & 0 & -\infty & \dots & -\infty \\
0 & 0 & 0 & \dots & -\infty \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & 0
\end{bmatrix} \in \mathbb{R}^{T \times T}
$$

<details>
<summary><strong>Mathematical Symbol Catalog & Tensor Dimensions</strong></summary>
<dl>
  <dt><strong>$T$ (Sequence Length)</strong></dt>
  <dd>The total number of tokens in the input context window (e.g., $T = 2048$ or $T = 8192$).</dd>
  <dt><strong>$d_k$ (Head Dimension)</strong></dt>
  <dd>The dimensionality of Query and Key vectors (typically $d_k = 64$ or $d_k = 128$).</dd>
  <dt><strong>$d_v$ (Value Dimension)</strong></dt>
  <dd>The dimensionality of Value vectors (typically $d_v = d_k = 64$).</dd>
  <dt><strong>$\mathbf{Q} \in \mathbb{R}^{T \times d_k}$</strong></dt>
  <dd>The packed Query matrix. Row $i$, denoted $\mathbf{q}_i^\top$, is the search query generated by token $i$.</dd>
  <dt><strong>$\mathbf{K} \in \mathbb{R}^{T \times d_k}$</strong></dt>
  <dd>The packed Key matrix. Row $j$, denoted $\mathbf{k}_j^\top$, is the matching label provided by token $j$.</dd>
  <dt><strong>$\mathbf{V} \in \mathbb{R}^{T \times d_v}$</strong></dt>
  <dd>The packed Value matrix. Row $j$, denoted $\mathbf{v}_j^\top$, is the information payload carried by token $j$.</dd>
  <dt><strong>$\mathbf{M} \in \mathbb{R}^{T \times T}$</strong></dt>
  <dd>The causal mask matrix. An upper-triangular matrix filled with $-\infty$ above the main diagonal and $0$ on and below the diagonal.</dd>
  <dt><strong>$i$ (Row Index / Query Position)</strong></dt>
  <dd>The observer token position seeking contextual information ($1 \le i \le T$).</dd>
  <dt><strong>$j$ (Column Index / Key Position)</strong></dt>
  <dd>The source token position being observed ($1 \le j \le T$).</dd>
  <dt><strong>$-\infty$ (Negative Infinity)</strong></dt>
  <dd>The mathematical absorbing element for exponentiation: $\lim_{x \to -\infty} \exp(x) = 0$.</dd>
</dl>
</details>

---

### 2. The Arithmetic of $-\infty$ Inside Softmax

Why does adding $-\infty$ before Softmax eliminate future tokens with mathematical perfection?

Let $S_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}$ be the raw scaled dot-product compatibility score between Query token $i$ and Key token $j$. The masked score matrix entry is:

$$
\widetilde{S}_{ij} = S_{ij} + M_{ij}
$$

Now apply the standard Softmax formula across row $i$:

$$
A_{ij} = \frac{\exp(\widetilde{S}_{ij})}{\sum_{k=1}^T \exp(\widetilde{S}_{ik})} = \frac{\exp(S_{ij} + M_{ij})}{\sum_{k=1}^T \exp(S_{ik} + M_{ik})}
$$

We evaluate this expression under two distinct cases:

#### Case A: Future Tokens ($j > i$)

For any token appearing after position $i$, the mask entry is $M_{ij} = -\infty$:

$$
S_{ij} + M_{ij} = S_{ij} + (-\infty) = -\infty
$$

Passing this into the exponential function yields:

$$
\exp(S_{ij} + M_{ij}) = \exp(-\infty) = 0
$$

Because the numerator is strictly zero, the attention weight assigned to any future token vanishes completely:

$$
A_{ij} = \frac{0}{\sum_{k=1}^T \exp(\widetilde{S}_{ik})} = 0.0000 \quad (\forall j > i)
$$

Future words receive strictly **0% attention**. No signal from future values $\mathbf{v}_j$ can leak into the output representation of token $i$.

#### Case B: Past and Present Tokens ($j \le i$)

For any token appearing at or before position $i$, the mask entry is $M_{ij} = 0$:

$$
S_{ij} + M_{ij} = S_{ij} + 0 = S_{ij}
$$

$$
\exp(S_{ij} + M_{ij}) = \exp(S_{ij})
$$

Now examine the denominator of the Softmax equation. The sum splits into two parts:

$$
\sum_{k=1}^T \exp(\widetilde{S}_{ik}) = \sum_{k=1}^i \exp(S_{ik}) + \sum_{k=i+1}^T \underbrace{\exp(-\infty)}_{= 0} = \sum_{k=1}^i \exp(S_{ik})
$$

The terms for $k > i$ vanish into zero! Therefore, for all valid past tokens ($j \le i$), the attention weight simplifies to:

$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^i \exp(S_{ik})} \quad (\text{for } j \le i)
$$

Notice the mathematical elegance:
1. Future weights are strictly $0$.
2. Past weights automatically sum to exactly $1.0$ ($100\%$):
   $$
   \sum_{j=1}^T A_{ij} = \sum_{j=1}^i A_{ij} + \sum_{j=i+1}^T 0 = \frac{\sum_{j=1}^i \exp(S_{ij})}{\sum_{k=1}^i \exp(S_{ik})} = 1.0
   $$
3. No manual truncation, re-weighting, or post-normalization loops are required!

---

### 3. Lower-Triangular Structure of the Attention Matrix

Because $A_{ij} = 0$ for all $j > i$, the resulting attention weight matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$ is guaranteed to be a **lower-triangular matrix**:

$$
\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) = \begin{bmatrix}
1.0 & 0 & 0 & \dots & 0 \\
A_{21} & A_{22} & 0 & \dots & 0 \\
A_{31} & A_{32} & A_{33} & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
A_{T1} & A_{T2} & A_{T3} & \dots & A_{TT}
\end{bmatrix}
$$

<figure>
<pre>
Attention Weight Matrix A (Lower-Triangular Geometry):

      j = 1     j = 2     j = 3     j = 4     j = 5
     (Key 1)   (Key 2)   (Key 3)   (Key 4)   (Key 5)
   ┌─────────────────────────────────────────────────┐
i=1│  100%   │    0%   │    0%   │    0%   │    0%   │  ◄── Token 1 can only see Token 1
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=2│   35%   │   65%   │    0%   │    0%   │    0%   │  ◄── Token 2 sees Tokens 1, 2
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=3│   10%   │   20%   │   70%   │    0%   │    0%   │  ◄── Token 3 sees Tokens 1, 2, 3
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=4│    5%   │   15%   │   30%   │   50%   │    0%   │  ◄── Token 4 sees Tokens 1..4
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=5│    2%   │    8%   │   15%   │   25%   │   50%   │  ◄── Token 5 sees Tokens 1..5
   └─────────────────────────────────────────────────┘
     ▲──────────────────▲ └─────────────────────────┘
        Allowed Past              Forbidden Future
      (Weights sum to 1)        (Strictly zero: 0%)
</pre>
<figcaption><strong>Figure 9.2:</strong> Lower-triangular geometry of the causal attention matrix. Each row forms an independent probability distribution over current and past tokens.</figcaption>
</figure>

---

### 4. Zero Gradient Leakage (Strict Mathematical Causality)

What happens to the backpropagation gradients through the mask?

Recall from Chapter 07 that the gradient of the loss $\mathcal{L}$ with respect to a pre-softmax score $\widetilde{S}_{ij}$ is:

$$
\frac{\partial \mathcal{L}}{\partial \widetilde{S}_{ij}} = \sum_{k=1}^T \frac{\partial \mathcal{L}}{\partial A_{ik}} \frac{\partial A_{ik}}{\partial \widetilde{S}_{ij}}
$$

For any future position $j > i$, since $A_{ij} \equiv 0$ identically and $\exp(\widetilde{S}_{ij}) = 0$, the derivative of the softmax output with respect to future scores is identically zero:

$$
\frac{\partial A_{ik}}{\partial S_{ij}} = 0 \quad (\forall j > i, \, \forall k)
$$

Consequently:

$$
\frac{\partial \mathcal{L}}{\partial S_{ij}} = 0 \quad (\forall j > i)
$$

The mathematical gradient cannot flow across the temporal barrier. Parameters updated during training never receive any signal from future tokens that should be invisible.

---

### 5. Breaking the Left-to-Right Limitation: Fill-in-the-Middle (FIM / Mid-fill)

While the lower-triangular causal mask $\mathbf{M}$ is ideal for generating new text from scratch (predicting future words from past words), it introduces a major limitation: **standard causal models cannot edit or fill in the middle of existing text**.

Consider a software developer writing code:
```python
def calculate_area(radius):
    # [CURSOR: Model needs to write the formula here]
    return area
```
Here, the developer already wrote the code before the cursor (**Prefix**, $P$) and the code after the cursor (**Suffix**, $S$). The missing logic is the **Middle** ($M$).

Under standard causal generation, the model can only condition on the prefix:

$$
P(M \mid P)
$$

The model is completely blind to the suffix $S$ because the lower-triangular mask blocks future tokens! The model might invent an entirely different variable name (e.g., `result = 3.14 * radius ** 2`), causing a runtime crash because the suffix expects `area`.

#### The Fill-in-the-Middle (FIM) Sequence Transformation

In 2022, researchers at OpenAI (Bavarian et al., <cite>"Efficient Training of Language Models to Fill in the Middle"</cite>) discovered that you do **not** need to modify the Transformer architecture or remove the causal mask. Instead, you simply transform the document sequence using three special delimiter tokens: $\langle\text{PRE}\rangle$, $\langle\text{SUF}\rangle$, and $\langle\text{MID}\rangle$.

Given an arbitrary document split into three contiguous segments $D = (P, M, S)$:

1. **PSM (Prefix-Suffix-Middle) Mode**:
   $$
   \tau_{\text{PSM}}(D) = \langle\text{PRE}\rangle \circ P \circ \langle\text{SUF}\rangle \circ S \circ \langle\text{MID}\rangle \circ M \circ \langle\text{EOT}\rangle
   $$

2. **SPM (Suffix-Prefix-Middle) Mode**:
   $$
   \tau_{\text{SPM}}(D) = \langle\text{SUF}\rangle \circ S \circ \langle\text{PRE}\rangle \circ P \circ \langle\text{MID}\rangle \circ M \circ \langle\text{EOT}\rangle
   $$

where $\langle\text{EOT}\rangle$ is the End-of-Transmission token.

#### How FIM Works Under the Lower-Triangular Causal Mask

Because the sequence has been permuted, look at what the standard lower-triangular causal mask $\mathbf{M}$ allows each section to attend to:

<figure>
<pre>
Attention Connectivity in PSM Fill-in-the-Middle:

Token Position:    [PRE]  ...Prefix...  [SUF]  ...Suffix...  [MID]  ...Middle...
                   ┌────────────────────────────────────────────────────────┐
[PRE] + Prefix     │   Can Attend To    │             BLOCKED               │
                   │    Prefix Only     │         (Future Masked)           │
                   ├────────────────────┴───────────────────────────────────┤
[SUF] + Suffix     │   Can Attend To Both Prefix AND Suffix                 │
                   │             (Lower-Triangular History)                 │
                   ├────────────────────────────────────────────────────────┤
[MID] + Middle     │   CAN ATTEND TO PREFIX, SUFFIX, AND GENERATED MIDDLE!  │
                   │   Full bidirectional context across both anchors!      │
                   └────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>Figure 9.3:</strong> Under the standard causal mask, rearranging the sequence into PSM format allows the Middle segment to attend to both the Prefix and Suffix without leaking future Middle tokens.</figcaption>
</figure>

Notice the mathematical elegance:
1. When generating the middle tokens $M$, every Query in $M$ can attend to all Key-Value pairs of $P$ and $S$ because they physically appear *earlier* in the sequence.
2. The causal mask still prevents token $m_t$ from peeking at future middle tokens $m_{t+1}$.
3. Standard autoregressive loss is applied strictly over the middle segment:
   $$
   \mathcal{L}_{\text{FIM}} = -\sum_{k=1}^{|M|} \log P(m_k \mid \langle\text{PRE}\rangle, P, \langle\text{SUF}\rangle, S, \langle\text{MID}\rangle, m_{\lt k})
   $$
4. A single model trained with 50% standard text and 50% FIM-permuted text becomes simultaneously capable of both standard left-to-right generation and arbitrary in-place document editing.

---

## Step 4: Where Did It Come From?

<figure>
<pre>
The Evolution of Sequence Causality:

1913: Andrey Markov ────────► Discrete conditional chain: P(w_t | w_{t-1})
                               (Linguistic counting on Pushkin's Eugene Onegin)
      │
      ▼
1948: Claude Shannon ───────► Information Theory: Autoregressive entropy of text
      │
      ▼
1990: Jeffrey Elman ────────► Recurrent Neural Networks (RNNs):
                               Causality enforced by temporal loops: h_t = f(h_{t-1}, x_t)
                               Problem: O(T) sequential steps! GPUs sit idle.
      │
      ▼
2017: Vaswani et al. ───────► "Attention Is All You Need" (Section 3.2.3):
                               Causal Masking: Replace sequential time with spatial matrix mask!
                               Result: O(1) sequential steps, 100% GPU parallel saturation.
      │
      ▼
2018: Radford et al. ───────► GPT (Generative Pre-trained Transformer):
                               The pure causal decoder-only standard for modern LLMs.
      │
      ▼
2022: Bavarian et al. ──────► Fill-in-the-Middle (OpenAI FIM):
                               Permute document to [PRE] P [SUF] S [MID] M:
                               Unlocks mid-document code editing using standard causal masks!
</pre>
<figcaption><strong>Figure 9.4:</strong> From sequential recurrence to spatial matrix masking and Fill-in-the-Middle permutation.</figcaption>
</figure>

### 1. From Temporal Loops to Spatial Masking

In classical architectures like <abbr title="Recurrent Neural Networks">RNNs</abbr> and <abbr title="Long Short-Term Memory">LSTMs</abbr>, causality was enforced by **physical execution order**:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)
$$

Because step $t$ required $\mathbf{h}_{t-1}$ from the previous clock tick, it was physically impossible for token $t$ to look at token $t+1$. Future tokens simply had not arrived yet!

However, this temporal recurrence created a catastrophic hardware bottleneck:
- To process a sequence of $T = 2048$ tokens, the GPU had to execute $2048$ separate sequential operations.
- The GPU's thousands of parallel tensor cores could not be utilized.

Vaswani et al. (2017) realized that **physical time could be converted into a spatial matrix coordinate**:
1. Feed all $T$ tokens into the GPU simultaneously in a single matrix $\mathbf{X} \in \mathbb{R}^{T \times d}$.
2. Compute all $T \times T$ dot-product affinities in parallel using dense matrix multiplication.
3. Apply the additive mask $\mathbf{M}$ with $-\infty$ to mathematically wipe out future interactions.

This breakthrough transformed $O(T)$ sequential training steps into an **$O(1)$ parallel operation**, unlocking the massive scalability that made modern LLMs possible.

---

### 2. Why Additive $-\infty$? What Broke When Trying Simpler Alternatives?

When designing causal attention, researchers evaluated several intuitive alternatives. Every simpler method broke mathematically:

<fieldset>
<legend><strong>Three Failed Attempts & The Mathematical Winner</strong></legend>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 9.1:</strong> Comparison of methods for enforcing sequence causality in Attention.</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">Method</th>
      <th align="center">Mathematical Formulation</th>
      <th align="left">Why It Fails</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Attempt 1: Multiply Scores by 0</strong></td>
      <td align="center">$\mathbf{S} \odot \mathbf{M}_{\text{binary}}$</td>
      <td>
        <del>Catastrophic failure!</del> When $S_{ij} = 0$, Softmax evaluates $\exp(0) = 1.0$. Future tokens receive a baseline probability weight of $1.0$, completely contaminating attention instead of vanishing!
      </td>
    </tr>
    <tr>
      <td><strong>Attempt 2: Zero Weights After Softmax</strong></td>
      <td align="center">$\mathbf{A} \odot \mathbf{M}_{\text{binary}}$</td>
      <td>
        <del>Broken probability axioms!</del> The remaining past weights sum to less than $1.0$ ($\sum_{j=1}^i A_{ij} < 1$). The output vector $\mathbf{o}_i$ shrinks in norm, destabilizing layer variance and requiring a second division pass.
      </td>
    </tr>
    <tr>
      <td><strong>Attempt 3: Sequential Slicing</strong></td>
      <td align="center">For $t=1 \dots T$: $\mathbf{q}_t \mathbf{K}_{1:t}^\top$</td>
      <td>
        <del>Hardware bottleneck!</del> Requires $T$ separate kernel launches with dynamic tensor shapes. Destroys GPU tensor core saturation and memory bandwidth efficiency.
      </td>
    </tr>
    <tr bgcolor="#fdfdf0">
      <td><strong>The Winner: Additive $-\infty$ Mask</strong></td>
      <td align="center">$\operatorname{softmax}(\mathbf{S} + \mathbf{M})$</td>
      <td>
        <ins><strong>Mathematically perfect!</strong></ins> Through $\exp(-\infty) = 0$, future weights vanish to exactly $0$, past weights automatically sum to $1.0$, and the entire operation runs in a single parallel GPU kernel.
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

### 3. Engineering Reality: Floating-Point Implementation in PyTorch

In theoretical mathematics, we write $-\infty$. But how do computer processors handle $-\infty$?

In IEEE 754 floating-point arithmetic:
- In 32-bit single precision (`torch.float32`), `float('-inf')` is well-supported.
- However, in 16-bit half precision (`torch.float16` or `torch.bfloat16`), literal infinity can easily trigger numerical hazards:
  - If an entire row is masked (which can occur in cross-attention padding), $\sum \exp(-\infty) = 0$, leading to $\frac{0}{0} = \text{NaN}$.
  - Subtracting or adding large numbers to $-\infty$ can cause compiler optimization instabilities.

Because of this, production LLM codebases (including Hugging Face Transformers, Megatron-LM, and vLLM) implement causal masking using one of two strategies:

1. **Large Finite Negative Constant**:
   In `torch.float16`, the minimum representable normal value is $-65504$. Models typically use:
   ```python
   mask_value = torch.finfo(torch.float16).min  # Approximately -65504.0
   ```
   Since $\exp(-65504) = 0.0$ in half precision, this produces exact zeros without infinity hazards.

2. **Kernel-Level Triangular Optimization (FlashAttention)**:
   In modern inference and training engines like **FlashAttention-2** (Dao, 2023), the $T \times T$ mask matrix $\mathbf{M}$ is never even written to GPU <abbr title="High Bandwidth Memory">HBM</abbr> memory! The CUDA kernel simply skips computing dot products whenever the thread coordinates satisfy $j > i$. This saves $O(T^2)$ memory bandwidth while delivering identical mathematical results.

---

## Step 5: Concrete Toy Example (Hand Arithmetic with 3 Tokens)

Let us walk through the exact, step-by-step arithmetic of causal masking with tiny numbers so you can verify every single calculation by hand.

### 1. Setup

Imagine a miniature language model reading a 3-token sentence:

<p align="center">
  <kbd>Token 1: "The"</kbd> &emsp;
  <kbd>Token 2: "cat"</kbd> &emsp;
  <kbd>Token 3: "sat"</kbd>
</p>

Here sequence length is $T = 3$, and head dimension is $d_k = 2$.

Suppose the Query-Key scaled dot products have already been computed, yielding the raw unmasked score matrix $\mathbf{S} \in \mathbb{R}^{3 \times 3}$:

$$
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} = \begin{bmatrix}
2.0 & 1.0 & 4.0 \\
0.0 & 3.0 & 1.0 \\
1.0 & 2.0 & 5.0
\end{bmatrix}
$$

Let the Value matrix $\mathbf{V} \in \mathbb{R}^{3 \times 2}$ carry the following semantic content vectors:

$$
\mathbf{V} = \begin{bmatrix}
10.0 & 0.0 \\
0.0 & 20.0 \\
30.0 & 30.0
\end{bmatrix}
$$

---

### 2. Applying the Causal Mask

The $3 \times 3$ causal mask matrix is:

$$
\mathbf{M} = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$

Add the mask to the raw scores: $\widetilde{\mathbf{S}} = \mathbf{S} + \mathbf{M}$:

$$
\widetilde{\mathbf{S}} = \begin{bmatrix}
2.0 + 0 & 1.0 + (-\infty) & 4.0 + (-\infty) \\
0.0 + 0 & 3.0 + 0 & 1.0 + (-\infty) \\
1.0 + 0 & 2.0 + 0 & 5.0 + 0
\end{bmatrix} = \begin{bmatrix}
2.0 & -\infty & -\infty \\
0.0 & 3.0 & -\infty \\
1.0 & 2.0 & 5.0
\end{bmatrix}
$$

---

### 3. Step-by-Step Softmax Computation by Row

Now compute the Softmax independently for each row.

<fieldset>
<legend><strong>Row 1: Token 1 ("The")</strong></legend>

Token 1 is at position $i = 1$. It may only attend to position $j \le 1$.

1. **Exponentiate Row 1**:
   $$
   \begin{aligned}
   \exp(\widetilde{S}_{11}) &= \exp(2.0) \approx 7.3891 \\
   \exp(\widetilde{S}_{12}) &= \exp(-\infty) = 0 \\
   \exp(\widetilde{S}_{13}) &= \exp(-\infty) = 0
   \end{aligned}
   $$

2. **Sum of Exponents**:
   $$
   \sum_{k=1}^3 \exp(\widetilde{S}_{1k}) = 7.3891 + 0 + 0 = 7.3891
   $$

3. **Normalize**:
   $$
   \begin{aligned}
   A_{11} &= \frac{7.3891}{7.3891} = 1.0000 \quad (100.0\%) \\
   A_{12} &= \frac{0}{7.3891} = 0.0000 \quad (0.0\%) \\
   A_{13} &= \frac{0}{7.3891} = 0.0000 \quad (0.0\%)
   \end{aligned}
   $$

Token 1 concentrates $100\%$ of its attention on itself! It cannot see `"cat"` or `"sat"`.
</fieldset>

<fieldset>
<legend><strong>Row 2: Token 2 ("cat")</strong></legend>

Token 2 is at position $i = 2$. It may attend to positions $j \in \{1, 2\}$, but not $j = 3$.

1. **Exponentiate Row 2**:
   $$
   \begin{aligned}
   \exp(\widetilde{S}_{21}) &= \exp(0.0) = 1.0000 \\
   \exp(\widetilde{S}_{22}) &= \exp(3.0) \approx 20.0855 \\
   \exp(\widetilde{S}_{23}) &= \exp(-\infty) = 0
   \end{aligned}
   $$

2. **Sum of Exponents**:
   $$
   \sum_{k=1}^3 \exp(\widetilde{S}_{2k}) = 1.0000 + 20.0855 + 0 = 21.0855
   $$

3. **Normalize**:
   $$
   \begin{aligned}
   A_{21} &= \frac{1.0000}{21.0855} \approx 0.0474 \quad (4.74\%) \\
   A_{22} &= \frac{20.0855}{21.0855} \approx 0.9526 \quad (95.26\%) \\
   A_{23} &= \frac{0}{21.0855} = 0.0000 \quad (0.00\%)
   \end{aligned}
   $$

Notice that $A_{21} + A_{22} + A_{23} = 0.0474 + 0.9526 + 0.0 = 1.0000$ ($100\%$). Token 2 attends heavily to itself ($95.26\%$) and slightly to `"The"` ($4.74\%$), while `"sat"` is completely invisible.
</fieldset>

<fieldset>
<legend><strong>Row 3: Token 3 ("sat")</strong></legend>

Token 3 is at position $i = 3$. All positions $j \in \{1, 2, 3\}$ are in its past or present. None are masked!

1. **Exponentiate Row 3**:
   $$
   \begin{aligned}
   \exp(\widetilde{S}_{31}) &= \exp(1.0) \approx 2.7183 \\
   \exp(\widetilde{S}_{32}) &= \exp(2.0) \approx 7.3891 \\
   \exp(\widetilde{S}_{33}) &= \exp(5.0) \approx 148.4132
   \end{aligned}
   $$

2. **Sum of Exponents**:
   $$
   \sum_{k=1}^3 \exp(\widetilde{S}_{3k}) = 2.7183 + 7.3891 + 148.4132 = 158.5206
   $$

3. **Normalize**:
   $$
   \begin{aligned}
   A_{31} &= \frac{2.7183}{158.5206} \approx 0.0171 \quad (1.71\%) \\
   A_{32} &= \frac{7.3891}{158.5206} \approx 0.0466 \quad (4.66\%) \\
   A_{33} &= \frac{148.4132}{158.5206} \approx 0.9363 \quad (93.63\%)
   \end{aligned}
   $$

Sum check: $0.0171 + 0.0466 + 0.9363 = 1.0000$ ($100\%$). Token 3 can gather clues from the entire sentence up to this point.
</fieldset>

---

### 4. The Final Masked Attention Weight Matrix

Assembling the normalized probabilities row by row yields the lower-triangular attention matrix $\mathbf{A}$:

$$
\mathbf{A} = \begin{bmatrix}
1.0000 & 0.0000 & 0.0000 \\
0.0474 & 0.9526 & 0.0000 \\
0.0171 & 0.0466 & 0.9363
\end{bmatrix}
$$

Let us visualize these attention distributions with visual probability meters:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 9.2:</strong> Visual attention probability distribution per token.</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">Query Token</th>
      <th align="center">j=1 ("The")</th>
      <th align="center">j=2 ("cat")</th>
      <th align="center">j=3 ("sat")</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Token 1 ("The")</strong></td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="1.0"></meter><br>
        <strong>100.0%</strong>
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0"></meter><br>
        <samp>0.0% (Masked)</samp>
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0"></meter><br>
        <samp>0.0% (Masked)</samp>
      </td>
    </tr>
    <tr>
      <td><strong>Token 2 ("cat")</strong></td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0474"></meter><br>
        4.74%
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.9526"></meter><br>
        <strong>95.26%</strong>
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0"></meter><br>
        <samp>0.0% (Masked)</samp>
      </td>
    </tr>
    <tr>
      <td><strong>Token 3 ("sat")</strong></td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0171"></meter><br>
        1.71%
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0466"></meter><br>
        4.66%
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.9363"></meter><br>
        <strong>93.63%</strong>
      </td>
    </tr>
  </tbody>
</table>

---

### 5. Value Retrieval ($\mathbf{O} = \mathbf{A}\mathbf{V}$)

Now multiply the attention weight matrix $\mathbf{A} \in \mathbb{R}^{3 \times 3}$ by the Value matrix $\mathbf{V} \in \mathbb{R}^{3 \times 2}$ to produce the final contextualized outputs $\mathbf{O} \in \mathbb{R}^{3 \times 2}$:

$$
\mathbf{O} = \mathbf{A}\mathbf{V} = \begin{bmatrix}
1.0000 & 0.0000 & 0.0000 \\
0.0474 & 0.9526 & 0.0000 \\
0.0171 & 0.0466 & 0.9363
\end{bmatrix}
\begin{bmatrix}
10.0 & 0.0 \\
0.0 & 20.0 \\
30.0 & 30.0
\end{bmatrix}
$$

Let us compute each output vector by explicit linear combination:

#### Row 1 Output ($\mathbf{o}_1^\top$):
$$
\mathbf{o}_1^\top = 1.0000 \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} + 0.0 \begin{bmatrix} 0.0 & 20.0 \end{bmatrix} + 0.0 \begin{bmatrix} 30.0 & 30.0 \end{bmatrix} = \begin{bmatrix} 10.0000 & 0.0000 \end{bmatrix}
$$
Token 1 retrieves strictly its own value vector.

#### Row 2 Output ($\mathbf{o}_2^\top$):
$$
\begin{aligned}
\mathbf{o}_2^\top &= 0.0474 \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} + 0.9526 \begin{bmatrix} 0.0 & 20.0 \end{bmatrix} + 0.0 \begin{bmatrix} 30.0 & 30.0 \end{bmatrix} \\
&= \begin{bmatrix} 0.4740 & 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 & 19.0520 \end{bmatrix} \\
&= \begin{bmatrix} 0.4740 & 19.0520 \end{bmatrix}
\end{aligned}
$$
Token 2 blends information from `"The"` and `"cat"`.

#### Row 3 Output ($\mathbf{o}_3^\top$):
$$
\begin{aligned}
\mathbf{o}_3^\top &= 0.0171 \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} + 0.0466 \begin{bmatrix} 0.0 & 20.0 \end{bmatrix} + 0.9363 \begin{bmatrix} 30.0 & 30.0 \end{bmatrix} \\
&= \begin{bmatrix} 0.1710 & 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 & 0.9320 \end{bmatrix} + \begin{bmatrix} 28.0890 & 28.0890 \end{bmatrix} \\
&= \begin{bmatrix} 0.1710 + 28.0890 & 0.9320 + 28.0890 \end{bmatrix} \\
&= \begin{bmatrix} 28.2600 & 29.0210 \end{bmatrix}
\end{aligned}
$$
Token 3 blends information across all three words in the sentence.

The final output matrix is:

$$
\mathbf{O} = \begin{bmatrix}
10.0000 & 0.0000 \\
0.4740 & 19.0520 \\
28.2600 & 29.0210
\end{bmatrix}
$$

Every number is verified by clean hand arithmetic. The future remained strictly unreadable at every single step!

---

## Step 6: Core Takeaway

> [!TIP] The Punchline of Causal Masking
> **Causal masking harnesses the mathematical identity $\exp(-\infty) = 0$ to construct an impenetrable temporal shield over future tokens inside a single matrix addition.**
>
> By turning physical time into a spatial upper-triangular mask, modern Large Language Models achieve the ultimate balance: they train across thousands of tokens simultaneously with full GPU parallelism, while strictly guaranteeing that the model learns to predict, never to cheat.
