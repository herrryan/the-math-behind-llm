# Chapter 11: Looking Through Different Glasses (Multi-Head Attention)


## Step 1: 3-Year-Old Intuition (The Detective Squad & The Three Pairs of Glasses) {: #step-1 }

!!! note "3-Year-Old Intuition: The Detective Squad and the Tinted Sunglasses"
    Imagine a detective agency run by three clever children: Timmy, Lily, and Sam.

    One afternoon, a mystery note arrives at their clubhouse:

    *"The brave puppy chased the flying butterfly yesterday because she was excited."*

    The children need to understand everything that happened in the story. But each child wears a different pair of **magic tinted glasses**:

    1. **Timmy wears Blue Grammar Glasses**:
       - When Timmy reads the note, his blue lenses only highlight **who did what**.
       - His eyes instantly connect *"puppy"* to *"chased"* and *"butterfly"*. He doesn't care whether it happened yesterday or a hundred years ago; he only cares about who is the hero and who is being chased!

    2. **Lily wears Red Emotion Glasses**:
       - When Lily reads the note, her red lenses only highlight **how everyone feels**.
       - Her eyes immediately connect *"brave"* and *"excited"* back to the *"puppy"*. She understands the happy, playful mood of the scene!

    3. **Sam wears Green Timeline Glasses**:
       - When Sam reads the note, his green lenses only care about **when and why**.
       - His eyes connect *"chased"* to *"yesterday"* and *"because"*. He anchors the action to a specific spot on the calendar!

    Now imagine what would happen if the clubhouse had only **one single child** who tried to wear all three colored lenses stacked on top of each other at the exact same moment!

    The world would turn into a muddy, blurry brownish mess! If you try to pay attention to grammar, emotion, and time simultaneously with a single pair of eyes, all the distinct clues blur into an unreadable average.

    Instead, each child works independently in their own corner, writes down their specialized findings on a colorful card, and then brings all three cards together to pin them side by side onto the master bulletin board.

    In Large Language Models, this detective squad is called **Multi-Head Attention (MHA)**. Rather than forcing a single attention mechanism to track every nuance of human language at once, the model splits its brain into multiple independent "heads", lets each head look at the sentence through its own specialized subspace, and then glues their insights together!

<figure>
<pre>
The Single-Head Bottleneck (One Eye Trying to See Everything):

Sentence: "The bank of the river approved the business loan."
Single Attention Head:
  "bank" ──► tries to look at "river" (geography) AND "loan" (finance) simultaneously
  Result: Both signals get averaged into a single blurry dot-product score!

The Multi-Head Solution (Specialized Detective Squad):

Input Vector [1 x 4] ───┬──► Head 1 (Syntax / Grammar)  ──► Focuses strongly on ("bank" ◄──► "loan")
                         ├──► Head 2 (Topic / Geography) ──► Focuses strongly on ("bank" ◄──► "river")
                         └──► Head 3 (Coreference)       ──► Focuses strongly on ("loan" ◄──► "approved")
                                              │
                                              ▼
                         Concat(Head 1, Head 2, Head 3)
                                              │
                                              ▼
                         Final Output Projection Matrix (W_O)
</pre>
<figcaption><strong>Figure 11.1:</strong> Multi-Head Attention allows the model to simultaneously interrogate syntactic, semantic, and contextual relationships across distinct subspaces.</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: The Subspace Averaging Bottleneck"
    In Chapter 08, we formulated the standard Scaled Dot-Product Attention:

    $$
    \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
    $$

    For any pair of tokens $i$ and $j$, the term $\mathbf{q}_i^\top \mathbf{k}_j$ produces **a single scalar score**.

    Once passed through Softmax, this scalar becomes a single non-negative weight $A_{ij} \in [0, 1]$.

    But natural language words participate in many different types of relationships simultaneously:
    - **Syntactic relationships**: Verb to direct object (*"ate"* $\to$ *"apple"*).
    - **Coreference relationships**: Pronoun to antecedent (*"it"* $\to$ *"robot"*).
    - **Long-distance dependencies**: Subject to distant verb separated by parenthetical clauses.
    - **Semantic disambiguation**: Word sense selection (*"apple"* as fruit vs. tech company).

    If a model has only a single attention head, it must compress all of these diverse relationships into a single probability distribution. If token $i$ assigns $80\%$ of its attention budget to its syntactic subject, it has almost zero probability mass left to track emotional tone or temporal order!

    Furthermore, an engineer might naively suggest: *"If 1 head is good, let's run 8 full-sized attention mechanisms in parallel!"*
    But if each head operated on the full hidden dimension $d_{\text{model}} = 4096$, the compute cost and memory would instantly explode by a factor of 8!

    *"How do we mathematically split a high-dimensional vector space into $h$ independent, specialized subspaces so the model attends to multiple perspectives simultaneously, while keeping total computational cost and total parameter count strictly identical to a single full-rank head?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Multi-Head Attention Architecture

In *Attention Is All You Need* (Vaswani et al., 2017), \lt dfn id="def-mha">Multi-Head Attention (MHA)</dfn> projects the Queries, Keys, and Values $h$ times with different, learned linear projections into lower-dimensional subspaces.

Given input sequence representations $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{T \times d_{\text{model}}}$:

$$
\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\operatorname{head}_1, \operatorname{head}_2, \dots, \operatorname{head}_h)\mathbf{W}^O
$$

where each individual attention head is calculated independently:

$$
\operatorname{head}_i = \operatorname{Attention}\left(\mathbf{Q}\mathbf{W}_i^Q, \, \mathbf{K}\mathbf{W}_i^K, \, \mathbf{V}\mathbf{W}_i^V\right) = \operatorname{softmax}\left(\frac{(\mathbf{Q}\mathbf{W}_i^Q)(\mathbf{K}\mathbf{W}_i^K)^\top}{\sqrt{d_k}}\right)(\mathbf{V}\mathbf{W}_i^V)
$$

---

### 2. Tensor Dimensions and the Subspace Partition

The dimensionality parameters of Multi-Head Attention are structured with mathematical precision:

\lt details>
\lt summary>\lt strong>Mathematical Symbol Catalog & Dimensionality Mapping</strong></summary>
\lt dl>
  \lt dt>\lt strong>$d_{\text{model}}$ (Model Dimension)</strong></dt>
  \lt dd>The width of the token representation vectors in the residual stream (e.g., $d_{\text{model}} = 512$ in standard Transformer, $4096$ in LLaMA-7B, $8192$ in LLaMA-70B).</dd>
  \lt dt>\lt strong>$h$ (Number of Heads)</strong></dt>
  \lt dd>The number of parallel attention subspaces (e.g., $h = 8$ in standard Transformer, $h = 32$ in LLaMA-7B, $h = 64$ in LLaMA-70B).</dd>
  \lt dt>\lt strong>$d_k$ (Key / Query Head Dimension)</strong></dt>
  \lt dd>The dimensionality of Query and Key vectors inside each individual head: $d_k = \frac{d_{\text{model}}}{h}$ (typically $d_k = 64$ or $128$).</dd>
  \lt dt>\lt strong>$d_v$ (Value Head Dimension)</strong></dt>
  \lt dd>The dimensionality of Value vectors inside each head (almost always set to $d_v = d_k = \frac{d_{\text{model}}}{h}$).</dd>
  \lt dt>\lt strong>$\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  \lt dd>Learned Query projection matrix for head $i \in \{1, 2, \dots, h\}$.</dd>
  \lt dt>\lt strong>$\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  \lt dd>Learned Key projection matrix for head $i \in \{1, 2, \dots, h\}$.</dd>
  \lt dt>\lt strong>$\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$</strong></dt>
  \lt dd>Learned Value projection matrix for head $i \in \{1, 2, \dots, h\}$.</dd>
  \lt dt>\lt strong>$\mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$</strong></dt>
  \lt dd>Learned Output projection matrix that mixes and combines all head representations back into the model dimension.</dd>
</dl>
</details>

---

### 3. The Computational Invariance Magic (Why MHA Costs the Same as 1 Head)

A common misconception among beginners is that having 8 heads makes the Transformer 8 times slower or 8 times bigger. In reality, **the total parameter count and the total FLOPs are strictly identical**!

Let us prove this rigorously.

#### A. Total Parameter Count of Projections

For a single gigantic head operating on full dimension $d_{\text{model}}$:
- Projection parameter count for $\mathbf{W}^Q$: $d_{\text{model}} \times d_{\text{model}} = d_{\text{model}}^2$.

Now calculate the parameter count for $h$ independent heads, where each head has dimension $d_k = \frac{d_{\text{model}}}{h}$:

$$
\sum_{i=1}^h \operatorname{size}(\mathbf{W}_i^Q) = h \times \left(d_{\text{model}} \times d_k\right) = h \times \left(d_{\text{model}} \times \frac{d_{\text{model}}}{h}\right) = d_{\text{model}}^2
$$

The factor of $h$ in the head count and the factor of $\frac{1}{h}$ in the subspace dimension cancel out completely!

The same cancellation applies to $\mathbf{W}^K$, $\mathbf{W}^V$, and the output projection $\mathbf{W}^O$:

$$
\operatorname{size}(\mathbf{W}^O) = (h \cdot d_v) \times d_{\text{model}} = \left(h \cdot \frac{d_{\text{model}}}{h}\right) \times d_{\text{model}} = d_{\text{model}}^2
$$

#### B. Total Attention FLOPs

For each head $i$, computing $\mathbf{Q}_i \mathbf{K}_i^\top$ takes $T^2 d_k$ multiply-accumulate operations.
Across all $h$ heads:

$$
\text{Total FLOPs} = h \times (T^2 d_k) = h \times \left(T^2 \frac{d_{\text{model}}}{h}\right) = T^2 d_{\text{model}}
$$

Multi-Head Attention divides the feature space into $h$ orthogonal channels without spending an extra single floating-point operation!

---

### 4. Modern KV Cache Variants: MHA vs. MQA vs. GQA

During LLM generation (inference), the model must store the Keys and Values of all past tokens in GPU memory (the \lt abbr title="Key-Value Cache">KV Cache</abbr>). As context windows expanded to 32k and 128k tokens, the memory required to store $h$ Key and Value matrices per layer became the ultimate hardware bottleneck.

To break this bottleneck, researchers engineered two major architectural evolutions:

\lt figure>
\lt pre>
Comparison of Attention Head Configurations:

1. Multi-Head Attention (MHA) ── Vaswani et al. (2017)
   Queries: [ Q_1 ] [ Q_2 ] [ Q_3 ] [ Q_4 ] [ Q_5 ] [ Q_6 ] [ Q_7 ] [ Q_8 ]
   Keys:    [ K_1 ] [ K_2 ] [ K_3 ] [ K_4 ] [ K_5 ] [ K_6 ] [ K_7 ] [ K_8 ]  ◄── 8 KV heads in memory
   Values:  [ V_1 ] [ V_2 ] [ V_3 ] [ V_4 ] [ V_5 ] [ V_6 ] [ V_7 ] [ V_8 ]

2. Multi-Query Attention (MQA) ── Shazeer (2019)
   Queries: [ Q_1 ] [ Q_2 ] [ Q_3 ] [ Q_4 ] [ Q_5 ] [ Q_6 ] [ Q_7 ] [ Q_8 ]
   Keys:    [                     Shared Single Key (K)                    ]  ◄── 1 KV head in memory
   Values:  [                    Shared Single Value (V)                   ]      (8x cache reduction!)

3. Grouped-Query Attention (GQA) ── Ainslie et al. (2023) [LLaMA-3, Mistral]
   Queries: [ Q_1   Q_2 ] [ Q_3   Q_4 ] [ Q_5   Q_6 ] [ Q_7   Q_8 ]
   Keys:    [   K_1     ] [   K_2     ] [   K_3     ] [   K_4     ]        ◄── 4 KV heads in memory
   Values:  [   V_1     ] [   V_2     ] [   V_3     ] [   V_4     ]          (Best balance of quality & speed)
</pre>
\lt figcaption>\lt strong>Figure 11.2:</strong> Structural difference between MHA (standard), MQA (extreme memory savings), and GQA (modern production standard).</figcaption>
</figure>

1. **Multi-Query Attention (MQA, Shazeer 2019)**:
   - Uses $h$ distinct Query heads, but only **1 shared Key head and 1 shared Value head**.
   - Reduces KV cache size by a factor of $h$ (e.g., an $8\times$ to $32\times$ memory reduction), enabling massive batch sizes.
2. **Grouped-Query Attention (GQA, Ainslie et al. 2023)**:
   - Groups $h$ Query heads into $g$ groups (where $1 \lt g \lt h$). Each group of Query heads shares a single Key-Value head pair.
   - For example, in LLaMA-3-70B ($h = 64$ Query heads, $g = 8$ KV heads), each KV head is shared across 8 Query heads.
   - Preserves almost 100% of MHA's representational capacity while slashing inference memory by $8\times$!

---

## Step 4: Where Did It Come From? {: #step-4 }

\lt figure>
\lt pre>
The Evolutionary Journey of Multi-Perspective Attention:

1990s-2015: Single Ensemble Classifiers ──► Train N separate models and average outputs.
                                             Massive parameter blowup (N x parameters).
      │
      ▼
2016: Self-Attention in NLP (Lin et al.) ──► Single-head self-attention.
                                             Suffered from subspace collapse; could only focus on 1 topic.
      │
      ▼
2017: Vaswani et al. (Vaswani et al.) ───► Multi-Head Attention (MHA)
                                             Discovered the dimension splitting trick (d_k = d_model / h).
                                             Zero parameter increase; multi-perspective representation.
      │
      ▼
2019: Noam Shazeer (Google) ─────────────► Multi-Query Attention (MQA)
                                             Identified KV-cache memory as the LLM inference bottleneck.
      │
      ▼
2023: Ainslie et al. (Google Research) ──► Grouped-Query Attention (GQA)
                                             Golden compromise between quality and inference throughput;
                                             Adopted by LLaMA-2/3, Mistral, Gemma, and DeepSeek.
</pre>
\lt figcaption>\lt strong>Figure 11.3:</strong> Historical timeline from early ensemble models to modern Grouped-Query Attention.</figcaption>
</figure>

### 1. The Historical Catalyst: The Collapse of Single-Head Attention

Before Multi-Head Attention was introduced in 2017, neural attention models used a single attention vector (e.g., Bahdanau et al., 2014; Lin et al., 2017). Researchers noticed that the single attention head suffered from severe **representation collapse**:

When a word had both syntactic duty (e.g., agreeing in number with a distant auxiliary verb) and semantic duty (e.g., modifying a noun), the single attention distribution was forced to compromise. It placed half its weight on the verb and half on the noun, resulting in a diluted, lukewarm linear combination that failed at both tasks!

By projecting $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into multiple orthogonal subspaces, Vaswani et al. enabled individual heads to specialize:
- Head 1: Syntactic subject-verb agreement.
- Head 2: Coreference resolution (pronouns to entities).
- Head 3: Positional proximity (attending to adjacent tokens).
- Head 4: Punctuation and sentence delimiter tracking.

---

### 2. Architectural Comparison: Trade-offs Across Paradigms

\lt fieldset>
\lt legend>\lt strong>Architectural Trade-offs: MHA vs. MQA vs. GQA</strong></legend>

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 11.1:</strong> Comparison of attention head architectures.</caption>
  \lt thead>
    \lt tr bgcolor="#f0eee6">
      \lt th align="left">Architecture</th>
      \lt th align="center">Query Heads</th>
      \lt th align="center">KV Heads</th>
      \lt th align="center">KV Cache Memory Ratio</th>
      \lt th align="left">Primary Trade-off / Verdict</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt td>\lt strong>Multi-Head Attention (MHA)</strong></td>
      \lt td align="center">$h$</td>
      \lt td align="center">$h$</td>
      \lt td align="center">$1.0\times$ (Baseline)</td>
      \lt td>
        \lt del>Heavy memory footprint!</del> Maximum expressive power during training, but KV cache consumes dozens of gigabytes during long-context inference.
      </td>
    </tr>
    \lt tr>
      \lt td>\lt strong>Multi-Query Attention (MQA)</strong></td>
      \lt td align="center">$h$</td>
      \lt td align="center">$1$</td>
      \lt td align="center">$\frac{1}{h}\times$ (Up to $32\times$ smaller)</td>
      \lt td>
        \lt del>Minor quality degradation.</del> Extremely fast inference and massive batching capacity, but can suffer slight accuracy drops on complex reasoning.
      </td>
    </tr>
    \lt tr bgcolor="#fdfdf0">
      \lt td>\lt strong>Grouped-Query Attention (GQA)</strong></td>
      \lt td align="center">$h$</td>
      \lt td align="center">$g$ ($1 \lt g \lt h$)</td>
      \lt td align="center">$\frac{g}{h}\times$ (Typically $8\times$ smaller)</td>
      \lt td>
        \lt ins>\lt strong>The Modern Production Standard!</strong></ins> Matches MHA accuracy across virtually all benchmarks while delivering MQA-level inference speed and memory savings.
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

## Step 5: Concrete Toy Example (Hand Arithmetic with $h = 2$ Heads) {: #step-5 }

Let us trace a complete numerical example with tiny numbers so you can verify every addition, multiplication, and concatenation by hand.

### 1. Setup

Let sequence length be $T = 2$ (tokens: \lt kbd>"The"</kbd> and \lt kbd>"cat"</kbd>).
Let model dimension be $d_{\text{model}} = 4$.
Let number of heads be $h = 2$.
Subspace dimension:

$$
d_k = d_v = \frac{d_{\text{model}}}{h} = \frac{4}{2} = 2
$$

Suppose the input token representations are:

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1^\top \\
\mathbf{x}_2^\top
\end{bmatrix} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 1.0
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

For simplicity, assume $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$.

---

### 2. Computing Head 1 ($i = 1$)

Let the projection matrices for Head 1 be:

$$
\mathbf{W}_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad
\mathbf{W}_1^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad
\mathbf{W}_1^V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{4 \times 2}
$$

Notice that Head 1 completely ignores the 3rd and 4th dimensions of $\mathbf{X}$! It projects into the subspace of the first two dimensions:

$$
\mathbf{Q}_1 = \mathbf{X}\mathbf{W}_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{K}_1 = \mathbf{X}\mathbf{W}_1^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{V}_1 = \mathbf{X}\mathbf{W}_1^V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

Compute the raw score matrix for Head 1 (with $\sqrt{d_k} = \sqrt{2} \approx 1.414$):

$$
\mathbf{Q}_1 \mathbf{K}_1^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}
$$

Scaled scores $\mathbf{S}_1 = \frac{\mathbf{Q}_1 \mathbf{K}_1^\top}{\sqrt{2}}$:

$$
\mathbf{S}_1 = \begin{bmatrix} \frac{1}{\sqrt{2}} & 0 \\ 0 & \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 & 0.0000 \\ 0.0000 & 0.7071 \end{bmatrix}
$$

Row-wise Softmax:
- Row 1: $e^{0.7071} \approx 2.0281$, $e^{0} = 1.0000$. $\sum = 3.0281$.
  $A_{11} = \frac{2.0281}{3.0281} \approx 0.67$, $A_{12} = \frac{1.0}{3.0281} \approx 0.33$.
- Row 2 (by symmetry): $A_{21} \approx 0.33$, $A_{22} \approx 0.67$.

$$
\mathbf{A}_1 \approx \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
$$

Head 1 output representation:

$$
\operatorname{head}_1 = \mathbf{A}_1 \mathbf{V}_1 = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0.67 & 0.66 \\ 0.33 & 1.34 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

---

### 3. Computing Head 2 ($i = 2$)

Now let the projection matrices for Head 2 focus on the 3rd and 4th dimensions:

$$
\mathbf{W}_2^Q = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{W}_2^K = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
\mathbf{W}_2^V = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 3 & 0 \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 2}
$$

Notice that $\mathbf{W}_2^K$ flips the two axes!
Compute projections:

$$
\mathbf{Q}_2 = \mathbf{X}\mathbf{W}_2^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{K}_2 = \mathbf{X}\mathbf{W}_2^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
\mathbf{V}_2 = \mathbf{X}\mathbf{W}_2^V = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

Compute raw dot products for Head 2:

$$
\mathbf{Q}_2 \mathbf{K}_2^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix}
$$

Scaled scores $\mathbf{S}_2$:

$$
\mathbf{S}_2 \approx \begin{bmatrix} 0.0000 & 0.7071 \\ 0.7071 & 0.0000 \end{bmatrix}
$$

Softmax weights $\mathbf{A}_2$:

$$
\mathbf{A}_2 \approx \begin{bmatrix} 0.33 & 0.67 \\ 0.67 & 0.33 \end{bmatrix}
$$

\lt mark>Notice the profound difference: Head 1 attended mostly to itself (diagonal = 0.67), while Head 2 attended mostly to the opposite token (off-diagonal = 0.67)!</mark>

Head 2 output representation:

$$
\operatorname{head}_2 = \mathbf{A}_2 \mathbf{V}_2 = \begin{bmatrix} 0.33 & 0.67 \\ 0.67 & 0.33 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.99 & 0.67 \\ 2.01 & 0.33 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

---

### 4. Concatenation of Heads

Now concatenate the two specialized 2D outputs along the feature dimension:

$$
\mathbf{H}_{\text{cat}} = \operatorname{Concat}(\operatorname{head}_1, \operatorname{head}_2) = \begin{bmatrix}
0.67 & 0.66 & 0.99 & 0.67 \\
0.33 & 1.34 & 2.01 & 0.33
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

Each row now contains the combined, multi-perspective intelligence of both heads!

---

### 5. Final Output Projection ($\mathbf{W}^O$)

Finally, the concatenated tensor is multiplied by the learned output projection matrix $\mathbf{W}^O \in \mathbb{R}^{4 \times 4}$. For simplicity, let $\mathbf{W}^O = \mathbf{I}_4$ (identity matrix):

$$
\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{H}_{\text{cat}} \mathbf{W}^O = \begin{bmatrix}
0.67 & 0.66 & 0.99 & 0.67 \\
0.33 & 1.34 & 2.01 & 0.33
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

Both tokens now possess rich, multi-dimensional representations incorporating both self-focus and cross-token contextual information, produced with zero parameter blowup!

---

## Step 6: Core Takeaway {: #step-6 }

!!! tip "Key Insight: The Punchline of Multi-Head Attention"
    **Multi-Head Attention divides the high-dimensional hidden space into parallel orthogonal sub-spaces, allowing the model to simultaneously track grammar, emotion, and context through multiple specialized lenses.**

    By setting each head dimension to $d_k = d_{\text{model}} / h$, the Transformer achieves rich ensemble-like expressiveness without adding a single extra parameter or floating-point operation over a naive single-head architecture.
