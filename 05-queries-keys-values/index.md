# Chapter 05: The Library Clue Hunt (Queries, Keys, and Values)

<nav aria-label="Table of Contents">
  <p>
    <strong>Table of Contents:</strong> 
    <a href="#step-1">1. 3-Year-Old Intuition</a> &bull; 
    <a href="#step-2">2. The Bridging Question</a> &bull; 
    <a href="#step-3">3. The Exact Math &amp; Formula</a> &bull; 
    <a href="#step-4">4. Where Did It Come From?</a> &bull; 
    <a href="#step-5">5. Concrete Toy Example</a> &bull; 
    <a href="#step-6">6. Core Takeaway</a>
  </p>
</nav>

---

<h2 id="step-1">Step 1: 3-Year-Old Intuition</h2>

Imagine holding your parents' hands and walking into a colossal **magical library**.

This library holds millions of wondrous storybooks. Today, your teacher gave you an exciting treasure hunt mission: **"Find out which book tells the story of the flying fire-breathing dragon!"**

Without magic, if you wanted to find the answer, you would have to pull every single book off the shelf, one by one, and read from page 1 to page 500. You would collapse from exhaustion after days and nights without even scratching the surface.

So, the wise head librarian invented an ingenious **"Three Magic Cards"** treasure-hunt game:

<figure>
<pre>
   [Your Wish Card: Query] ──────► "I want to read about a flying fire dragon!"
                                          │
                                          ▼ (Hold your card against the shelves)
                                  ┌───────┴───────┐
                                  ▼               ▼
     [Shelf Exterior Label: Key 1] "Cake Baking"   [Shelf Exterior Label: Key 2] "Ancient Wyverns"
             Match Score: 0% ✘                             Match Score: 95% ✔
                                                                 │
                                                                 ▼ (High match! Pull and open book)
                                                   [Book Internal Story: Value 2]
                                                   "Once upon a time, a red wyvern breathed fire..."
</pre>
<figcaption><strong>Figure 5.1:</strong> The Library Clue Hunt with three cards. You hold a Wish Card (Query), match it against Shelf Labels (Keys), and only crack open the books with top scores to read their actual stories (Values).</figcaption>
</figure>

### 1. The First Card: The Wish Card (Query, or Q &mdash; "What am I looking for?")
- This is the slip of paper you clutch tightly when entering the library.
- It encapsulates **your current burning question**. For instance: *"Who can tell me something about breathing fire?"*

### 2. The Second Card: The Shelf Label Card (Key, or K &mdash; "What am I about?")
- Every book on every shelf has a short **index label pasted on its outer spine**.
- The label does not contain the whole 500-page story; it merely **advertises its topic to passers-by**: e.g., *"Baking"*, *"Gardening"*, *"Ancient Dragons"*, *"Astronomy"*.
- Its sole mission in life is to bump against your Wish Card (Query) to see **how well the two of you match**.

### 3. The Third Card: The Treasure Story Itself (Value, or V &mdash; "My actual content")
- When you discover that a book's label card (Key) matches your wish card (Query) with flying colors, you excitedly pull the book off the shelf and open its pages.
- The dense, rich **knowledge, narrative, and information inside the book** is the treasure story itself (Value).

---

### Why Can't the Shelf Label (Key) and the Story (Value) Be the Same Thing?

Imagine if the librarian were lazy and taped the entire 500-page book onto the outer spine. The shelves would be an unreadable, cluttered mess!
More importantly: **"indexing clues" and "payload content" are fundamentally different in purpose**.
- When searching for a cure for a cold, the query keyword matches an index label (Key = *"Fever Medicine"*);
- But what you actually want to extract and swallow is the chemical formula inside (Value = *"Acetaminophen 500mg"*).

Inside a modern Large Language Model (the Transformer brain):
- Every single word (Token) acts simultaneously as a **searcher** (emitting a Query) and a **treasured book** (offering a Key for others to inspect, and offering a Value for others to absorb).
- **These are the legendary three musketeers of Attention: Query, Key, and Value!**

---

<h2 id="step-2">Step 2: The Bridging Question</h2>

In Chapters 01 through 04, we saw how words turn into static embedding vectors $\mathbf{x} \in \mathbb{R}^d$ and pass through feed-forward networks for spatial transformations and non-linear gating. In our hands-on workshop (Lab 01), we even trained our very first neural brain in 80 lines of pure Python based on Bengio (2003).

Yet, our lab experiment brutally exposed a **fatal structural flaw**:
In traditional feed-forward architectures, each word's vector representation at the input stage is **completely rigid, isolated, and context-blind**!

Consider this classic human language ambiguity:
> *"He sat on the muddy **bank** of the river, while the investment **bank** approved his loan."*

<figure>
<pre>
   Static Embedding Table (Lookup Matrix E)
   ┌─────────┬──────────────────────┐
   │ Token   │ Static Embedding x   │
   ├─────────┼──────────────────────┤
   │ "bank"  │ [ 0.82, -0.41, 0.15 ] │ ◄── Identical static vector regardless of context!
   └─────────┴──────────────────────┘

   Token #1 "bank" ──► Surrounded by "river", "muddy", "water" ──► True meaning is [Hydrological Shore]!
   Token #2 "bank" ──► Surrounded by "investment", "loan", "cash" ──► True meaning is [Financial Institution]!
</pre>
<figcaption><strong>Figure 5.2:</strong> The "Polysemy Amnesia" of static embeddings. Without dynamic contextual retrieval, the identical word token receives the exact same starting vector in every sentence.</figcaption>
</figure>

Computers face three pressing bridging questions:
1. **The Contextual Need**: Token #1 (`"bank"`) must have the active capability to **look around** and inquire of neighboring words: *"Hey! Who among you is related to rivers or water? Lend me your information so I can update who I am!"*
2. **The Computational Translation**: How do we convert the physical game of *"comparing wish cards against shelf labels to extract stories"* into lightning-fast matrix multiplications on GPUs?
3. **The Projection Dilemma**: Why can't a word simply use its raw input vector $\mathbf{x}$ to take dot products with other raw vectors? Why must we invent **three completely separate linear matrices** ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$) to project the vector into three distinct subspaces?

---

<h2 id="step-3">Step 3: The Exact Math &amp; Formula</h2>

---

### 1. Mathematical Definitions

Given an input text sequence of $T$ tokens, its embedded representation forms an input feature matrix:

$$
\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

Where:
- $T$ is the sequence length (number of tokens in the context window, e.g., 2048 or 4096);
- $d_{\text{model}}$ is the model's hidden dimension (e.g., $d_{\text{model}} = 4096$ in LLaMA-3 8B, or $12288$ in GPT-3 175B).

To grant the model the ability to observe tokens under different specialized roles, the Transformer introduces **three independent, learnable linear projection parameter matrices**:

$$
\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

Through three parallel matrix multiplications, the raw input matrix $\mathbf{X}$ is projected into three new geometric spaces:

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{T \times d_k} \quad (\text{Query Matrix})
$$

$$
\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{T \times d_k} \quad (\text{Key Matrix})
$$

$$
\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{T \times d_v} \quad (\text{Value Matrix})
$$

In standard single-head attention setups, one typically sets $d_k = d_v = d_{\text{model}}$. In modern Multi-Head Attention with $h$ attention heads, the dimensional budget is partitioned evenly:

$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

For example, in LLaMA-3 8B ($d_{\text{model}} = 4096, h = 32$), each head operates in an attention subspace of dimension $d_k = \frac{4096}{32} = 128$.

---

### 2. Single-Token Row-Vector Decomposition

The macro matrix product is simply the batch packing of individual token vector transformations. For token $i$ represented by row vector $\mathbf{x}_i^\top \in \mathbb{R}^{1 \times d_{\text{model}}}$:

$$
\mathbf{q}_i^\top = \mathbf{x}_i^\top \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k} \quad (\text{Query Vector for Token } i)
$$

$$
\mathbf{k}_i^\top = \mathbf{x}_i^\top \mathbf{W}_K \in \mathbb{R}^{1 \times d_k} \quad (\text{Key Vector for Token } i)
$$

$$
\mathbf{v}_i^\top = \mathbf{x}_i^\top \mathbf{W}_V \in \mathbb{R}^{1 \times d_v} \quad (\text{Value Vector for Token } i)
$$

<figure>
<pre>
                      Raw Input Token Vector x_i  [1 × d_model]
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   W_Q [d_model × d_k]       W_K [d_model × d_k]       W_V [d_model × d_v]
         │                         │                         │
         ▼                         ▼                         ▼
   Query Vector q_i [1 × d_k] Key Vector k_i [1 × d_k]  Value Vector v_i [1 × d_v]
   "What am I seeking?"       "What tags do I offer?"    "What payload do I deliver?"
</pre>
<figcaption><strong>Figure 5.3:</strong> A single input vector branches simultaneously into three distinct specialized roles via three learnable projection matrices.</figcaption>
</figure>

Each vector fulfills an exact mathematical function:
- <dfn id="def-query-vector"><strong>Query Vector $\mathbf{q}_i$</strong></dfn>: The **retrieval probe** emitted by token $i$. It asks: *"To disambiguate my meaning in this sentence, what specific syntactic or semantic clues do I need?"*
- <dfn id="def-key-vector"><strong>Key Vector $\mathbf{k}_j$</strong></dfn>: The **public advertising card** displayed by token $j$. It advertises: *"I possess these grammatical roles, semantic features, and entity tags; compare your query against me!"*
- <dfn id="def-value-vector"><strong>Value Vector $\mathbf{v}_j$</strong></dfn>: The **information payload** of token $j$. It declares: *"If you find my key relevant, this is the actual knowledge vector I contribute to your updated representation."*

---

### 3. First-Principles Deep Dive: Why Three Separate Matrices? (What Breaks if $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$?)

A natural question arises for anyone first encountering the Transformer:
> *"If each word already has a feature vector $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$, why can't tokens just take dot products directly with each other (i.e., set $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$)? Why spend $3\times$ the parameter budget learning $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$?"*

Let us trace the **three fatal mathematical disasters** that occur if one forces $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$:

#### Disaster 1: The Symmetry Trap
Vector dot products are inherently commutative:

$$
\mathbf{x}_i \cdot \mathbf{x}_j = \mathbf{x}_j \cdot \mathbf{x}_i
$$

If raw embeddings interact directly, **the attention score of Token $i$ towards Token $j$ would be 100% strictly identical to the attention score of Token $j$ towards Token $i$**!

Natural language relationships, however, are deeply **asymmetric and directed**:
- In the clause *"The kitten (Token $A$) chased (Token $B$) the mouse (Token $C$)"*:
  - The transitive verb *"chased"* desperately needs to attend to its subject (*"kitten"*) and direct object (*"mouse"*) to know who did what;
  - But the noun *"kitten"* does not need to allocate 90% of its attention back to *"chased"* just to establish that it is a feline animal!
- In an adjective-noun phrase like *"extremely beautiful"*:
  - The adjective *"beautiful"* must attend to *"extremely"* to gauge its intensity;
  - But the adverb *"extremely"* is an auxiliary modifier that does not need to duplicate the full physical noun entity.

By introducing separate $\mathbf{W}_Q$ and $\mathbf{W}_K$, the attention score becomes:

$$
\text{Score}(i \to j) = \mathbf{q}_i^\top \mathbf{k}_j = (\mathbf{x}_i^\top \mathbf{W}_Q)(\mathbf{x}_j^\top \mathbf{W}_K)^\top = \mathbf{x}_i^\top (\mathbf{W}_Q \mathbf{W}_K^\top) \mathbf{x}_j
$$

Since the matrix product $\mathbf{W}_Q \mathbf{W}_K^\top$ is generally **non-symmetric** ($\mathbf{W}_Q \mathbf{W}_K^\top \ne \mathbf{W}_K \mathbf{W}_Q^\top$), this **destroys the symmetry trap**, allowing the model to freely learn asymmetric directed relationships!

#### Disaster 2: The Self-Absorption Bias
In linear algebra, any non-zero vector's dot product with itself equals the square of its Euclidean length:

$$
\mathbf{x}_i \cdot \mathbf{x}_i = \|\mathbf{x}_i\|^2 = \sum_{m=1}^d x_{im}^2 > 0
$$

By the Cauchy-Schwarz inequality:

$$
|\mathbf{x}_i \cdot \mathbf{x}_j| \le \|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|
$$

If all word vectors have roughly comparable norms, **a word's dot product with itself is almost guaranteed to be larger than its dot product with any other word in the dictionary**:
- Without $\mathbf{W}_Q$ and $\mathbf{W}_K$, self-similarity $\mathbf{x}_i \cdot \mathbf{x}_i$ utterly overpowers surrounding tokens;
- After Softmax normalization, every token assigns 99% of its attention budget to itself, becoming a context-blind narcissist that ignores the rest of the sentence;
- With distinct projection matrices, $\mathbf{q}_i^\top \mathbf{k}_i = \mathbf{x}_i^\top (\mathbf{W}_Q \mathbf{W}_K^\top) \mathbf{x}_i$. As long as $\mathbf{W}_Q \mathbf{W}_K^\top$ is not a positive-definite identity matrix, a token's self-score can easily be lower than its score with informative context words, forcing the network to look outward.

#### Disaster 3: Addressing vs. Payload Conflict
In computer architecture and database design, it is a timeless principle: **a memory address (Key) is not memory content (Value)**.
- **Addressing Space (Query & Key)**: Specialized in routing, tags, and relational predicates (e.g., *"needs a singular verb"*, *"is a temporal adverb"*). These are control-flow signals;
- **Payload Space (Value)**: Specialized in semantic substance and factual knowledge (e.g., *"small furry quadruped, weighs 4kg, is sleeping"*).
- Forcing $\mathbf{V} = \mathbf{X}$ corrupts representation learning by tangling routing metadata directly into factual payload vectors across successive layers.
- An independent $\mathbf{W}_V$ gives the model complete freedom: **even if two tokens have identical key match scores, the model can choose to deliver completely different semantic payloads**.

---

### 4. Mathematical Symbol & Dimension Catalog

<details name="ch05-specs" open>
<summary><strong>Click to expand: Q, K, V Mathematical Symbols &amp; Dimensions</strong></summary>

<dl>
  <dt><strong>$\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$</strong></dt>
  <dd>Input feature matrix after embedding and positional encoding, representing $T$ tokens of dimension $d_{\text{model}}$.</dd>

  <dt><strong>$T \in \mathbb{N}^+$</strong></dt>
  <dd>Sequence length (context window size), e.g., 2048, 4096, or 8192.</dd>

  <dt><strong>$d_{\text{model}} \in \mathbb{N}^+$</strong></dt>
  <dd>The model's residual stream hidden dimension (e.g., 4096).</dd>

  <dt><strong>$d_k \in \mathbb{N}^+$</strong></dt>
  <dd>Subspace dimension for Queries and Keys. To compute dot products $\mathbf{q}_i^\top \mathbf{k}_j$, $\mathbf{Q}$ and $\mathbf{K}$ must have identical column dimensions.</dd>

  <dt><strong>$d_v \in \mathbb{N}^+$</strong></dt>
  <dd>Subspace dimension for Values. While theoretically $d_v$ can differ from $d_k$, modern engineering uniformly adopts $d_v = d_k$.</dd>

  <dt><strong>$\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  <dd>Query projection weight matrix.</dd>

  <dt><strong>$\mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  <dd>Key projection weight matrix.</dd>

  <dt><strong>$\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$</strong></dt>
  <dd>Value projection weight matrix.</dd>

  <dt><strong>$\mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{T \times T}$</strong></dt>
  <dd>Raw attention affinity matrix before scaling and Softmax. Element $(i, j)$ represents the unnormalized affinity of token $i$ for token $j$.</dd>
</dl>

</details>

---

<h2 id="step-4">Step 4: Where Did It Come From?</h2>

How did computer science transition from rigid lookup tables to the differentiable Q, K, V attention mechanism powering modern AI?

<dl>
  <dt><time datetime="1970">1970</time> &mdash; <strong>Edgar F. Codd</strong>: Relational Databases and Key-Value Lookups</dt>
  <dd>
    Turing Award laureate Codd formalized database theory. Classic database retrieval is <strong>hard and discrete</strong>: given a query (<code>SELECT * WHERE Key = 'Bank'</code>), a hash table or B-tree looks up the exact single matching value.
    <br>
    <strong>Why it couldn't work in neural networks:</strong> Discrete exact matching (<code>if Key == Query</code>) has a derivative of zero almost everywhere. Gradients cannot flow backwards!
  </dd>

  <dt><time datetime="2014">2014</time> &mdash; <strong>Dzmitry Bahdanau, Kyunghyun Cho &amp; Yoshua Bengio</strong>: The Birth of Soft Attention</dt>
  <dd>
    To fix the information bottleneck of recurrent neural networks (RNNs) in machine translation, Bengio's group introduced <strong>differentiable soft addressing</strong>: rather than picking one discrete word, the model computes a continuous probability distribution across all source tokens.
    <br>
    <strong>Its limitations:</strong> Decoder states acted as queries while encoder states served as keys and values simultaneously. Role differentiation had not yet emerged, and processing remained trapped in slow sequential recurrence.
    <cite>"Neural Machine Translation by Jointly Learning to Align and Translate", ICLR 2015</cite>.
  </dd>

  <dt><time datetime="2015">2015</time> &mdash; <strong>Minh-Thang Luong, Hieu Pham &amp; Christopher D. Manning</strong>: Dot-Product Attention</dt>
  <dd>
    Manning's team at Stanford replaced Bahdanau's expensive multi-layer perceptron scoring with fast vector dot products: $\text{Score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^\top \mathbf{h}$, laying the groundwork for massive GPU parallelization.
  </dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Ashish Vaswani et al. (Google Brain &amp; Google Research)</strong>: Q, K, V Self-Attention</dt>
  <dd>
    In the landmark paper <em>"Attention Is All You Need"</em>, the authors discarded RNNs and CNNs entirely. Borrowing the Query-Key-Value concept from database systems, they formalized <strong>linear projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$</strong> for self-attention.
    <br>
    <strong>Why pure linear projections instead of non-linear MLPs?</strong>
    Linear matrix multiplications achieve blistering throughput on GPU Tensor Cores ($O(T \cdot d^2)$ with hundreds of TFLOPs). Non-linearity is supplied downstream by the row-wise Softmax and the channel-wise SwiGLU / FFN, achieving maximum computational efficiency and mathematical elegance.
    <cite>"Attention Is All You Need", NeurIPS 2017</cite>.
  </dd>
</dl>

---

<h2 id="step-5">Step 5: Concrete Toy Example</h2>

To ensure you can verify every single calculation by hand, let us walk through a **minimal toy example with small integers**.

### Setup
- A 3-token sentence:
  - $\text{Token}_1 =$ <kbd>"The"</kbd>
  - $\text{Token}_2 =$ <kbd>"river"</kbd>
  - $\text{Token}_3 =$ <kbd>"bank"</kbd>
- Hidden dimension: $d_{\text{model}} = 4$;
- Attention subspace dimension: $d_k = 2, d_v = 2$.

---

### 1. Input Feature Matrix $\mathbf{X} \in \mathbb{R}^{3 \times 4}$

Assume the 4-dimensional embedding row vectors for the 3 tokens are:

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1^\top \\
\mathbf{x}_2^\top \\
\mathbf{x}_3^\top
\end{bmatrix} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 2.0 & 0.0 & 1.0 \\
1.0 & 1.0 & 0.0 & 2.0
\end{bmatrix}
$$

Where:
- $\mathbf{x}_1 = [1, 0, 1, 0]$ (the function word "The")
- $\mathbf{x}_2 = [0, 2, 0, 1]$ (the hydrological word "river")
- $\mathbf{x}_3 = [1, 1, 0, 2]$ (the ambiguous word "bank")

---

### 2. The Three Projection Matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{4 \times 2}$

We choose clean integer weights for transparent manual arithmetic:

$$
\mathbf{W}_Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
\mathbf{W}_K = \begin{bmatrix}
0 & 1 \\
1 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
\mathbf{W}_V = \begin{bmatrix}
1 & 0 \\
0 & 0 \\
0 & 2 \\
0 & 1
\end{bmatrix}
$$

---

### 3. Step-by-Step Manual Projections

#### Query Matrix $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{3 \times 2}$

For Row 1: $\mathbf{q}_1^\top = [1, 0, 1, 0] \mathbf{W}_Q$:
- Col 1: $1\times 1 + 0\times 0 + 1\times 0 + 0\times 1 = 1$
- Col 2: $1\times 0 + 0\times 1 + 1\times 1 + 0\times 0 = 1$
- $\mathbf{q}_1^\top = [1, 1]$

For Row 2: $\mathbf{q}_2^\top = [0, 2, 0, 1] \mathbf{W}_Q$:
- Col 1: $0\times 1 + 2\times 0 + 0\times 0 + 1\times 1 = 1$
- Col 2: $0\times 0 + 2\times 1 + 0\times 1 + 1\times 0 = 2$
- $\mathbf{q}_2^\top = [1, 2]$

For Row 3: $\mathbf{q}_3^\top = [1, 1, 0, 2] \mathbf{W}_Q$ (Query for <kbd>"bank"</kbd>):
- Col 1: $1\times 1 + 1\times 0 + 0\times 0 + 2\times 1 = 1 + 2 = 3$
- Col 2: $1\times 0 + 1\times 1 + 0\times 1 + 2\times 0 = 1$
- $\mathbf{q}_3^\top = [3, 1]$

$$
\mathbf{Q} = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
3 & 1
\end{bmatrix}
$$

---

#### Key Matrix $\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{3 \times 2}$

For Row 1: $\mathbf{k}_1^\top = [1, 0, 1, 0] \mathbf{W}_K$:
- Col 1: $1\times 0 + 0\times 1 + 1\times 1 + 0\times 0 = 1$
- Col 2: $1\times 1 + 0\times 0 + 1\times 0 + 0\times 1 = 1$
- $\mathbf{k}_1^\top = [1, 1]$

For Row 2: $\mathbf{k}_2^\top = [0, 2, 0, 1] \mathbf{W}_K$ (Key for <kbd>"river"</kbd>):
- Col 1: $0\times 0 + 2\times 1 + 0\times 1 + 1\times 0 = 2$
- Col 2: $0\times 1 + 2\times 0 + 0\times 0 + 1\times 1 = 1$
- $\mathbf{k}_2^\top = [2, 1]$

For Row 3: $\mathbf{k}_3^\top = [1, 1, 0, 2] \mathbf{W}_K$:
- Col 1: $1\times 0 + 1\times 1 + 0\times 1 + 2\times 0 = 1$
- Col 2: $1\times 1 + 1\times 0 + 0\times 0 + 2\times 1 = 1 + 2 = 3$
- $\mathbf{k}_3^\top = [1, 3]$

$$
\mathbf{K} = \begin{bmatrix}
1 & 1 \\
2 & 1 \\
1 & 3
\end{bmatrix}
$$

---

#### Value Matrix $\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{3 \times 2}$

For Row 1: $\mathbf{v}_1^\top = [1, 0, 1, 0] \mathbf{W}_V = [1\times 1 + 0, 0 + 1\times 2] = [1, 2]$

For Row 2: $\mathbf{v}_2^\top = [0, 2, 0, 1] \mathbf{W}_V = [0, 1\times 1] = [0, 1]$

For Row 3: $\mathbf{v}_3^\top = [1, 1, 0, 2] \mathbf{W}_V = [1\times 1, 2\times 1] = [1, 2]$

$$
\mathbf{V} = \begin{bmatrix}
1 & 2 \\
0 & 1 \\
1 & 2
\end{bmatrix}
$$

---

### 4. Raw Attention Scores $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{3 \times 3}$

Now, let us calculate the dot products of each Query with each Key:

$$
\mathbf{S} = \mathbf{Q} \mathbf{K}^\top = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
3 & 1
\end{bmatrix} \begin{bmatrix}
1 & 2 & 1 \\
1 & 1 & 3
\end{bmatrix}
$$

Computing each row:
- Row 1 (Query "The"):
  - $S_{11} = 1\times 1 + 1\times 1 = 2$
  - $S_{12} = 1\times 2 + 1\times 1 = 3$
  - $S_{13} = 1\times 1 + 1\times 3 = 4$
- Row 2 (Query "river"):
  - $S_{21} = 1\times 1 + 2\times 1 = 3$
  - $S_{22} = 1\times 2 + 2\times 1 = 4$
  - $S_{23} = 1\times 1 + 2\times 3 = 7$
- Row 3 (Query <mark>"bank"</mark>):
  - $S_{31} = \mathbf{q}_3 \cdot \mathbf{k}_1 = 3\times 1 + 1\times 1 = 4$ (match with "The")
  - $S_{32} = \mathbf{q}_3 \cdot \mathbf{k}_2 = 3\times 2 + 1\times 1 = \mathbf{7}$ (match with <mark>"river"</mark>!)
  - $S_{33} = \mathbf{q}_3 \cdot \mathbf{k}_3 = 3\times 1 + 1\times 3 = 6$ (self-attention score)

$$
\mathbf{S} = \begin{bmatrix}
2 & 3 & 4 \\
3 & 4 & 7 \\
4 & \mathbf{7} & 6
\end{bmatrix}
$$

---

### 5. Physical Verification of Results

Notice Row 3 for the ambiguous token `"bank"`:
- Score with `"The"`: $4$
- Score with itself `"bank"`: $6$
- Score with `"river"`: **$7$**!

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 5.1:</strong> Raw attention match scores computed by ambiguous token "bank"</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">Target Token (Key)</th>
      <th scope="col" align="center" width="25%">Dot Product $\mathbf{q}_3 \cdot \mathbf{k}_j$</th>
      <th scope="col" align="center" width="20%">Raw Affinity Score</th>
      <th scope="col" align="left" width="35%">Semantic Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><kbd>"The"</kbd></th>
      <td align="center">$3\times 1 + 1\times 1 = 4$</td>
      <td align="center">4.0</td>
      <td>Syntactic article; provides minimal contextual disambiguation</td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"river"</kbd></th>
      <td align="center">$3\times 2 + 1\times 1 = \mathbf{7}$</td>
      <td align="center"><mark><strong>7.0</strong></mark></td>
      <td><strong>Core clue word! Successfully identifies hydrological shore context!</strong></td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"bank"</kbd></th>
      <td align="center">$3\times 1 + 1\times 3 = 6$</td>
      <td align="center">6.0</td>
      <td>Self-reference; preserves baseline lexical identity</td>
    </tr>
  </tbody>
</table>

Visualizing match intensity:
- Attention towards `"river"`:
  <meter min="0" max="10" low="3" high="6" optimum="8" value="7.0">7.0 / 10</meter> (Highest weight; incorporates hydrological meaning)
- Attention towards itself:
  <meter min="0" max="10" low="3" high="6" optimum="8" value="6.0">6.0 / 10</meter> (Second highest)
- Attention towards `"The"`:
  <meter min="0" max="10" low="3" high="6" optimum="8" value="4.0">4.0 / 10</meter> (Lowest weight)

Through the collaborative projection of $\mathbf{W}_Q$ and $\mathbf{W}_K$, `"bank"` successfully escaped the self-absorption trap and directed its highest attention score towards the pivotal clue word `"river"`.

In the upcoming Chapter 06, we will discover how **Softmax** converts these raw logits $[4, 7, 6]$ into strict probability weights summing to 100%, which then aggregate the **Value vectors**.

---

<h2 id="step-6">Step 6: Core Takeaway</h2>

<fieldset>
<legend><strong>Core Memory Card</strong></legend>
<p>
<strong>Query, Key, and Value represent the three specialized personalities that transform static word embeddings into dynamic contextual intelligence:</strong><br>
Raw embeddings $\mathbf{X}$ are static and isolated; by projecting them through three distinct linear matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$, every token gains a <strong>probe to seek clues (Query)</strong>, a <strong>tag to advertise identity (Key)</strong>, and a <strong>payload to deliver knowledge (Value)</strong>. This projection breaks the algebraic curse of dot-product symmetry and self-absorption, laying the bedrock for context-aware language comprehension.
</p>
</fieldset>

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../04b-lab-micro-brain/index.html">&larr; Hands-on Lab 01: Training Your First Brain in 80 Lines of Pure Python</a> &bull;
    <a href="../index.html">Course Overview</a> &bull;
    <a href="../06-softmax-function/index.html">Chapter 06: The Fair Voting Booth (The Softmax Function) &rarr;</a>
  </p>
</nav>
