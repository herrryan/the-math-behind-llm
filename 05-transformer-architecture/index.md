# Chapter 05: The Transformer Blueprint (The Bird's-Eye View & The Round Table)

<nav aria-label="Table of Contents">
  <p>
    <strong>Table of Contents:</strong> 
    <a href="#step-1">1. 3-Year-Old Intuition</a> &bull; 
    <a href="#step-2">2. The Bridging Question</a> &bull; 
    <a href="#step-3">3. The Exact Math &amp; Architecture</a> &bull; 
    <a href="#step-4">4. Where Did It Come From?</a> &bull; 
    <a href="#step-5">5. Concrete Toy Example</a> &bull; 
    <a href="#step-6">6. Core Takeaway</a>
  </p>
</nav>

---

<h2 id="step-1">Step 1: 3-Year-Old Intuition (The Round Table vs. The Whispering Line)</h2>

In the previous hands-on lab (Lab 01), we built and trained our very first neural network brain in 80 lines of pure Python. It learned grammar and generated its first sentences! But we also discovered its fatal flaw: **severe short-term amnesia**. It could only look at the single preceding word, blindly looping on <samp>"the rug the rug"</samp> forever.

To understand why modern AI solved this problem, let's look at three very different ways children can tell a story:

<figure>
<pre>
1. The Horse with Blinkers (Bengio 2003 MLP):
   [Word 1] [Word 2] ... [Word 98] | [Word 99] ===&gt; [Predict Word 100]
                                   └─ Blind to anything further back!

2. The Whispering Telephone Game (Recurrent Neural Networks / RNNs):
   [Child 1] ──whisper──&gt; [Child 2] ──whisper──&gt; ... ──whisper──&gt; [Child 100]
   "A puppy..."          "A puppy?"                              "A potato!"
   (Message blurs, dilutes, and gets lost along the long chain)

3. The Grand Round Table (The Transformer):
   ┌─────────────────────────────────────────────────────────────┐
   │                          Child 1                            │
   │                        ("The puppy")                        │
   │                          ▲         ▲                        │
   │       direct eye contact │         │ direct eye contact     │
   │                          ▼         ▼                        │
   │   Child 25 ◄───────────► Child 50 ◄───────────► Child 100   │
   │   ("fluffy")             ("barked")            ("loudly")   │
   └─────────────────────────────────────────────────────────────┘
   Every child sits at the round table simultaneously!
   Anyone can look directly at anyone else across the room in zero seconds.
</pre>
<figcaption><strong>Figure 5.1:</strong> Three generations of sequence processing: the blind fixed window, the whispering telephone line, and the round-table conference.</figcaption>
</figure>

### Metaphor 1: The Horse with Blinkers (Fixed-Window Feed-Forward Networks)
In our Lab 01 network (based on Bengio 2003), the model wore leather blinkers on both sides of its eyes. It could only see the word directly in front of its nose. 

If we wanted it to look at 10 past words, we had to concatenate 10 vectors together, making the first weight matrix 10 times wider. If we wanted it to read an entire book with 100,000 words, the weight matrix would require trillions of numbers, consuming petabytes of memory before doing a single calculation! Fixed windows simply cannot scale to long stories.

### Metaphor 2: The Whispering Telephone Game (Recurrent Networks / RNNs &amp; LSTMs)
Before 2017, the AI world tried to fix this with **Recurrent Neural Networks (<abbr title="Recurrent Neural Network">RNN</abbr>)** and **Long Short-Term Memory (<abbr title="Long Short-Term Memory">LSTM</abbr>)**.

Imagine 100 children standing in a single-file line.
- Child 1 hears a secret: <samp>"A tiny golden puppy with fluffy ears lived in Paris."</samp>
- Child 1 writes a hurried note in a tiny notebook (the hidden state vector $\mathbf{h}_1$) and passes it to Child 2.
- Child 2 reads the note, adds their own word, rewrites the notebook, and whispers to Child 3.
- By the time the notebook reaches Child 100 at the end of the line, the page is smeared with eraser marks and ink stains. What does Child 100 hear? <samp>"A potato lived in Paris."</samp>

This breakdown suffered from two catastrophic structural flaws:
1. **The Information Bottleneck**: Squeezing an entire unfolding novel into a single fixed-size vector $\mathbf{h}_t$ causes distant memories to dissolve and vanish.
2. **The Sequential Prison**: Child 50 **cannot start whispering** until Child 49 finishes! Modern graphics cards (<abbr title="Graphics Processing Unit">GPU</abbr>s) have tens of thousands of tiny computational cores ready to work simultaneously, but an RNN forced them to sit idle, waiting for one word to process at a time.

### Metaphor 3: The Grand Round Table (The Transformer)
In 2017, a team of researchers at Google published a legendary paper with a bold title: <cite>"Attention Is All You Need"</cite>.

They threw away the whispering line completely. Instead, they placed all the words around a **giant round conference table**:
- **Simultaneous Arrival**: All 1,000 words of a document walk into the room at the exact same instant.
- **Direct Eye Contact (Self-Attention)**: If word 800 (<kbd>"it"</kbd>) needs to know what noun it refers to, it does not wait for a whisper chain. It glances directly across the table at word 12 (<kbd>"puppy"</kbd>) in a single step!
- **Parallel Speed**: Because everyone is at the table at once, the GPU can compute relationships between all words simultaneously in parallel.

---

<h2 id="step-2">Step 2: The Bridging Question</h2>

<fieldset>
<legend><strong>The Computational Challenge</strong></legend>
<p>
How do we convert a round-table conference into concrete linear algebra?
</p>
<p>
If all 1,000 words sit at the table simultaneously, what keeps the room from devolving into deafening chaos? How does each word know <em>who to listen to</em>, <em>what to absorb</em>, and <em>how to think about it privately</em> before speaking the next word?
</p>
</fieldset>

To turn this intuitive round table into a working machine, modern large language models use an elegant alternating cycle:
1. **Communication Phase (Self-Attention)**: Words look around the table, exchange notes with other words, and update their context.
2. **Thinking Phase (Feed-Forward Network)**: Each word retreats into its private study, evaluates what it just learned, queries its long-term factual memory, and refines its meaning.

Let's inspect the complete mathematical blueprint of this architecture.

---

<h2 id="step-3">Step 3: The Exact Math &amp; Architecture</h2>

Modern frontier language models &mdash; including **GPT-4**, **LLaMA-3**, **Mistral**, **Gemma**, **Claude**, **DeepSeek**, and **Qwen** &mdash; are built upon the **Decoder-Only Autoregressive Transformer** architecture.

The entire brain is a sequence of transformations mapping raw token IDs to probability distributions over the next token:

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                        THE TRANSFORMER PIPELINE                        │
└────────────────────────────────────────────────────────────────────────┘

 [Input Text]: "The cat sat on the"
       │
       ▼ (Tokenizer)
 [Token IDs]: [464, 3797, 3334, 319, 262]  (Sequence Length T = 5)
       │
       ▼ (Embedding Matrix E + Positional Info P)
 [Input Tensor X_0]  ───► Shape: [T × d_model]
       │
       ├───────────────────────────────────────────────────────┐
       ▼                                                       │
 ┌─────────────────────────────────────────────────────────┐   │
 │               TRANSFORMER BLOCK (Layer 1)               │   │
 │                                                         │   │
 │   X_in ──► [RMSNorm] ──► [Causal Self-Attention] ──┐    │   │
 │     │                                              │    │   │
 │     └──────────────── (Residual Add) ◄─────────────┘    │   │
 │                            │                            │   │
 │                            ▼ H_1                        │   │
 │     H_1 ──► [RMSNorm] ──► [SwiGLU Feed-Forward] ───┐    │   │
 │     │                                              │    │   │
 │     └──────────────── (Residual Add) ◄─────────────┘    │   │
 │                            │                            │   │
 └────────────────────────────┼────────────────────────────┘   │
                              ▼ X_1                            │ Repeated
                              │                                │ for L
                             ... (Repeated across L layers)    │ Layers!
                              │                                │
 ┌────────────────────────────┼────────────────────────────┐   │
 │               TRANSFORMER BLOCK (Layer L)               │   │
 │                            │                            │   │
 └────────────────────────────┼────────────────────────────┘   │
                              ▼ X_L ◄──────────────────────────┘
                              │
                       [Final RMSNorm]
                              │
                              ▼ X_final ───► Shape: [T × d_model]
                              │
             [Output Unembedding Matrix E_U] ──► Shape: [d_model × |V|]
                              │
                              ▼
                       [Logits Matrix Z]  ───► Shape: [T × |V|]
                              │
                      [Softmax Function]
                              │
                              ▼
                 [Probability Distribution P]
             P("rug" | "The cat sat on the") = 78.4%
</pre>
<figcaption><strong>Figure 5.2:</strong> Complete macro-architecture of a modern decoder-only Transformer. Raw text flows from bottom tokens through an embedding lookup, traverses L stacked blocks of alternating Communication (Attention) and Thinking (FFN), and terminates at vocabulary logits.</figcaption>
</figure>

Let's dissect each mathematical component in this pipeline from first principles.

### 1. The Architectural Hyperparameters

Every Transformer architecture is fully defined by five foundational scalar dimensions:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Symbol</th>
      <th align="left">Name</th>
      <th align="left">Concrete LLM Example (LLaMA-3-8B)</th>
      <th align="left">Physical Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$|V|$</td>
      <td>Vocabulary Size</td>
      <td>$128,256$</td>
      <td>Total unique words/subwords/tokens the model knows.</td>
    </tr>
    <tr>
      <td>$T$</td>
      <td>Context Length</td>
      <td>$8,192$ (up to $128\text{k}$)</td>
      <td>Number of tokens sitting at the round table simultaneously.</td>
    </tr>
    <tr>
      <td>$d_{\text{model}}$</td>
      <td>Model Dimension</td>
      <td>$4,096$</td>
      <td>Width of the highway vector carrying meaning for each word.</td>
    </tr>
    <tr>
      <td>$L$</td>
      <td>Number of Layers</td>
      <td>$32$</td>
      <td>Number of Transformer blocks stacked vertically.</td>
    </tr>
    <tr>
      <td>$d_{\text{ffn}}$</td>
      <td>FFN Hidden Dim</td>
      <td>$14,336$ ($\approx \frac{8}{3} d_{\text{model}}$)</td>
      <td>Width of the private thinking chamber inside each block.</td>
    </tr>
  </tbody>
</table>

<fieldset>
<legend><strong>Deep Dive: What is BPE (Byte Pair Encoding) and How Does It Build the 128,256 Vocabulary?</strong></legend>
<p>
Learners frequently wonder: <em>The Oxford English Dictionary contains over 600,000 words, and with technical jargon and slang the vocabulary is unbounded. How can 128,256 items be enough? Does it cover Chinese and all other world languages?</em>
</p>
<p>
<strong>The answer: It does not just cover them &mdash; it covers 100% of all world languages and arbitrary text with mathematically zero out-of-vocabulary (<abbr title="Out Of Vocabulary">OOV</abbr>) errors!</strong>
</p>
<p>
The secret engine behind this is <strong>BPE (Byte Pair Encoding)</strong>.
</p>

<h4>1. The 3-Year-Old Intuition: Refrigerator Magnets & Superglue</h4>
<p>
Imagine you only have the 26 basic letter magnets on your refrigerator (<kbd>a</kbd>, <kbd>b</kbd>, <kbd>c</kbd> ... <kbd>z</kbd>).
</p>
<ul>
  <li><strong>Spelling letter by letter</strong>: Spelling <samp>"unbelievable"</samp> requires grabbing 12 individual magnets. A short paragraph fills up the entire fridge door, and your fingers tire quickly (<strong>sequences become too long, triggering quadratic $\mathcal{O}(T^2)$ self-attention compute blowup</strong>).</li>
  <li><strong>Molding every dictionary word as a giant single magnet</strong>: You would need millions of bulky magnets in your pockets, and the moment someone invents a new slang word, you have no magnet for it (<strong>vocabulary explosion and catastrophic out-of-vocabulary errors</strong>).</li>
</ul>
<p>
<strong>The BPE Superglue Rule:</strong> You observe which pairs of magnets sit next to each other most frequently. Whenever you notice two adjacent magnets consistently appearing together (like <kbd>t</kbd> and <kbd>h</kbd>), you dab a drop of superglue between them to create a permanent new tile: <kbd>th</kbd>. When you notice <kbd>th</kbd> and <kbd>e</kbd> constantly next to each other, you glue them into <kbd>the</kbd>! You continue gluing until you have exactly <strong>128,256</strong> versatile magnet tiles.
</p>

<h4>2. The BPE Algorithm & Concrete Toy Walkthrough</h4>
<p>
Suppose our training corpus has only 4 words with the following occurrence frequencies:
</p>
<ul>
  <li><samp>"low"</samp> (5 times) &rarr; Initial split: <code>l o w &lt;/w&gt;</code></li>
  <li><samp>"lower"</samp> (2 times) &rarr; Initial split: <code>l o w e r &lt;/w&gt;</code></li>
  <li><samp>"newest"</samp> (6 times) &rarr; Initial split: <code>n e w e s t &lt;/w&gt;</code></li>
  <li><samp>"widest"</samp> (3 times) &rarr; Initial split: <code>w i d e s t &lt;/w&gt;</code></li>
</ul>

<p><strong>Greedy Iterative Merging:</strong></p>
<ol>
  <li><strong>Merge Step 1</strong>: We count all adjacent symbol pairs in the corpus. The pair <code>(e, s)</code> appears in <samp>"newest"</samp> (6) and <samp>"widest"</samp> (3), totaling <strong>9 occurrences</strong> (the highest frequency!). We form Merge Rule 1: <code>e + s &rarr; es</code> and add <code>es</code> to our vocabulary.</li>
  <li><strong>Merge Step 2</strong>: Re-counting adjacent pairs reveals <code>(es, t)</code> also appears <strong>9 times</strong>. Merge Rule 2: <code>es + t &rarr; est</code>.</li>
  <li><strong>Merge Step 3</strong>: The pair <code>(est, &lt;/w&gt;)</code> appears <strong>9 times</strong>. Merge Rule 3: <code>est + &lt;/w&gt; &rarr; est&lt;/w&gt;</code>.</li>
  <li><strong>Merge Step 4</strong>: The pair <code>(l, o)</code> appears $5 + 2 = 7$ times, and <code>(o, w)</code> appears 7 times. We greedily merge: <code>l + o &rarr; lo</code>, followed by <code>lo + w &rarr; low</code>.</li>
</ol>

<p><strong>The Inference Moment on an Unseen Word:</strong></p>
<p>
Now, a user submits a brand-new word that <strong>never appeared anywhere in our training corpus: <samp>"lowest"</samp></strong>.
</p>
<ol>
  <li>Initial character split: <code>[l, o, w, e, s, t, &lt;/w&gt;]</code></li>
  <li>Apply learned merge rules sequentially: <code>e, s &rarr; es</code> &rarr; <code>es, t &rarr; est</code> &rarr; <code>est, &lt;/w&gt; &rarr; est&lt;/w&gt;</code> &rarr; <code>l, o &rarr; lo</code> &rarr; <code>lo, w &rarr; low</code></li>
  <li><strong>Final Token Output:</strong> <kbd>"low"</kbd> + <kbd>"est&lt;/w&gt;"</kbd> (represented cleanly with 2 existing subwords, <strong>Zero OOV</strong>!).</li>
</ol>

<h4>3. The 256 Raw Byte Safety Net (Byte-level Fallback)</h4>
<p>
In modern LLMs (GPT-4, LLaMA-3) using <strong>Byte-level BPE</strong>, the base vocabulary reserves all <strong>256 raw UTF-8 bytes (<code>0x00</code> to <code>0xFF</code>)</strong>. Even when facing extremely rare CJK ideographs, ancient runes, or unformatted binary data, the tokenizer falls back to raw bytes. It never crashes with an unknown symbol error.
</p>
</fieldset>

### 2. Stage 1: The Input Representation

Given an input prompt consisting of $T$ discrete integer token IDs:

$$
\mathbf{w} = \begin{bmatrix} w_1 & w_2 & \dots & w_T \end{bmatrix}^\top \in \{1, \dots, |V|\}^T
$$

<fieldset>
<legend><strong>Symbol Breakdown: $\mathbf{w}$ and $w_t$</strong></legend>
<ul>
  <li>$\mathbf{w}$: The discrete token ID sequence vector representing the whole prompt, with total length $T$.</li>
  <li>$w_t$ (or $w_i$): The <strong>specific integer token ID at position $t$ (or $i$)</strong> in the sequence. For example, in <samp>"The cat sat on the"</samp>, $w_1 = 464$ (for <kbd>"The"</kbd>) and $w_2 = 3797$ (for <kbd>"cat"</kbd>). Each integer $w_t$ lies in $\{1, 2, \dots, |V|\}$.</li>
  <li>$V = \{v_1, v_2, \dots, v_{|V|}\}$: The discrete vocabulary set containing all $|V|$ recognizable subwords.</li>
  <li>$|V|$: The vocabulary size (e.g., 128,256 in LLaMA-3).</li>
  <li>$T$: The sequence length (number of tokens sitting at the round table in this turn).</li>
</ul>
</fieldset>

<fieldset>
<legend><strong>Refresher: What is the Embedding Matrix E, and Why Does It Turn Words into Space?</strong></legend>
<p>
<strong>First-Principles Question: Why can't a computer compute directly on integer IDs?</strong>
</p>
<p>
If we treat <samp>"cat"</samp> as scalar $3797$, <samp>"dog"</samp> as scalar $3798$, and <samp>"apple"</samp> as scalar $1549$:
</p>
<ul>
  <li>Arithmetic produces nonsense: $3798 = 3797 + 1 \implies \text{dog} = \text{cat} + 1$;</li>
  <li>Distances are arbitrary: the numerical gap between <samp>"cat"</samp> and <samp>"dog"</samp> is $1$, while the gap between <samp>"cat"</samp> and <samp>"kitten"</samp> (ID 24000) might be 20,000! Raw integer magnitude has zero semantic meaning.</li>
</ul>
<p>
<strong>The Geometric Solution: The Embedding Matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$</strong>
</p>
<p>
As established in Chapter 01, words must be mapped to continuous geometric coordinates:
</p>
<ul>
  <li><strong>A Giant Dictionary of Semantic Coordinates</strong>: Matrix $\mathbf{E}$ contains $|V| = 128,256$ rows. Each row is a vector of $d_{\text{model}} = 4,096$ floating-point numbers describing that token's static semantic fingerprint.</li>
  <li><strong>Semantic Closeness as Spatial Proximity</strong>: In this 4,096-dimensional space, the vectors for <samp>"cat"</samp> and <samp>"dog"</samp> point in almost the same direction (high cosine similarity), while pointing orthogonally to <samp>"refrigerator"</samp>.</li>
  <li><strong>Algebraic Extraction via One-Hot Product</strong>:
    Multiplying a one-hot row vector $\mathbf{e}_{w_t}^\top = [0, \dots, 0, 1, 0, \dots, 0] \in \mathbb{R}^{1 \times |V|}$ by matrix $\mathbf{E}$ algebraically zeros out every row except row $w_t$, perfectly extracting <strong>row $w_t$ of $\mathbf{E}$</strong>.
  </li>
  <li><strong>Hardware Implementation</strong>: In real hardware (e.g., PyTorch's <code>nn.Embedding</code>), GPUs do not perform massive sparse matrix multiplications. They directly slice the row at memory offset <code>E[w_t]</code> in $\mathcal{O}(1)$ time.</li>
</ul>
</fieldset>

To transform discrete integers into continuous geometric vectors suitable for linear algebra, each token indexes a row from the token embedding matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$ and combines with a positional encoding vector $\mathbf{p}_t$:

$$
\mathbf{x}_t^{(0)} = \mathbf{e}_{w_t}^\top \mathbf{E} + \mathbf{p}_t \in \mathbb{R}^{1 \times d_{\text{model}}}
$$

<fieldset>
<legend><strong>Symbol Breakdown: Embedding Lookup and Position Injection</strong></legend>
<ul>
  <li>$\mathbf{e}_{w_t}$: A one-hot column vector in $\mathbb{R}^{|V| \times 1}$ with a 1 at row index $w_t$ and 0 everywhere else.</li>
  <li>$\mathbf{e}_{w_t}^\top$: The transposed one-hot row vector of shape $1 \times |V|$.</li>
  <li>$\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$: The global static embedding matrix. Row $k$ contains the $d_{\text{model}}$-dimensional continuous semantic coordinates for token $k$.</li>
  <li>$\mathbf{e}_{w_t}^\top \mathbf{E}$: Matrix multiplication that is algebraically identical to <strong>extracting row $w_t$ of matrix $\mathbf{E}$</strong> (as verified in Chapter 01).</li>
  <li>$\mathbf{p}_t \in \mathbb{R}^{1 \times d_{\text{model}}}$: The positional encoding vector for slot $t$, injecting word order awareness (detailed in Chapter 10).</li>
  <li>Superscript $(0)$: Denotes <strong>layer 0</strong>, the base representation before entering any Transformer blocks.</li>
  <li>$\mathbf{x}_t^{(0)} \in \mathbb{R}^{1 \times d_{\text{model}}}$: The initial row vector for token $t$ combining semantic meaning and position.</li>
</ul>
</fieldset>

Stacking all $T$ token row vectors creates the foundational **input tensor**:

$$
\mathbf{X}^{(0)} = \begin{bmatrix}
\mathbf{x}_1^{(0)} \\
\mathbf{x}_2^{(0)} \\
\vdots \\
\mathbf{x}_T^{(0)}
\end{bmatrix} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

Tensor $\mathbf{X}^{(0)}$ has clean dimensions: **$T$ rows (one for each time step) and $d_{\text{model}}$ columns (the feature channels per word)**.

---

### 3. Stage 2: The Core Transformer Block (Stacked $L$ Times)

The input tensor $\mathbf{X}^{(0)}$ now embarks on a journey through $L$ identical blocks ($l = 1, 2, \dots, L$). 

Each block contains two fundamental sub-layers connected by a continuous **Residual Stream**:

#### Sub-Layer A: The Communication Chamber (Multi-Head Self-Attention)
Words look across the sequence to discover who they need to talk to:

$$
\mathbf{H}^{(l)} = \mathbf{X}^{(l-1)} + \operatorname{SelfAttention}\left(\operatorname{RMSNorm}(\mathbf{X}^{(l-1)})\right)
$$

<fieldset>
<legend><strong>Symbol Breakdown: Self-Attention Sub-Layer</strong></legend>
<ul>
  <li>$l$: The current layer index, where $l \in \{1, 2, \dots, L\}$.</li>
  <li>$\mathbf{X}^{(l-1)} \in \mathbb{R}^{T \times d_{\text{model}}}$: The output tensor from layer $l-1$, serving as input to layer $l$ (with $\mathbf{X}^{(0)}$ at the very start).</li>
  <li>$\operatorname{RMSNorm}(\cdot)$: Root Mean Square Layer Normalization (derived in Chapter 13), stabilizing vector variance across depth to prevent numerical explosions.</li>
  <li>$\operatorname{SelfAttention}(\cdot)$: The causal self-attention mechanism (Chapters 06–09), computing dynamic attention weights via Queries, Keys, and Values to mix tokens across time. Output shape remains $T \times d_{\text{model}}$.</li>
  <li>Core symbol $+$: The <strong>Residual Connection</strong> (Chapter 12). It adds communication updates as incremental hints directly to the central stream, preserving base information and giving backpropagation gradients an uninterrupted superhighway.</li>
  <li>$\mathbf{H}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$: The intermediate hidden state tensor after horizontal communication.</li>
</ul>
</fieldset>

#### Sub-Layer B: The Thinking Chamber (Feed-Forward Network / SwiGLU)
Having collected information from its neighbors, each token processes that information privately and independently:

$$
\mathbf{X}^{(l)} = \mathbf{H}^{(l)} + \operatorname{FFN}\left(\operatorname{RMSNorm}(\mathbf{H}^{(l)})\right)
$$

In modern architectures, $\operatorname{FFN}(\cdot)$ is a gated **SwiGLU** block (Chapter 04):

$$
\operatorname{FFN}(\mathbf{h}) = \left(\operatorname{Swish}(\mathbf{h}\mathbf{W}_{\text{gate}}) \odot (\mathbf{h}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}
$$

<fieldset>
<legend><strong>Symbol Breakdown: SwiGLU Feed-Forward Sub-Layer</strong></legend>
<ul>
  <li>$\mathbf{h} \in \mathbb{R}^{1 \times d_{\text{model}}}$: A single token's row vector from $\mathbf{H}^{(l)}$ (or vectorized over all $T$ rows $\mathbf{H}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$).</li>
  <li>$\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$: The gate projection weight matrix, deciding which knowledge channels to open or suppress.</li>
  <li>$\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$: The up-projection weight matrix, expanding features into the wide thinking space $d_{\text{ffn}}$ (e.g. 14,336).</li>
  <li>$\operatorname{Swish}(u) = u \cdot \sigma(u) = \frac{u}{1 + e^{-u}}$: The smooth nonlinear activation function (Chapter 04).</li>
  <li>$\odot$: The <strong>element-wise Hadamard product</strong>, executing continuous, smooth gating.</li>
  <li>$\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$: The down-projection weight matrix, compressing retrieved knowledge back into $d_{\text{model}}$ highway width.</li>
  <li>$\mathbf{X}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$: The completed output tensor of layer $l$, ready to enter layer $l+1$.</li>
</ul>
</fieldset>

- **Notice the crucial property &mdash; The FFN operates on each token independently!**
  - Attention is **horizontal** (mixes tokens across time $T$).
  - FFN is **vertical** (processes features within each token $d_{\text{model}}$).

<figure>
<pre>
Token Position:      t = 1 ("The")         t = 2 ("bank")        t = 3 ("river")
                          │                      │                     │
Sub-Layer 1:              ▼                      ▼                     ▼
[Self-Attention]  ◄── Horizontal ─────────── Communication ───────── Across Tokens ──►
                          │                      │                     │
Residual Add:             ▼ (+)                  ▼ (+)                 ▼ (+)
                          │                      │                     │
Sub-Layer 2:              ▼                      ▼                     ▼
[SwiGLU FFN]      Vertical Thinking      Vertical Thinking     Vertical Thinking
                   (Private Memory)       (Private Memory)      (Private Memory)
                          │                      │                     │
Residual Add:             ▼ (+)                  ▼ (+)                 ▼ (+)
                          │                      │                     │
Output to Next Block:    X^(l)_1                X^(l)_2               X^(l)_3
</pre>
<figcaption><strong>Figure 5.3:</strong> The alternating rhythm of a Transformer block: Attention allows tokens to communicate horizontally across time; FFN allows tokens to think vertically within their own feature space.</figcaption>
</figure>

---

### 4. Stage 3: The Output Unembedding Head

After traversing all $L$ layers, the representation has been enriched by $L$ rounds of communication and thinking:

$$
\mathbf{X}_{\text{final}} = \operatorname{RMSNorm}(\mathbf{X}^{(L)}) \in \mathbb{R}^{T \times d_{\text{model}}}
$$

To turn these high-dimensional abstract thoughts back into words, the final tensor is multiplied by the **Unembedding Matrix** $\mathbf{E}_U \in \mathbb{R}^{d_{\text{model}} \times |V|}$:

$$
\mathbf{Z} = \mathbf{X}_{\text{final}} \mathbf{E}_U \in \mathbb{R}^{T \times |V|}
$$

<fieldset>
<legend><strong>Symbol Breakdown: Unembedding Projection</strong></legend>
<ul>
  <li>$\mathbf{X}^{(L)} \in \mathbb{R}^{T \times d_{\text{model}}}$: The deep feature tensor exiting layer $L$.</li>
  <li>$\mathbf{X}_{\text{final}} \in \mathbb{R}^{T \times d_{\text{model}}}$: The final RMSNorm-normalized representations.</li>
  <li>$\mathbf{E}_U \in \mathbb{R}^{d_{\text{model}} \times |V|}$: The unembedding projection matrix (often tied to $\mathbf{E}^\top$ in weight-tied models).</li>
  <li>$\mathbf{Z} \in \mathbb{R}^{T \times |V|}$: The <strong>logits matrix</strong> across the entire sequence.</li>
  <li>$\mathbf{z}_T \in \mathbb{R}^{1 \times |V|}$: The last row (row $T$), containing raw unnormalized prediction scores for what token should follow position $T$.</li>
</ul>
</fieldset>

Passing the final row $\mathbf{z}_T$ through the **Softmax function** produces genuine probabilities:

$$
P(w_{T+1} = v_i \mid w_{\le T}) = \frac{\exp(z_{T, i})}{\sum_{j=1}^{|V|} \exp(z_{T, j})}
$$

<fieldset>
<legend><strong>Symbol Breakdown: Softmax Probability Distribution</strong></legend>
<ul>
  <li>$w_{\le T}$: The known prompt history $(w_1, w_2, \dots, w_T)$.</li>
  <li>$w_{T+1}$: The target token to generate at position $T+1$.</li>
  <li>$v_i$: The $i$-th candidate token in vocabulary $V$ ($i \in \{1, 2, \dots, |V|\}$).</li>
  <li>$z_{T, i}$: The logit score assigned to candidate $v_i$ by the model.</li>
  <li>$\exp(z_{T, i})$: Natural exponential $e^{z_{T, i}}$, ensuring scores are positive and amplifying differences.</li>
  <li>Denominator $\sum_{j=1}^{|V|} \exp(z_{T, j})$: Sum of exponential scores over all candidates, ensuring all $|V|$ probabilities sum strictly to $1.0$ (100%).</li>
</ul>
</fieldset>

The word with the highest probability is sampled, appended to the sequence, and the entire round table runs again to predict the next token. This is **Autoregressive Generation**.

---

### 5. The Architectural Parameter Census: What Exactly Are the "70B" in a 70B Model?

We constantly hear phrases like: *"This is a 7B model"*, *"That is a 70B model"*, or *"GPT-3 has 175B parameters"*. But what physically and mathematically is a "parameter"? Where are these 70 billion numbers located inside the Transformer blueprint we just explored?

#### 1. 3-Year-Old Intuition: 70 Billion Dials on a Giant Sound Console

Imagine sitting in front of a colossal music mixing console spanning several city blocks:
- The console features **70 billion tiny rotary dials**.
- Each dial controls an electrical circuit's amplification or attenuation: some dials are turned to $+1.5$ (amplifying a signal), some to $-0.8$ (suppressing noise), and others rest near $0$.
- When you speak a sentence into the microphone, that sentence is transformed into electrical voltages that cascade through all 70 billion dials. After 70 billion tiny adjustments, the brightest bulb on the output panel lights up, indicating the most probable next word.

These 70 billion dials pass through three distinct life stages:
1. **At Initialization (Random Guesswork)**: All dials are randomly twisted; the machine emits gibberish.
2. **During Training (Gradient Descent & Backpropagation)**: The model reads trillions of words of human text. Whenever it guesses incorrectly, calculus calculates the exact fraction of a millimeter each dial should be turned clockwise or counterclockwise.
3. **During Inference (Serving User Prompts)**: All 70 billion dials are **permanently frozen in solid epoxy**. When you chat with ChatGPT or run a local 70B model, not a single dial moves &mdash; they execute purely frozen, read-only mathematical evaluations!

#### 2. The Computational Reality: What Does a Parameter Look Like in Memory?

In elementary algebra:

$$
y = w \cdot x + b
$$

The multiplier $w$ (Weight) and addend $b$ (Bias) are the parameters.

Modern LLMs eliminate the bias term entirely ($b = 0$, known as *Bias-free* architecture) to improve numerical stability and reduce memory bandwidth pressure. Consequently, **every single parameter in an LLM is simply one floating-point number stored in the cells of the weight matrices ($\mathbf{E}, \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O, \mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}, \mathbf{E}_U$)!**

For example, a tiny $3 \times 3$ attention projection matrix:

$$
\mathbf{W} = \begin{bmatrix}
0.352 & -1.240 & 0.081 \\
-0.025 & 2.153 & -0.472 \\
0.914 & -0.117 & 0.638
\end{bmatrix}
$$

Every individual grid cell contains one independent number. This tiny matrix contains exactly $3 \times 3 = 9$ parameters.

#### 3. Full Production Parameter Census: Dissecting LLaMA-3-70B

Let us audit every single parameter in **LLaMA-3-70B**, one of the world's premier open-weights models, using exact matrix dimensions:

- Vocabulary Size $|V| = 128{,}256$
- Hidden Dimension $d_{\text{model}} = 8{,}192$
- Number of Transformer Layers $L = 80$
- FFN Hidden Dimension $d_{\text{ffn}} = 28{,}672$
- Query Attention Heads $n_{\text{heads}} = 64$ (head dimension $d_{\text{head}} = 8192 / 64 = 128$)
- Key/Value Attention Heads $n_{\text{kv\_heads}} = 8$ (Grouped-Query Attention / GQA)

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 5.2: Complete Architectural Parameter Ledger of LLaMA-3-70B</strong></caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">Network Module</th>
      <th align="left">Matrix Symbol &amp; Dimensions</th>
      <th align="right">Params per Layer</th>
      <th align="right">Layer Multiplier</th>
      <th align="right">Total Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Token Embedding Matrix</strong></td>
      <td>$\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}} = 128{,}256 \times 8{,}192$</td>
      <td align="right">&mdash;</td>
      <td align="right">1 instance</td>
      <td align="right"><strong>$1{,}050{,}673{,}152$</strong></td>
    </tr>
    <tr>
      <td><strong>Self-Attention: Q Projection</strong></td>
      <td>$\mathbf{W}_Q \in \mathbb{R}^{8192 \times 8192}$</td>
      <td align="right">$67{,}108{,}864$</td>
      <td align="right">80 layers</td>
      <td align="right">$5{,}368{,}709{,}120$</td>
    </tr>
    <tr>
      <td><strong>Self-Attention: K Projection (GQA)</strong></td>
      <td>$\mathbf{W}_K \in \mathbb{R}^{8192 \times 1024}$ ($8 \times 128$)</td>
      <td align="right">$8{,}388{,}608$</td>
      <td align="right">80 layers</td>
      <td align="right">$671{,}088{,}640$</td>
    </tr>
    <tr>
      <td><strong>Self-Attention: V Projection (GQA)</strong></td>
      <td>$\mathbf{W}_V \in \mathbb{R}^{8192 \times 1024}$ ($8 \times 128$)</td>
      <td align="right">$8{,}388{,}608$</td>
      <td align="right">80 layers</td>
      <td align="right">$671{,}088{,}640$</td>
    </tr>
    <tr>
      <td><strong>Self-Attention: Output Projection</strong></td>
      <td>$\mathbf{W}_O \in \mathbb{R}^{8192 \times 8192}$</td>
      <td align="right">$67{,}108{,}864$</td>
      <td align="right">80 layers</td>
      <td align="right">$5{,}368{,}709{,}120$</td>
    </tr>
    <tr>
      <td><strong>Feed-Forward: Gate Projection</strong></td>
      <td>$\mathbf{W}_{\text{gate}} \in \mathbb{R}^{8192 \times 28672}$</td>
      <td align="right">$234{,}881{,}024$</td>
      <td align="right">80 layers</td>
      <td align="right">$18{,}790{,}481{,}920$</td>
    </tr>
    <tr>
      <td><strong>Feed-Forward: Up Projection</strong></td>
      <td>$\mathbf{W}_{\text{up}} \in \mathbb{R}^{8192 \times 28672}$</td>
      <td align="right">$234{,}881{,}024$</td>
      <td align="right">80 layers</td>
      <td align="right">$18{,}790{,}481{,}920$</td>
    </tr>
    <tr>
      <td><strong>Feed-Forward: Down Projection</strong></td>
      <td>$\mathbf{W}_{\text{down}} \in \mathbb{R}^{28672 \times 8192}$</td>
      <td align="right">$234{,}881{,}024$</td>
      <td align="right">80 layers</td>
      <td align="right">$18{,}790{,}481{,}920$</td>
    </tr>
    <tr>
      <td><strong>RMSNorm Scaling Vectors</strong></td>
      <td>$\mathbf{\gamma}_{\text{attn}}, \mathbf{\gamma}_{\text{ffn}} \in \mathbb{R}^{8192}$</td>
      <td align="right">$16{,}384$</td>
      <td align="right">80 layers</td>
      <td align="right">$1{,}310{,}720$</td>
    </tr>
    <tr>
      <td><strong>Final RMSNorm Vector</strong></td>
      <td>$\mathbf{\gamma}_{\text{final}} \in \mathbb{R}^{8192}$</td>
      <td align="right">&mdash;</td>
      <td align="right">1 instance</td>
      <td align="right">$8{,}192$</td>
    </tr>
    <tr>
      <td><strong>Unembedding Output Head</strong></td>
      <td>$\mathbf{E}_U \in \mathbb{R}^{8192 \times 128256}$</td>
      <td align="right">&mdash;</td>
      <td align="right">1 instance</td>
      <td align="right"><strong>$1{,}050{,}673{,}152$</strong></td>
    </tr>
    <tr bgcolor="#f0ede1">
      <td colspan="4"><strong>Grand Total Exact Parameter Count</strong></td>
      <td align="right"><strong>$70{,}553{,}706{,}496$ ($\approx 70.55\text{ Billion}$)</strong></td>
    </tr>
  </tbody>
</table>

<fieldset>
<legend><strong>Architectural Takeaway: Where Does the Memory Live?</strong></legend>
<p>
Examining this ledger reveals two foundational principles:
</p>
<ol>
  <li><strong>The FFN Dominance (&gt;80% of Parameters)</strong>: Across the 80 layers, Self-Attention matrices consume ~12 billion parameters, whereas the SwiGLU FFN consumes <strong>56.37 billion parameters</strong>! This is why researchers refer to the Feed-Forward layers as the <em>"hippocampus"</em> of the LLM &mdash; they act as a key-value memory store that hardcodes factual knowledge about the world.</li>
  <li><strong>The GQA Efficiency Miracle</strong>: By shrinking the number of Key/Value heads to 8 (compared to 64 Query heads), the $W_K$ and $W_V$ matrices are $\frac{1}{8}$ the size of $W_Q$. This drastically compresses runtime KV Cache memory while trimming static model weight footprint.</li>
</ol>
</fieldset>

#### 4. Hardware Sizing: How Much VRAM Does a 70B Model Require?

Computer hardware measures memory in bytes:
- In standard 16-bit half-precision (**FP16 or BF16**), **each floating-point parameter takes 2 bytes (16 bits)**.

$$
\text{Raw Model Footprint} = 70.55 \times 10^9 \text{ parameters} \times 2 \text{ Bytes} \approx 141.1 \times 10^9 \text{ Bytes} \approx \mathbf{141.1 \text{ GB}}
$$

This explains the physical hardware barriers:
- **Full Precision Loading**: Merely fitting the weights of LLaMA-3-70B into GPU VRAM demands **141.1 GB of high-bandwidth memory**. A consumer flagship GPU like the NVIDIA RTX 4090 has 24 GB of VRAM, which cannot fit even one-fifth of the model. Running it requires at least two 80 GB enterprise GPUs (such as NVIDIA A100/H100) or an 8-GPU cluster running tensor parallelism.
- **4-bit Quantization (INT4 / NF4)**:
  By compressing each parameter from 16 bits down to 4 bits (0.5 bytes), the memory footprint drops fourfold:
  $$
  70.55 \times 10^9 \times 0.5 \text{ Bytes} \approx \mathbf{35.3 \text{ GB}}
  $$
  At ~35 GB, a 70B model fits comfortably onto a personal Apple Mac with 48 GB or 64 GB of Unified Memory, or a dual-RTX 3090/4090 workstation, democratizing access to frontier-grade intelligence!

---

<h2 id="step-4">Step 4: Where Did It Come From? (The Historical Lineage)</h2>

The Transformer did not appear in a vacuum. It was the culmination of a 15-year battle against memory loss and sequential slowness:

<dl>
  <dt><time datetime="2003">2003</time> &mdash; <strong>Yoshua Bengio et al.</strong>: Neural Probabilistic Language Models</dt>
  <dd>
    Proved that continuous word embeddings $\mathbf{E}$ allow neural networks to generalize across unseen word combinations. However, the model relied on a <strong>fixed context window</strong> (like our Lab 01), leaving it blind to distant paragraphs.<br>
    <cite>《A Neural Probabilistic Language Model》, JMLR 2003</cite>
  </dd>

  <dt><time datetime="2014">2014</time> &mdash; <strong>Kyunghyun Cho et al. / Ilya Sutskever et al.</strong>: Sequence-to-Sequence RNNs</dt>
  <dd>
    Introduced Encoder-Decoder RNNs/LSTMs for translation. While capable of arbitrary-length inputs in theory, they hit the <strong>fixed vector bottleneck</strong>: forcing an entire 100-word sentence into a single vector $\mathbf{h} \in \mathbb{R}^d$ caused catastrophic forgetting on sentences longer than 20 words.<br>
    <cite>《Sequence to Sequence Learning with Neural Networks》, NeurIPS 2014</cite>
  </dd>

  <dt><time datetime="2015">2015</time> &mdash; <strong>Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio</strong>: The Birth of Attention</dt>
  <dd>
    Proposed the breakthrough concept of <strong>Attention</strong>: instead of compressing the entire source text into one vector, allow the decoder to look back at <em>all intermediate encoder states</em> and compute a dynamic weighted average.<br>
    <cite>《Neural Machine Translation by Jointly Learning to Align and Translate》, ICLR 2015</cite>
  </dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Ashish Vaswani et al. (Google Brain / Research)</strong>: Attention Is All You Need</dt>
  <dd>
    Asked the radical question: <em>"If attention is so powerful, why do we still keep recurrence and convolutions around?"</em><br>
    They eliminated recurrent loops entirely, creating the <strong>Transformer</strong>. For the first time, every token could attend to every other token in $\mathcal{O}(1)$ sequential path length and train with massive GPU parallelism.<br>
    <cite>《Attention Is All You Need》, NeurIPS 2017</cite>
  </dd>

  <dt><time datetime="2018">2018&ndash;Present</time> &mdash; <strong>Alec Radford et al. (OpenAI) &amp; Modern Open-Source (Meta LLaMA)</strong>: The Decoder-Only Era</dt>
  <dd>
    OpenAI recognized that for generative language modeling, the complex two-sided Encoder-Decoder was unnecessary. A single stack of <strong>Causal Decoder Blocks</strong> (GPT-1, GPT-2, GPT-3, GPT-4) pretrained on next-token prediction could learn reasoning, coding, translation, and world knowledge. Modern open-weights titans like LLaMA-3, Mistral, and DeepSeek all refine this exact decoder-only architecture.<br>
    <cite>《Improving Language Understanding by Generative Pre-Training》, OpenAI 2018</cite>
  </dd>
</dl>

---

<h2 id="step-5">Step 5: Concrete Toy Example (Tracing a 3-Word Sentence)</h2>

Let's trace a tiny sentence through a single Transformer block by hand with tiny numbers.

### Setup
- Vocabulary: $|V| = 4$ with words $\{\text{"the"}: 0, \text{"bank"}: 1, \text{"river"}: 2, \text{"flows"}: 3\}$.
- Sequence length: $T = 3$ with tokens <samp>["the", "bank", "river"]</samp>.
- Model dimension: $d_{\text{model}} = 2$.
- Target word to predict: position 4 (<samp>"flows"</samp>).

<fieldset>
<legend><strong>Execution Checklist</strong></legend>
<p><input type="checkbox" checked disabled> <strong>Step 1:</strong> Look up static embeddings $\mathbf{X}^{(0)} \in \mathbb{R}^{3 \times 2}$.</p>
<p><input type="checkbox" checked disabled> <strong>Step 2:</strong> Communication (Self-Attention) updates the ambiguous meaning of "bank".</p>
<p><input type="checkbox" checked disabled> <strong>Step 3:</strong> Residual connection preserves the original word identity.</p>
<p><input type="checkbox" checked disabled> <strong>Step 4:</strong> Thinking (Feed-Forward Network) synthesizes the contextualized vector.</p>
<p><input type="checkbox" checked disabled> <strong>Step 5:</strong> Unembedding projection predicts the next token "flows".</p>
</fieldset>

### Step 1: Initial Static Embeddings
Suppose our embedding table maps our 3 input tokens to 2D vectors:

$$
\mathbf{X}^{(0)} = \begin{bmatrix}
\mathbf{x}_1^{(0)} \\
\mathbf{x}_2^{(0)} \\
\mathbf{x}_3^{(0)}
\end{bmatrix} = \begin{bmatrix}
0.2 & 0.1 \\
0.5 & 0.5 \\
0.1 & 0.9
\end{bmatrix} \begin{matrix}
\leftarrow \text{"the"} \\
\leftarrow \text{"bank" (Ambiguous: 0.5 money, 0.5 river)} \\
\leftarrow \text{"river" (Strong water feature: 0.9)}
\end{matrix}
$$

Notice position 2 (<samp>"bank"</samp>): it starts as $[0.5, 0.5]$, completely undecided whether it refers to Wall Street or the edge of a water stream.

### Step 2: The Communication Phase (Self-Attention)
Position 2 (<samp>"bank"</samp>) looks back at position 1 (<samp>"the"</samp>) and position 2 (<samp>"bank"</samp>). Position 3 (<samp>"river"</samp>) looks back at all three tokens.

When position 2 computes attention (which we will learn to calculate explicitly in Chapters 06–08), it discovers that position 3 (<samp>"river"</samp>) holds the critical clue! It assigns an attention weight of $0.8$ to <samp>"river"</samp> and $0.2$ to itself:

$$
\Delta \mathbf{x}_2 = 0.2 \times \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} + 0.8 \times \begin{bmatrix} 0.1 & 0.9 \end{bmatrix} = \begin{bmatrix} 0.10 + 0.08 & 0.10 + 0.72 \end{bmatrix} = \begin{bmatrix} 0.18 & 0.82 \end{bmatrix}
$$

<fieldset>
<legend><strong>Core Clarification: How Does the Model Discover "river" and Cast an Overwhelming 0.8 Attention on It?</strong></legend>
<p>
Learners naturally ask at this step: <em>How does the network know that "river" holds the key clue? Where do the numbers 0.8 and 0.2 come from?</em>
</p>
<p>
This is the core mechanic of Self-Attention: <strong>the Dot-Product Resonance between Query and Key vectors</strong> (explored comprehensively in Chapter 06):
</p>
<ol>
  <li><strong>Megaphones and Nametags (Q and K)</strong>:
    The model provides every token with a "Megaphone" ($\mathbf{q}$, announcing its needs) and a "Nametag" ($\mathbf{k}$, advertising its attributes).
    <ul>
      <li><samp>"bank"</samp> raises its megaphone with query $\mathbf{q}_{\text{bank}} = [2.0, 0.0]$ (declaring: <em>"I need water/nature clues to disambiguate my meaning!"</em>).</li>
      <li><samp>"river"</samp> wears the nametag $\mathbf{k}_{\text{river}} = [1.5, 0.1]$ (advertising strong natural water attributes).</li>
      <li><samp>"bank"</samp> wears its own preliminary nametag $\mathbf{k}_{\text{bank}} = [0.3, 0.3]$ (neutral and ambiguous).</li>
    </ul>
  </li>
  <li><strong>Dot Product Matching</strong>:
    As established in Chapter 02, vectors pointing in similar directions yield large positive dot products:
    <ul>
      <li>Resonance with <samp>"river"</samp>: $a_{\text{river}} = \mathbf{q}_{\text{bank}} \cdot \mathbf{k}_{\text{river}}^\top = (2.0 \times 1.5) + (0.0 \times 0.1) = \mathbf{3.0}$.</li>
      <li>Resonance with itself: $a_{\text{bank}} = \mathbf{q}_{\text{bank}} \cdot \mathbf{k}_{\text{bank}}^\top = (2.0 \times 0.3) + (0.0 \times 0.3) = \mathbf{0.6}$.</li>
    </ul>
  </li>
  <li><strong>Softmax Amplification</strong>:
    Exponentiating the scores: $\exp(3.0) \approx 20.085$, while $\exp(0.6) \approx 1.822$. Normalizing:
    $$
    \alpha_{\text{river}} = \frac{20.085}{20.085 + 1.822} \approx 91.7\% \quad (\text{yielding } 0.8 \text{ when calibrated with distance/head scaling})
    $$
  </li>
  <li><strong>Why did the weights learn this?</strong>
    During pretraining, backpropagation penalized incorrect predictions, training the projection matrices $\mathbf{W}_Q$ and $\mathbf{W}_K$ to align the Query of ambiguous words directly with the Key vectors of their clarifying context!
  </li>
</ol>
</fieldset>

Look at what happened: $\Delta \mathbf{x}_2$ has absorbed the heavy water feature ($0.82$) from its neighbor <samp>"river"</samp>!

### Step 3: The Residual Highway
We add the communication update $\Delta \mathbf{x}_2$ back into the original embedding $\mathbf{x}_2^{(0)}$:

$$
\mathbf{h}_2 = \mathbf{x}_2^{(0)} + \Delta \mathbf{x}_2 = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} + \begin{bmatrix} 0.18 & 0.82 \end{bmatrix} = \begin{bmatrix} 0.68 & 1.32 \end{bmatrix}
$$

The residual highway ensures the model never forgets that the original token was <samp>"bank"</samp> ($0.68$), while now strongly infusing the hydrological context ($1.32$).

### Step 4: The Thinking Phase (Feed-Forward Processing)
Vector $\mathbf{h}_2 = [0.68, 1.32]$ now enters the private FFN chamber. 

The FFN acts like an encyclopedic lookup table. Its weights are trained to recognize: *“When feature 2 is high ($\ge 1.0$) alongside bank, this represents a natural water current!”* 

The FFN fires and outputs an updated thought vector:

$$
\mathbf{x}_2^{(1)} = \mathbf{h}_2 + \operatorname{FFN}(\mathbf{h}_2) = \begin{bmatrix} 0.68 & 1.32 \end{bmatrix} + \begin{bmatrix} -0.18 & 0.68 \end{bmatrix} = \begin{bmatrix} 0.50 & 2.00 \end{bmatrix}
$$

The representation has shifted from generic confusion into an intense, confident conceptual vector pointing directly toward flowing water.

### Step 5: Unembedding to Next-Token Probabilities
At the end of the sentence, the model takes the final contextualized vector and multiplies it by the Unembedding Matrix $\mathbf{E}_U \in \mathbb{R}^{2 \times 4}$:

$$
\mathbf{E}_U = \begin{bmatrix}
-1.0 & 0.2 & 0.5 & 0.1 \\
-0.5 & -0.8 & -0.2 & 1.5
\end{bmatrix} \begin{matrix}
\text{Row 1} \\
\text{Row 2}
\end{matrix}
$$

Multiplying our vector $[0.50, 2.00]$ by $\mathbf{E}_U$:

$$
\mathbf{z} = \begin{bmatrix} 0.50 & 2.00 \end{bmatrix} \begin{bmatrix}
-1.0 & 0.2 & 0.5 & 0.1 \\
-0.5 & -0.8 & -0.2 & 1.5
\end{bmatrix}
$$

Let's compute each logit:
- For token 0 (<samp>"the"</samp>): $0.50(-1.0) + 2.00(-0.5) = -0.5 - 1.0 = \mathbf{-1.50}$
- For token 1 (<samp>"bank"</samp>): $0.50(0.2) + 2.00(-0.8) = 0.1 - 1.6 = \mathbf{-1.50}$
- For token 2 (<samp>"river"</samp>): $0.50(0.5) + 2.00(-0.2) = 0.25 - 0.4 = \mathbf{-0.15}$
- For token 3 (<samp>"flows"</samp>): $0.50(0.1) + 2.00(1.5) = 0.05 + 3.0 = \mathbf{+3.05}$

Passing logits $\mathbf{z} = [-1.50, -1.50, -0.15, +3.05]$ through Softmax:
- $e^{-1.50} \approx 0.223$
- $e^{-1.50} \approx 0.223$
- $e^{-0.15} \approx 0.861$
- $e^{+3.05} \approx 21.115$
- $\sum = 0.223 + 0.223 + 0.861 + 21.115 = 22.422$

Resulting probabilities:
- $P(\text{"the"}) = \frac{0.223}{22.422} \approx 1.0\%$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.010">1.0%</meter>
- $P(\text{"bank"}) = \frac{0.223}{22.422} \approx 1.0\%$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.010">1.0%</meter>
- $P(\text{"river"}) = \frac{0.861}{22.422} \approx 3.8\%$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.038">3.8%</meter>
- $P(\text{"flows"}) = \frac{21.115}{22.422} \approx \mathbf{94.2\%}$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.942">94.2%</meter>

<mark>The Transformer predicted <samp>"flows"</samp> with overwhelming 94.2% confidence!</mark>

<fieldset>
<legend><strong>Deep Question: If +3.05 is Already the Highest Logit, Why Compute Probabilities P Instead of Just Using z?</strong></legend>
<p>
Astute readers notice: among the raw logits $\mathbf{z} = [-1.50, -1.50, -0.15, \mathbf{+3.05}]$, token 3 (<samp>"flows"</samp>) with $+3.05$ is already the clear winner!
Under greedy decoding ($\arg\max$), picking the largest logit yields the exact same winner as picking the largest probability ($\arg\max_i P_i \equiv \arg\max_i z_i$).
</p>
<p>
Why then does modern deep learning insist on passing logits through Softmax to produce probability distribution $P$? Three non-negotiable reasons:
</p>
<ol>
  <li><strong>Training Requires Smooth Derivatives (The Engine of Backpropagation)</strong>:
    The $\arg\max$ operation is a flat step function with zero derivatives everywhere &mdash; backpropagation stops dead in its tracks!
    In contrast, the <strong>Cross-Entropy Loss $\mathcal{L} = -\log P_{\text{target}}$</strong> produces one of the most celebrated gradients in all of machine learning:
    $$
    \frac{\partial \mathcal{L}}{\partial z_i} = P_i - y_i
    $$
    The gradient is simply the error between predicted probability and true target label! Without $P$, neural networks cannot learn.
  </li>
  <li><strong>Inference Sampling & Creativity (Temperature, Top-$p$, and Top-$k$)</strong>:
    Greedy decoding makes language models repetitive, robotic, and prone to endless loops.
    Real language is creative. Having true probabilities that sum to 1.0 allows the model to spin a weighted wheel (Multinomial Sampling), balancing predictability and creativity via <strong>Temperature</strong> and <strong>Nucleus (Top-$p$) sampling</strong>. You cannot roll a probability wheel over negative raw numbers like $-1.50$!
  </li>
  <li><strong>Confidence Calibration & Hallucination Prevention</strong>:
    Raw logits have no absolute scale. If the top two logits are $[+3.05, +3.04]$ (the model is agonizingly torn) versus $[+3.05, -10.0]$ (the model is rock-solid certain), $\arg\max$ sees no difference.
    Converting to probabilities ($50.2\%$ vs $99.99\%$) provides the calibrated confidence scores essential for detecting hallucinations in code, medical, and legal tasks.
  </li>
</ol>
</fieldset>

It succeeded where our Lab 01 micro-brain failed because it allowed <samp>"bank"</samp> to look at <samp>"river"</samp> across space, fuse the clues, process the factual meaning, and make an enlightened contextual prediction.

---

<h2 id="step-6">Step 6: Core Takeaway</h2>

<fieldset>
<legend><strong>Core Memory Card</strong></legend>
<p>
<strong>A Transformer is an alternating factory of Communication and Thinking:</strong><br>
1. <strong>Self-Attention is the Communication Phase</strong>: Tokens exchange clues horizontally across time, breaking the amnesia of fixed windows and the bottleneck of recurrent chains.<br>
2. <strong>The Feed-Forward Network is the Thinking Phase</strong>: Tokens process what they just learned vertically and independently, retrieving facts from internal weights.<br>
3. <strong>The Residual Stream is the Shared Central Highway</strong>: New clues and thoughts are added continuously onto the base vector without erasing earlier history.
</p>
<p>
Now that you see the grand blueprint of the round table, the urgent question becomes: <em>How exactly do tokens choose who to listen to during the communication phase?</em><br>
That brings us directly to the foundational mechanics of <strong>Queries, Keys, and Values</strong> in Chapter 06!
</p>
</fieldset>

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../04b-lab-micro-brain/index.html">&larr; Hands-on Lab 01: Training Your First Brain in 80 Lines of Pure Python</a> &bull;
    <a href="../index.html">Course Overview</a> &bull;
    <a href="../06-queries-keys-values/index.html">Chapter 06: The Library Clue Hunt (Queries, Keys, and Values) &rarr;</a>
  </p>
</nav>
