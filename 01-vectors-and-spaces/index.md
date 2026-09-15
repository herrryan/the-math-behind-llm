# Chapter 01: The Word Map (Vectors & Embeddings)

<nav aria-label="Table of Contents">
  <p>
    <strong>Table of Contents:</strong> 
    <a href="#step-1">1. Intuition</a> &bull; 
    <a href="#step-2">2. Bridging Question</a> &bull; 
    <a href="#step-3">3. Exact Math</a> &bull; 
    <a href="#step-4">4. Origin</a> &bull; 
    <a href="#step-5">5. Toy Example</a> &bull; 
    <a href="#step-6">6. Core Takeaway</a>
  </p>
</nav>

<hr>

<h2 id="step-1">Step 1: 3-Year-Old Intuition (The Giant Playground Map)</h2>

> [!INTUITION] 3-Year-Old Intuition: The Giant Playground Map
> Imagine you bring all your toys outside into a giant grassy playground.
> 
> You arrange the playground into large sandboxes where similar toys sit side by side in neighboring sandboxes: teddy bears, kittens, and puppies in the soft cuddly sandbox; wooden airplanes and racecars in the zoom-machine sandbox; and bananas and apples in the snack sandbox.
> 
> Now imagine drawing chalk grid lines across the playground:
> - Walk 2 steps North and 8 steps East: you reach the kitten in the cuddly animal sandbox.
> - Walk 9 steps North and 1 step East: you reach the airplane in the machine sandbox.
> 
> Every toy now has an exact **address** on the playground map. 
> 
> If two toys are similar, their addresses sit close together in neighboring sandboxes. If two toys have nothing in common (like a fluffy kitten and a giant Boeing 747 airplane), their addresses are far apart on opposite sides of the playground.
> 
> This is what an LLM does with human language. It gives every word an address on a giant mathematical playground map so words with similar meanings sit right next to each other!

---

<h2 id="step-2">Step 2: The Bridging Question (From Words to Numbers)</h2>

In Chapter 00, we learned that a Large Language Model predicts probabilities over words. 

However, computer chips and GPUs are built from silicon and electric circuits. They do not know what the English letters <kbd>"c"</kbd>, <kbd>"a"</kbd>, <kbd>"t"</kbd> mean. A computer processor cannot multiply the word <kbd>"kitten"</kbd> by $0.90$.

**How do we convert human words into physical coordinates (lists of numbers) that a computer chip can store, add, and multiply?**

### The Failed Shortcut: Why We Cannot Simply Number Words 1, 2, 3

A tempting first idea is to assign a single integer ID to each word:
- $\text{"cat"} = 1$
- $\text{"dog"} = 2$
- $\text{"kitten"} = 3$
- $\text{"airplane"} = 4$

This simple scalar numbering fails catastrophically for two fundamental reasons:

1. **False Numerical Ordering**:  
   Computers treat numbers with natural order: $1 < 2 < 3 < 4$. Does it make sense to claim that an airplane is "greater than" a cat, or that a dog sits "halfway between" a cat and a kitten? Of course not! Words are distinct categorical concepts, not ranked quantities.

2. **False Arithmetic Equalities**:  
   In mathematics, $1 + 2 = 3$. If we used scalar IDs, the computer would conclude:
   $$\text{"cat"} (1) + \text{"dog"} (2) = \text{"kitten"} (3)$$
   This is meaningless nonsense. Adding a cat and a dog does not produce a baby cat!

### The True Solution: Multi-Dimensional Vectors

Instead of a single scalar number, we describe each word using an entire **list of coordinates**, where each coordinate measures an independent semantic property (e.g., cuddliness, size, living creature, mechanical nature).

This list of coordinates is called a **Vector** ($\mathbf{x} \in \mathbb{R}^d$), and the geometric space it lives in is called the **Embedding Space**.

---

<h2 id="step-3">Step 3: The Exact Math &amp; Formulas</h2>

### 1. The One-Hot Representation Vector ($\mathbf{e}_i$)

Before a word receives its multi-dimensional coordinates, it starts as an entry in the vocabulary $V$ of size $|V|$.

Each word at index $i \in \{1, 2, \dots, |V|\}$ is formally represented as a **one-hot vector** $\mathbf{e}_i \in \{0, 1\}^{|V|}$. This is a tall column vector filled entirely with zeros, except for a single $1$ at the word's designated position:

$$
\mathbf{e}_i = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} \leftarrow \text{index } i
$$

Formally, the $j^{\text{th}}$ entry of one-hot vector $\mathbf{e}_i$ is defined by the Kronecker delta:

$$
(\mathbf{e}_i)_j = \begin{cases} 1 & \text{if } j = i \\ 0 & \text{if } j \neq i \end{cases}
$$

---

### 2. The Embedding Matrix ($\mathbf{E}$)

The master word map of an LLM is stored as a large 2D parameter matrix called the **Embedding Matrix**, denoted by $\mathbf{E}$:

$$
\mathbf{E} \in \mathbb{R}^{|V| \times d}
$$

Where:
- $|V|$ is the total number of words in the vocabulary (the number of rows).
- $d$ is the **embedding dimension** or **hidden size** (the number of columns).

Each row $i$ of the matrix $\mathbf{E}$, written as $\mathbf{E}[i, :]$, is the dense vector embedding for word $i$:

$$
\mathbf{E} = \begin{bmatrix} 
\text{---} & \mathbf{x}_1^\top & \text{---} \\ 
\text{---} & \mathbf{x}_2^\top & \text{---} \\ 
& \vdots & \\ 
\text{---} & \mathbf{x}_{|V|}^\top & \text{---} 
\end{bmatrix} \in \mathbb{R}^{|V| \times d}
$$

---

### 3. The Embedding Lookup Equation

How does the neural network retrieve the coordinate vector $\mathbf{x}_i \in \mathbb{R}^d$ for word $i$?

Mathematically, word lookup is defined as the matrix multiplication of the transposed one-hot vector $\mathbf{e}_i^\top$ with the embedding matrix $\mathbf{E}$:

$$
\mathbf{x}_i^\top = \mathbf{e}_i^\top \mathbf{E} \in \mathbb{R}^{1 \times d}
$$

If written in standard column-vector orientation:

$$
\mathbf{x}_i = \mathbf{E}^\top \mathbf{e}_i \in \mathbb{R}^{d \times 1}
$$

Because $\mathbf{e}_i$ contains all zeros except for a $1$ at position $i$, this matrix multiplication acts like an **optical selector switch**: it zeros out every other row and extracts precisely row $i$ from matrix $\mathbf{E}$:

<figure>
<pre>
One-Hot Vector e_i^T (1 × |V|)           Embedding Matrix E (|V| × d)          Output Vector x_i (1 × d)
┌───────────────────────────────┐     ┌───────────────────────────────────┐     ┌────────────────────────┐
│ [ 0,  ...,  1,  ...,  0 ]     │  ×  │ Row 1:    [  0.12,  -0.45,  ... ] │  =  │ [  0.95,   0.15,  ... ] │
└──────────────┬────────────────┘     │ ...                               │     └────────────────────────┘
               │                      │ Row i:    [  0.95,   0.15,  ... ] │  ◄── Row i selected!
               └──────────────────────► ...                               │
                                      │ Row |V|:  [ -0.80,   0.21,  ... ] │
                                      └───────────────────────────────────┘
</pre>
<figcaption><strong>Figure 1.1:</strong> Vector lookup formulated as one-hot matrix multiplication: the solitary 1 selects row i.</figcaption>
</figure>

> [!NOTE] Practical Engineering Note: Lookup vs. Matrix Multiply
> In mathematical theory, vector lookup is expressed as $\mathbf{e}_i^\top \mathbf{E}$ because neural networks are defined as continuous differentiable matrix operations.
> 
> In production code (such as PyTorch's `torch.nn.Embedding`), multiplying a 100,000-element sparse vector on GPU is wasteful. Hardware libraries instead perform an instantaneous direct array index `E[i]`, which runs in $O(1)$ time and produces the exact same numerical result.

---

### Mathematical Symbol Breakdown Table

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 1.1:</strong> Mathematical symbols, dimensions, and physical meanings for vector embeddings.</caption>
  <thead>
    <tr>
      <th align="left">Symbol</th>
      <th align="left">Formal Name</th>
      <th align="left">Shape / Dimension</th>
      <th align="left">3-Year-Old Meaning</th>
      <th align="left">Concrete Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$V$</td>
      <td>Vocabulary Set</td>
      <td>Set of tokens</td>
      <td>The master toy box of all known words</td>
      <td>$V = \{\text{"cat"}, \text{"kitten"}, \text{"dog"}, \text{"airplane"}\}$</td>
    </tr>
    <tr>
      <td>$|V|$</td>
      <td>Vocabulary Size</td>
      <td>Scalar integer</td>
      <td>Total count of words in the toy box</td>
      <td>$|V| = 4$ in toy model ($32{,}000$ to $128{,}000$ in LLMs)</td>
    </tr>
    <tr>
      <td>$d$</td>
      <td>Embedding Dimension</td>
      <td>Scalar integer</td>
      <td>How many chalk directions are on our playground map</td>
      <td>$d = 2$ in our toy example ($4{,}096$ in LLaMA-3-8B)</td>
    </tr>
    <tr>
      <td>$\mathbf{e}_i$</td>
      <td>One-Hot Vector</td>
      <td>$\mathbb{R}^{|V| \times 1}$</td>
      <td>A light switch board with only switch $i$ turned ON</td>
      <td>$\mathbf{e}_2 = [0, 1, 0, 0]^\top$ (selects word 2)</td>
    </tr>
    <tr>
      <td>$\mathbf{E}$</td>
      <td>Embedding Matrix</td>
      <td>$\mathbb{R}^{|V| \times d}$</td>
      <td>The master address book storing coordinates for every toy</td>
      <td>Matrix of shape $4 \times 2$</td>
    </tr>
    <tr>
      <td>$\mathbf{x}_i^\top$ (or $\mathbf{x}_i$)</td>
      <td>Embedding Vector</td>
      <td>$\mathbb{R}^{1 \times d}$ (row) / $\mathbb{R}^{d \times 1}$ (column)</td>
      <td>The specific list of GPS coordinates for word $i$</td>
      <td>$\mathbf{x}_{\text{kitten}}^\top = [0.95, 0.15]$</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><strong>Symbol Glossary (Definition List)</strong></summary>

<dl>
  <dt><strong>Dense Vector</strong></dt>
  <dd>A vector where almost all entries are non-zero real numbers ($\mathbb{R}$), packing rich semantic information into a compact dimensional space.</dd>
  
  <dt><strong>Sparse Vector</strong></dt>
  <dd>A vector where the overwhelming majority of entries are zero (such as a one-hot vector with a single 1 and 99,999 zeros).</dd>
  
  <dt><strong>Embedding Space ($\mathbb{R}^d$)</strong></dt>
  <dd>The $d$-dimensional continuous vector space spanned by the coordinate axes where semantic concepts live as geometric points.</dd>
  
  <dt><strong>Kronecker Delta ($\delta_{ij}$)</strong></dt>
  <dd>A binary piecewise function that equals 1 if indices $i$ and $j$ match, and 0 otherwise.</dd>
</dl>
</details>

---

<h2 id="step-4">Step 4: Where Did It Come From?</h2>

> [!ORIGIN] The Historical Journey: From Firth to Word2Vec
> How did AI researchers discover that words could be treated as points on a map?

### 1. The Distributional Hypothesis (1957)
In 1957, British linguist John Rupert Firth wrote one of the most famous sentences in the history of linguistics:

> *"You shall know a word by the company it keeps."*  
> &mdash; **J. R. Firth (1957)**

Firth observed that words with similar meanings naturally appear in similar sentence environments:
- <kbd>"The fluffy [kitten] drank warm milk from the saucer."</kbd>
- <kbd>"The fluffy [cat] drank warm milk from the saucer."</kbd>

Because <kbd>"kitten"</kbd> and <kbd>"cat"</kbd> share nearly identical surrounding context words, their mathematical coordinates should be close together.

### 2. The Orthogonality Catastrophe of One-Hot Vectors
Before dense embeddings, early computational linguistics relied entirely on one-hot vectors $\mathbf{e}_i$. 

This caused a fatal mathematical flaw known as the **Orthogonality Catastrophe**:

Take any two distinct one-hot words $\mathbf{e}_i$ and $\mathbf{e}_j$ where $i \neq j$. If you compute their inner dot product:

$$
\mathbf{e}_i \cdot \mathbf{e}_j = \sum_{k=1}^{|V|} (\mathbf{e}_i)_k (\mathbf{e}_j)_k = 0
$$

In linear algebra, when the dot product between two non-zero vectors is zero, the vectors are **perpendicular (orthogonal)**.

In one-hot space, every single word is 90 degrees away from every other word!
- Distance from <kbd>"puppy"</kbd> to <kbd>"dog"</kbd>: $\sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1.414$
- Distance from <kbd>"puppy"</kbd> to <kbd>"nuclear reactor"</kbd>: $\sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1.414$

The computer was completely blind to meaning: it considered a puppy just as unrelated to a dog as it was to a submarine or a sandwich!

### 3. The Dense Vector Revolution: Word2Vec (2013)
In 2013, Tomas Mikolov and his team at Google introduced **Word2Vec**. Instead of huge orthogonal one-hot vectors, they trained a neural network to compress words into small, continuous dense vectors ($d = 300$).

When researchers plotted these vectors, something astonishing emerged: the geometry had spontaneously organized semantic concepts into parallel lines!

The most celebrated discovery was vector analogy arithmetic:

$$
\mathbf{x}_{\text{king}} - \mathbf{x}_{\text{man}} + \mathbf{x}_{\text{woman}} \approx \mathbf{x}_{\text{queen}}
$$

Subtracting the vector for <kbd>"man"</kbd> from <kbd>"king"</kbd> stripped away the "male" concept, leaving behind pure "royalty". Adding <kbd>"woman"</kbd> yielded the exact coordinates of <kbd>"queen"</kbd>!

In modern Transformers, the embedding layer $\mathbf{E}$ is learned directly from training data through backpropagation, creating a rich geometric universe of human concepts.

---

<h2 id="step-5">Step 5: Concrete Toy Example</h2>

> [!EXAMPLE] Concrete Toy Example: The 2D Pet-and-Vehicle Playground
> Let's perform the exact vector math by hand using pencil-and-paper numbers.

### The Setup
- **Vocabulary Set ($|V| = 4$)**:
  1. $w_1 = \text{"cat"}$
  2. $w_2 = \text{"kitten"}$
  3. $w_3 = \text{"dog"}$
  4. $w_4 = \text{"airplane"}$

- **Embedding Dimension ($d = 2$)**:
  - **Axis 1 ($x_1$) &mdash; "Furriness / Cuddliness"**: Measures how soft, furry, and pet-like the concept is ($0.0 = \text{cold metal}, 1.0 = \text{maximum fluff}$).
  - **Axis 2 ($x_2$) &mdash; "Physical Size"**: Measures how large the physical object is ($0.0 = \text{fits in palm}, 1.0 = \text{giant vehicle}$).

### The Embedding Matrix ($\mathbf{E} \in \mathbb{R}^{4 \times 2}$)

$$
\mathbf{E} = \begin{bmatrix}
0.90 & 0.30 \\
0.95 & 0.15 \\
0.85 & 0.55 \\
0.02 & 0.98
\end{bmatrix} \quad \begin{matrix} 
\leftarrow \text{Row 1: "cat"} \\ 
\leftarrow \text{Row 2: "kitten"} \\ 
\leftarrow \text{Row 3: "dog"} \\ 
\leftarrow \text{Row 4: "airplane"} 
\end{matrix}
$$

---

### Step-by-Step Lookup for $w_2 = \text{"kitten"}$

The one-hot vector for <kbd>"kitten"</kbd> (index 2) is:

$$
\mathbf{e}_2^\top = \begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix}
$$

Now, multiply $\mathbf{e}_2^\top$ by $\mathbf{E}$:

$$
\mathbf{x}_{\text{kitten}}^\top = \mathbf{e}_2^\top \mathbf{E} = \begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix}
0.90 & 0.30 \\
0.95 & 0.15 \\
0.85 & 0.55 \\
0.02 & 0.98
\end{bmatrix}
$$

Let's compute each coordinate by hand:

$$
\text{Coordinate } 1 = (0 \times 0.90) + (1 \times 0.95) + (0 \times 0.85) + (0 \times 0.02) = \mathbf{0.95}
$$

$$
\text{Coordinate } 2 = (0 \times 0.30) + (1 \times 0.15) + (0 \times 0.55) + (0 \times 0.98) = \mathbf{0.15}
$$

$$
\mathbf{x}_{\text{kitten}}^\top = [0.95, 0.15]
$$

The one-hot selector perfectly extracted Row 2!

---

### Embedding Coordinates Table

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 1.2:</strong> Coordinates, visual feature meters, and physical semantic interpretation.</caption>
  <thead>
    <tr>
      <th align="left">Token ($w_i$)</th>
      <th align="center">Index</th>
      <th align="right">Furriness ($x_1$)</th>
      <th align="center">Furriness Meter</th>
      <th align="right">Size ($x_2$)</th>
      <th align="center">Size Meter</th>
      <th align="left">Semantic Profile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><kbd>"cat"</kbd></td>
      <td align="center">1</td>
      <td align="right">0.90</td>
      <td align="center"><meter min="0" max="1" value="0.90">90%</meter></td>
      <td align="right">0.30</td>
      <td align="center"><meter min="0" max="1" value="0.30">30%</meter></td>
      <td align="left">Very furry, small-to-medium pet</td>
    </tr>
    <tr>
      <td align="left"><kbd>"kitten"</kbd></td>
      <td align="center">2</td>
      <td align="right"><mark><strong>0.95</strong></mark></td>
      <td align="center"><meter min="0" max="1" value="0.95">95%</meter></td>
      <td align="right"><mark><strong>0.15</strong></mark></td>
      <td align="center"><meter min="0" max="1" value="0.15">15%</meter></td>
      <td align="left">Ultra furry, tiny palm-sized baby</td>
    </tr>
    <tr>
      <td align="left"><kbd>"dog"</kbd></td>
      <td align="center">3</td>
      <td align="right">0.85</td>
      <td align="center"><meter min="0" max="1" value="0.85">85%</meter></td>
      <td align="right">0.55</td>
      <td align="center"><meter min="0" max="1" value="0.55">55%</meter></td>
      <td align="left">Furry, medium-sized companion</td>
    </tr>
    <tr>
      <td align="left"><kbd>"airplane"</kbd></td>
      <td align="center">4</td>
      <td align="right">0.02</td>
      <td align="center"><meter min="0" max="1" value="0.02">2%</meter></td>
      <td align="right">0.98</td>
      <td align="center"><meter min="0" max="1" value="0.98">98%</meter></td>
      <td align="left">Zero fur, massive metal flying machine</td>
    </tr>
  </tbody>
</table>

---

### Step-by-Step Distance Calculation

How far apart are our words on this 2D playground map? We use the classical **Euclidean Distance Formula**:

$$
\text{Distance}(\mathbf{u}, \mathbf{v}) = \sqrt{(u_1 - v_1)^2 + (u_2 - v_2)^2}
$$

#### 1. Distance Between "kitten" and "cat":

$$
\begin{aligned}
\text{Distance}(\text{kitten}, \text{cat}) &= \sqrt{(0.95 - 0.90)^2 + (0.15 - 0.30)^2} \\
&= \sqrt{(0.05)^2 + (-0.15)^2} \\
&= \sqrt{0.0025 + 0.0225} \\
&= \sqrt{0.0250} \\
&\approx \mathbf{0.158}
\end{aligned}
$$

#### 2. Distance Between "kitten" and "airplane":

$$
\begin{aligned}
\text{Distance}(\text{kitten}, \text{airplane}) &= \sqrt{(0.95 - 0.02)^2 + (0.15 - 0.98)^2} \\
&= \sqrt{(0.93)^2 + (-0.83)^2} \\
&= \sqrt{0.8649 + 0.6889} \\
&= \sqrt{1.5538} \\
&\approx \mathbf{1.246}
\end{aligned}
$$

Let's compare the two distances:

$$
\frac{1.246}{0.158} \approx \mathbf{7.89\times \text{ farther away!}}
$$

<kbd>"airplane"</kbd> is nearly **8 times farther** from <kbd>"kitten"</kbd> than <kbd>"cat"</kbd> is!

---

### The 2D Playground Map Visualized

<figure>
<pre>
Size (Axis 2)
  ▲
  │
1.00│  ["airplane"] (0.02, 0.98)
  │     (Massive flying machine)
0.80│
  │
0.60│                                     ["dog"] (0.85, 0.55)
  │
0.40│                                     ["cat"] (0.90, 0.30)
  │                                           │ (Distance ≈ 0.158)
0.20│                                     ["kitten"] (0.95, 0.15)
  │                                       (Cuddly Pet Cluster!)
0.00└─────────────────────────────────────────────────────────────► Furriness (Axis 1)
   0.00        0.20        0.40        0.60        0.80       1.00
</pre>
<figcaption><strong>Figure 1.2:</strong> Visual 2D geometric playground map. Notice how "cat", "kitten", and "dog" form a tight pet cluster, while "airplane" is isolated in the upper left corner.</figcaption>
</figure>

---

<h2 id="step-6">Step 6: Core Takeaway</h2>

> [!TIP] Core Takeaway
> An embedding matrix $\mathbf{E}$ is the **universal translation bridge** of an LLM: it converts flat, isolated human words into living, geometric coordinates in multi-dimensional space.
> 
> By placing words with similar meanings close together, the neural network can reason about concepts, analogies, and contexts using pure linear algebra.
> 
> But this brings us to our next crucial question: **When an LLM wants to measure how closely two words are related, what exact mathematical ruler does it use to measure alignment?**

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../00-next-word-prediction/index.html">&larr; Chapter 00: The Next-Word Guessing Game</a> &nbsp;|&nbsp; 
    <a href="../index.html">Home / Curriculum Overview</a> &nbsp;|&nbsp; 
    <strong>Next Chapter:</strong> <a href="../02-dot-product-and-similarity/index.html">Chapter 02: Measuring Closeness (Dot Product &amp; Cosine Similarity) &rarr;</a>
  </p>
</nav>
