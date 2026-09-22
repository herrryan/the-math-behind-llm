# Chapter 03: The Magic Stretching Box (Matrix Multiplication)



## Step 1: 3-Year-Old Intuition {: #step-1 }

Imagine you have a giant sheet of stretchy rubber lying flat on your playroom floor. 

Drawn on this rubber sheet is a neat square grid of lines, like graph paper. In the middle of the sheet sits a tiny red toy car at the center coordinate $(0, 0)$.

<figure>
<pre>
       Original Square Grid                  Stretched Rubber Sheet
       ┌───┬───┬───┐                                 /───/───/───/
       │   │   │   │                                /   /   /   /
       ├───┼───┼───┤           Press &amp;             /───/───/───/
       │   │ ★ │   │          Stretch             /   / ★ /   /
       ├───┼───┼───┤       ────────────►         /───/───/───/
       │   │   │   │                            /   /   /   /
       └───┴───┴───┘                           /───/───/───/
</pre>
<figcaption><strong>Figure 3.1:</strong> Pulling and stretching a rubber sheet transforms squares into tilted parallelograms while keeping lines straight and parallel.</figcaption>
</figure>

Now, imagine you grab the corners of the rubber sheet and pull:
- You pull the right side out to make it twice as wide (**stretching**).
- You twist your hands to tilt the sheet sideways (**rotating and shearing**).
- You push down from the top to squash it flatter (**squishing**).

Notice three golden rules of this stretching game:

1. **The Origin Stays Pinned**: The center dot $(0, 0)$ where the red car sits never moves. It is nailed to the floor.
2. **Straight Lines Stay Straight**: No matter how hard you stretch or twist the sheet, the grid lines never bend, curve, or turn into wobbly squiggles.
3. **Parallel Lines Stay Parallel**: Lines that started out running alongside each other still run alongside each other after the stretch.

In mathematics, this rubber-sheet stretching game has a formal name: a **Linear Transformation**. 

A **Matrix** is nothing more than a tiny instruction booklet that tells the computer exactly how much the rubber sheet was pulled, twisted, and squashed.

---

## Step 2: The Bridging Question {: #step-2 }

In Chapter 01 and Chapter 02, we learned that words live as stationary arrows (vectors) on a concept map.

For example, the word <kbd>"bank"</kbd> enters the model with a fixed starting coordinate. But in human language, words do not mean the same thing in every sentence:
- In *"river bank"*, the word needs to move toward water, nature, and geography.
- In *"bank deposit"*, the word needs to move toward money, finance, and vaults.

A static dictionary of embeddings is not yet an AI brain. Words cannot remain frozen in place!

This brings us to two fundamental questions:
1. **What is a "Weight" ($\mathbf{W}$), and how is it different from the "Embeddings" ($\mathbf{x}$) we built in Chapter 01?**
2. **How does an artificial neural network move, rotate, and reshape thousands of word embeddings simultaneously using simple arithmetic?**

---

## Step 3: The Exact Math & Formula {: #step-3 }

Every layer of a modern neural network reshapes vector spaces using the fundamental equation of deep learning: the <dfn id="def-affine-transformation"><strong>Affine Linear Transformation</strong></dfn>.

---

### 1. From Embeddings to Weights: What Are Weights?

Before writing the linear equation, let's establish the fundamental distinction between two types of numbers in <abbr title="Artificial Intelligence">AI</abbr>:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 3.1:</strong> The Fundamental Divide: Embeddings vs. Weights</caption>
  <thead>
    <tr bgcolor="#eef2f7">
      <th scope="col" align="left" width="18%">Concept</th>
      <th scope="col" align="left" width="32%">Embedding Vector ($\mathbf{x}$)</th>
      <th scope="col" align="left" width="32%">Weight Matrix ($\mathbf{W}$)</th>
      <th scope="col" align="left" width="18%">Tangible Analogy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left" bgcolor="#f8f9fa"><strong>What is it?</strong></th>
      <td>The <strong>Input Data</strong>: the numerical coordinates of a specific word or token.</td>
      <td>The <strong>Processing Engine</strong>: the learned rules that transform and manipulate data.</td>
      <td>Dough vs. Pasta Maker</td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="left" bgcolor="#f8f9fa"><strong>Where does it come from?</strong></th>
      <td>Retrieved dynamically from the vocabulary table whenever a user enters a word.</td>
      <td>Trained across billions of text examples and <em>frozen in memory</em> during inference.</td>
      <td>Object vs. Optical Lens</td>
    </tr>
    <tr>
      <th scope="row" align="left" bgcolor="#f8f9fa"><strong>Does it change?</strong></th>
      <td>Yes! Every token in a sentence brings a different embedding vector into the model.</td>
      <td>No! The weights stay identical regardless of which sentence is being read.</td>
      <td>Passenger vs. Train Track</td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="left" bgcolor="#f8f9fa"><strong>Grammar Role</strong></th>
      <td>The <strong>Noun</strong>: the subject being acted upon.</td>
      <td>The <strong>Verb</strong>: the action, lens, or operation performed upon the subject.</td>
      <td>Actor vs. Director</td>
    </tr>
  </tbody>
</table>

<fieldset>
<legend><strong>Connecting Back to Chapter 01: You Already Met a Weight Matrix!</strong></legend>
In Chapter 01, we introduced the <strong>Embedding Matrix</strong> $\mathbf{E} \in \mathbb{R}^{|V| \times d}$. 
Notice something profound: $\mathbf{E}$ was actually our very first matrix of weights! It stored $|V|$ vectors, translating a discrete word ID (a one-hot vector) into continuous coordinates:



$$
\mathbf{x}_i^\top = \mathbf{e}_i^\top \mathbf{E}
$$



Now, a layer's <strong>Weight Matrix</strong> $\mathbf{W}$ continues that journey: it takes an existing word embedding $\mathbf{x}$ and translates it into an entirely new concept space. 

If the embedding matrix $\mathbf{E}$ is the <strong>dictionary</strong> that gives words their initial definitions, the weight matrix $\mathbf{W}$ is the <strong>thinking lens</strong> that interprets how those words interact.
</fieldset>

---

### 2. The Core Transformation Formula

For an input embedding vector $\mathbf{x} \in \mathbb{R}^{k \times 1}$, a weight matrix $\mathbf{W} \in \mathbb{R}^{m \times k}$, and a bias vector $\mathbf{b} \in \mathbb{R}^{m \times 1}$, the transformed output vector $\mathbf{y} \in \mathbb{R}^{m \times 1}$ is:



$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$



Let's inspect every piece of this equation:



$$
\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix} = \begin{bmatrix} W_{1,1} & W_{1,2} & \cdots & W_{1,k} \\ W_{2,1} & W_{2,2} & \cdots & W_{2,k} \\ \vdots & \vdots & \ddots & \vdots \\ W_{m,1} & W_{m,2} & \cdots & W_{m,k} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_k \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
$$



- $\mathbf{W}\mathbf{x}$ performs the **rubber-sheet transformation**: stretching, rotating, shearing, or changing the dimension of the embedding space.
- $+\, \mathbf{b}$ performs a **rigid translation**: sliding the entire transformed coordinate system across space without changing its shape.

---

### 3. The Inner Dimension Compatibility Rule

Two matrices can be multiplied **if and only if** the number of columns in the first matrix equals the number of rows in the second matrix.

<figure>
<pre>
             Matrix A                   Matrix B                  Output C
      ┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
    m │                    │   k │                    │   m │                    │
      │                    │  ───│                    │  ───│                    │
    ▼ │                    │   ▼ │                    │   ▼ │                    │
      └────────────────────┘     └────────────────────┘     └────────────────────┘
         ◄─────── k ───────►        ◄─────── n ───────►        ◄─────── n ───────►
                   ▲                          ▲
                   └──── MUST MATCH EXACTLY ──┘
                             (k == k)
</pre>
<figcaption><strong>Figure 3.2:</strong> The inner dimension matching constraint. The inner dimension $k$ collapses during multiplication, leaving an $(m \times n)$ output matrix.</figcaption>
</figure>

Written formally:



$$
(m \times k) \times (k \times n) \longrightarrow (m \times n)
$$



If matrix $\mathbf{A} \in \mathbb{R}^{m \times k}$ and matrix $\mathbf{B} \in \mathbb{R}^{k \times n}$, each entry $C_{i, j}$ in the resulting matrix $\mathbf{C} \in \mathbb{R}^{m \times n}$ is computed by taking the **dot product** between row $i$ of matrix $\mathbf{A}$ and column $j$ of matrix $\mathbf{B}$:



$$
C_{i, j} = \sum_{r=1}^k A_{i, r} B_{r, j} = A_{i, 1} B_{1, j} + A_{i, 2} B_{2, j} + \dots + A_{i, k} B_{k, j}
$$



<fieldset>
<legend><strong>Matrix Multiplication Is Just a Grid of Dot Products!</strong></legend>
Notice the deep connection to Chapter 02:
Every single cell in the product matrix $\mathbf{C}$ is simply the <strong>dot product</strong> between a horizontal row from the left matrix and a vertical column from the right matrix.

Matrix multiplication is a structured, parallel factory for computing thousands of dot products in one single sweep.
</fieldset>

---

### 4. The Geometric Secret: Where Do the Basis Vectors Land?

Why should someone studying neural language models care about "where basis vectors land"?

When developers first learn matrix multiplication, they are taught the row-by-column dot product formula:



$$
y_i = (\text{Row } i \text{ of } \mathbf{W}) \cdot \mathbf{x}
$$



While this explains how silicon hardware computes numbers, it leaves our mental model of the neural network completely blind. It makes a linear layer look like a dry, disconnected grid of arithmetic.

Viewing matrix multiplication through its **columns** provides the direct mechanical blueprint for how an AI transforms linguistic features.

#### What Is a "Basis Vector" in Terms of Embeddings?

Recall from Chapter 01 that every word embedding $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$ is composed of coordinates representing semantic traits (such as $x_1 = 2$ for furriness and $x_2 = 1$ for playfulness).

The standard basis vectors are simply the purest, most extreme embeddings possible in that space:
- $\hat{\mathbf{i}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ represents an embedding containing **100% pure Feature 1** (e.g., pure *"Feline Nature"*) and 0% of anything else.
- $\hat{\mathbf{j}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ represents an embedding containing **100% pure Feature 2** (e.g., pure *"Playfulness"*) and 0% of anything else.

Every real word embedding is just a recipe of these pure ingredients: $\mathbf{x} = x_1 \hat{\mathbf{i}} + x_2 \hat{\mathbf{j}}$.

#### The Matrix Columns as a Feature Translation Dictionary

Now, watch what happens when our weight matrix lens $\mathbf{W} = \begin{bmatrix} W_{1,1} & W_{1,2} \\ W_{2,1} & W_{2,2} \end{bmatrix}$ acts on these pure basis embeddings:



$$
\mathbf{W} \hat{\mathbf{i}} = \begin{bmatrix} W_{1,1} & W_{1,2} \\ W_{2,1} & W_{2,2} \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} W_{1,1} \\ W_{2,1} \end{bmatrix} = \text{Column 1 of } \mathbf{W}
$$





$$
\mathbf{W} \hat{\mathbf{j}} = \begin{bmatrix} W_{1,1} & W_{1,2} \\ W_{2,1} & W_{2,2} \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} W_{1,2} \\ W_{2,2} \end{bmatrix} = \text{Column 2 of } \mathbf{W}
$$



This reveals the foundational insight of neural transformations:

> **The columns of a weight matrix are simply the landing coordinates of pure basis features after being transformed!**

Each column of $\mathbf{W}$ acts as a **semantic dictionary entry**:
- **Column 1 ($\mathbf{w}_{:, 1}$)** answers: *"If an input word possesses 1 unit of Feature 1 (Feline Nature), what new downstream traits should this layer produce?"*
- **Column 2 ($\mathbf{w}_{:, 2}$)** answers: *"If an input word possesses 1 unit of Feature 2 (Playfulness), what new downstream traits should this layer produce?"*

#### Matrix Multiplication as a "Concept Recipe Blender"

When a real word embedding $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$ (such as $\text{"cat"} = [2, 1]^\top$) enters the neural network layer, the matrix multiplication computes:



$$
\mathbf{W}\mathbf{x} = x_1 (\text{Column 1 of } \mathbf{W}) + x_2 (\text{Column 2 of } \mathbf{W})
$$



The embedding vector $\mathbf{x}$ is not an abstract math puzzle—it is a **blender recipe**:
> *"Take $x_1$ scoops of Column 1's concept, and $x_2$ scoops of Column 2's concept, and blend them together into the output representation!"*

<fieldset>
<legend><strong>Why This Basis-Column Intuition Unlocks Future Chapters</strong></legend>
This column perspective is the key to understanding how deeper neural architectures work:
<ol>
  <li><strong>Attention Projections ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$) in Module 3</strong>: When a word embedding is transformed into a "Query", the columns of $\mathbf{W}_Q$ define the coordinate axes of the search space—translating raw word traits into <em>"What questions is this word actively asking about its surrounding context?"</em></li>
  <li><strong>Feed-Forward Memory Networks (FFN) in Module 5</strong>: In models like LLaMA and GPT, intermediate layers expand embeddings into thousands of dimensions and project them back down. The columns of these projection matrices act as key-value memory slots storing factual knowledge.</li>
  <li><strong>Mechanistic Interpretability</strong>: When AI safety researchers peer inside a trained model to find "honesty vectors" or "refusal directions", they are directly analyzing the directions formed by linear combinations of these matrix columns.</li>
</ol>
</fieldset>

---

### 5. Deep Learning Convention: Row Vectors and Batches

In mathematics textbooks, vectors are traditionally written as vertical columns: $\mathbf{y} = \mathbf{W}\mathbf{x}$.

However, in deep learning software (PyTorch, JAX, Hugging Face), text is processed in **batches of tokens**, where each token is represented as a **horizontal row vector**.

For a sequence of $T$ tokens, each with dimension $d_{\text{in}}$:
- The input token matrix is $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{in}}}$ (each row is one token embedding).
- The weight matrix is $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$.
- The bias vector is $\mathbf{b} \in \mathbb{R}^{1 \times d_{\text{out}}}$ (broadcasted across all $T$ rows).

The forward pass is written with the input on the left:



$$
\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$



Dimension check:



$$
(T \times d_{\text{in}}) \times (d_{\text{in}} \times d_{\text{out}}) \longrightarrow (T \times d_{\text{out}})
$$



Both representations are mathematically equivalent under the transpose identity: $(\mathbf{W}\mathbf{x})^\top = \mathbf{x}^\top \mathbf{W}^\top$.

---

### Mathematical Notation Catalog

<details name="ch03-deep-dives">
<summary><strong>Click to expand: Complete Mathematical Notation Catalog</strong></summary>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 3.2:</strong> Formal Notation Catalog for Chapter 03</caption>
  <thead>
    <tr bgcolor="#eef2f7">
      <th scope="col" align="center">Symbol</th>
      <th scope="col" align="left">Mathematical Name</th>
      <th scope="col" align="center">Standard Shape</th>
      <th scope="col" align="left">Physical Meaning in an LLM</th>
      <th scope="col" align="left">Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="center" bgcolor="#f8f9fa">$\mathbf{W}$</th>
      <td>Weight Matrix</td>
      <td>$\mathbb{R}^{m \times k}$ (or $\mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$)</td>
      <td>The learnable linear transformation parameters</td>
      <td>Feed-forward or projection weights</td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="center" bgcolor="#f8f9fa">$\mathbf{x}$</th>
      <td>Input Vector</td>
      <td>$\mathbb{R}^{k \times 1}$</td>
      <td>Single token embedding before transformation</td>
      <td>$\mathbf{x} \in \mathbb{R}^{4096 \times 1}$ (Llama 3)</td>
    </tr>
    <tr>
      <th scope="row" align="center" bgcolor="#f8f9fa">$\mathbf{X}$</th>
      <td>Batch Input Matrix</td>
      <td>$\mathbb{R}^{T \times d_{\text{in}}}$</td>
      <td>Sequence of $T$ token vectors stacked as rows</td>
      <td>$2048 \text{ tokens} \times 4096 \text{ dims}$</td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="center" bgcolor="#f8f9fa">$\mathbf{b}$</th>
      <td>Bias Vector</td>
      <td>$\mathbb{R}^{m \times 1}$</td>
      <td>Constant offset shifting the origin of coordinates</td>
      <td>Added to every transformed point</td>
    </tr>
    <tr>
      <th scope="row" align="center" bgcolor="#f8f9fa">$\mathbf{y}, \mathbf{Y}$</th>
      <td>Output Representation</td>
      <td>$\mathbb{R}^{m \times 1}$ (or $\mathbb{R}^{T \times d_{\text{out}}}$)</td>
      <td>Reshaped features ready for the next layer</td>
      <td>Updated contextual hidden states</td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="center" bgcolor="#f8f9fa">$\hat{\mathbf{i}}, \hat{\mathbf{j}}$</th>
      <td>Basis Vectors</td>
      <td>$\mathbb{R}^{d \times 1}$</td>
      <td>Unit coordinate axes defining standard orientation</td>
      <td>$[1, 0]^\top, [0, 1]^\top$</td>
    </tr>
  </tbody>
</table>

<dl>
  <dt><dfn id="def-linear-transformation"><strong>Linear Transformation</strong></dfn></dt>
  <dd>A mathematical mapping $T(\mathbf{x})$ between vector spaces that satisfies two properties: additivity $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ and scalar homogeneity $T(c\mathbf{u}) = cT(\mathbf{u})$.</dd>
  
  <dt><dfn id="def-affine-map"><strong>Affine Transformation</strong></dfn></dt>
  <dd>A linear transformation followed by a vector translation: $f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$. While a strict linear map must fix the origin at zero, an affine map can slide the origin anywhere in space.</dd>

  <dt><dfn id="def-gemm"><strong><abbr title="General Matrix Multiply">GEMM</abbr> (General Matrix Multiply)</strong></dfn></dt>
  <dd>The standard high-performance computing primitive computing $\mathbf{C} \leftarrow \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}$. In <abbr title="Basic Linear Algebra Subprograms">BLAS</abbr> libraries, it is the most heavily optimized computational kernel in all of modern machine learning.</dd>
</dl>
</details>

---

## Step 4: Where Did It Come From? {: #step-4 }

### 1. The Historical Origin

<dl>
  <dt><time datetime="1858">1858</time> &mdash; <strong>Arthur Cayley: Linear Substitutions</strong></dt>
  <dd>Matrix algebra was formalized by British mathematician Arthur Cayley in his landmark treatise <cite>A Memoir on the Theory of Matrices</cite>. Cayley did not invent matrices to store tables of data; he invented them specifically to represent compositions of linear substitutions in systems of equations.</dd>

  <dt><time datetime="1958">1958</time> &mdash; <strong>Frank Rosenblatt: Synaptic Weight Multiplication</strong></dt>
  <dd>In the Perceptron, early neural network pioneers adopted matrix-vector dot products to model how multiple biological dendrite inputs combine into a single neuron's activation threshold.</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Vaswani et al.: Massively Parallel Attention Projections</strong></dt>
  <dd>In <cite>Attention Is All You Need</cite>, recurrent sequential steps were replaced entirely by massive, parallel matrix projections ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$), leveraging the full parallel compute bandwidth of modern hardware.</dd>
</dl>

---

### 2. Why Linear Transformations? Two Irreplaceable Superpowers

Why does every deep learning architecture—from the oldest Multilayer Perceptron to modern Transformers—rely on matrix multiplication?

#### Superpower 1: Exact, Smooth Differentiability
When training an <abbr title="Large Language Model">LLM</abbr> with billions of parameters, we must calculate how changing every single weight $W_{i, j}$ affects the final prediction loss $\mathcal{L}$.

Because matrix multiplication is purely linear additions and multiplications, its derivative is astonishingly simple:



$$
y_i = \sum_{r=1}^k W_{i, r} x_r + b_i \implies \frac{\partial y_i}{\partial W_{i, j}} = x_j
$$



The rate of change with respect to weight $W_{i, j}$ is simply the input activation $x_j$! This enables the **Backpropagation algorithm** to update billions of parameters simultaneously without solving complex equations.

#### Superpower 2: Preserving Geometric Structure
Linear transformations preserve collinearity and parallel lines. If three word vectors form an analogy in input space:



$$
\mathbf{x}_{\text{king}} - \mathbf{x}_{\text{man}} + \mathbf{x}_{\text{woman}} \approx \mathbf{x}_{\text{queen}}
$$



Applying a linear transformation $\mathbf{W}$ preserves this exact relationship:



$$
\mathbf{W}(\mathbf{x}_{\text{king}} - \mathbf{x}_{\text{man}} + \mathbf{x}_{\text{woman}}) = \mathbf{W}\mathbf{x}_{\text{king}} - \mathbf{W}\mathbf{x}_{\text{man}} + \mathbf{W}\mathbf{x}_{\text{woman}} \approx \mathbf{W}\mathbf{x}_{\text{queen}}
$$



The model can rotate, project, and stretch concepts into new sub-spaces without tearing apart semantic analogies.

---

### 3. The Silicon Secret: Why GPUs Are Matrix Multiplication Monsters

<details name="ch03-deep-dives">
<summary><strong>Hardware Deep-Dive: Systolic Arrays and Arithmetic Intensity</strong></summary>

<p>Consider how much work a computer must do to multiply two $(N \times N)$ matrices:</p>
<ul>
  <li><strong>Data to store in memory</strong>: $2 \times N^2$ numbers (Matrices $\mathbf{A}$ and $\mathbf{B}$).</li>
  <li><strong>Math operations to compute</strong>: $2 \times N^3$ arithmetic operations (multiplications and additions).</li>
</ul>

<p>Notice the ratio:</p>



$$
\frac{\text{Operations}}{\text{Memory Transfers}} = \frac{O(N^3)}{O(N^2)} = O(N)
$$



<p>This is the holy grail of computer architecture, known as <strong>high arithmetic intensity</strong>! For large matrices, a <abbr title="Graphics Processing Unit">GPU</abbr> loads a number from memory once and reuses it hundreds of times across different dot products.</p>

<p>Modern <abbr title="Artificial Intelligence">AI</abbr> accelerators (such as NVIDIA H100 <abbr title="Graphics Processing Unit">GPUs</abbr> or Google <abbr title="Tensor Processing Unit">TPUs</abbr>) use <strong>Systolic Arrays</strong> of Tensor Cores. Data flows rhythmically through a physical 2D grid of silicon multipliers like blood pumping through a heart, executing trillions of matrix operations per second without waiting for slow memory transfers.</p>
</details>

---

## Step 5: Concrete Toy Example {: #step-5 }

<fieldset>
  <legend><strong>Numerical Execution Pipeline Checklist</strong></legend>
  <p><input type="checkbox" checked disabled> <strong>Step 5.1:</strong> Verify dimension compatibility: $(2 \times 2) \times (2 \times 1) \longrightarrow (2 \times 1)$</p>
  <p><input type="checkbox" checked disabled> <strong>Step 5.2:</strong> Compute raw linear matrix-vector projection $\mathbf{W}\mathbf{x} = [5, 3]^\top$ via parallel dot products</p>
  <p><input type="checkbox" checked disabled> <strong>Step 5.3:</strong> Apply bias translation $\mathbf{b} = [1, -1]^\top \longrightarrow \mathbf{y} = [6, 2]^\top$</p>
  <p><input type="checkbox" checked disabled> <strong>Step 5.4:</strong> Inspect transformed basis axes $\mathbf{W}\hat{\mathbf{i}} = [2, 0]^\top$ and $\mathbf{W}\hat{\mathbf{j}} = [1, 3]^\top$</p>
</fieldset>

Let's step through an exact numerical example by hand.

We have an input word vector for <kbd>"cat"</kbd> in 2-dimensional space:
- Axis 1: **Furriness** $= 2$
- Axis 2: **Playfulness** $= 1$



$$
\mathbf{x}_{\text{cat}} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$



We want to project this word through a weight matrix $\mathbf{W}$ that stretches furriness and shears playfulness, followed by a bias shift $\mathbf{b}$:



$$
\mathbf{W} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$



Let's compute the output vector $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ step by step.

---

### Step 5.1: Verify Dimension Compatibility

Before doing any math, always check the shapes:
- Weight matrix $\mathbf{W}$: shape is $(2 \times 2)$, so $m = 2, k = 2$.
- Input vector $\mathbf{x}$: shape is $(2 \times 1)$, so $k = 2, n = 1$.
- Inner dimensions match: $k = 2 = 2$.
- Output shape will be: $(m \times n) = (2 \times 1)$.

Dimensions are 100% compatible.

---

### Step 5.2: Multiply Matrix $\mathbf{W}$ by Vector $\mathbf{x}$

Compute each row coordinate using the dot product formula:



$$
\begin{aligned}
y_1^{\text{raw}} &= (\text{Row 1 of } \mathbf{W}) \cdot \mathbf{x} \\
&= (W_{1,1} \times x_1) + (W_{1,2} \times x_2) \\
&= (2 \times 2) + (1 \times 1) \\
&= 4 + 1 \\
&= \mathbf{5}
\end{aligned}
$$





$$
\begin{aligned}
y_2^{\text{raw}} &= (\text{Row 2 of } \mathbf{W}) \cdot \mathbf{x} \\
&= (W_{2,1} \times x_1) + (W_{2,2} \times x_2) \\
&= (0 \times 2) + (3 \times 1) \\
&= 0 + 3 \\
&= \mathbf{3}
\end{aligned}
$$



So the raw linear projection is:



$$
\mathbf{W}\mathbf{x} = \begin{bmatrix} 5 \\ 3 \end{bmatrix}
$$



---

### Step 5.3: Add the Bias Vector $\mathbf{b}$

Now add the constant translation vector $\mathbf{b} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$:



$$
\mathbf{y} = \begin{bmatrix} 5 \\ 3 \end{bmatrix} + \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 5 + 1 \\ 3 + (-1) \end{bmatrix} = \begin{bmatrix} \mathbf{6} \\ \mathbf{2} \end{bmatrix}
$$



Our final transformed vector is:



$$
\mathbf{y}_{\text{cat}} = \begin{bmatrix} \mathbf{6} \\ \mathbf{2} \end{bmatrix}
$$



<p>
  <strong>Transformed Coordinates:</strong>
  Dimension 1: <mark><strong>6.0</strong></mark> &nbsp;|&nbsp; 
  Dimension 2: <mark><strong>2.0</strong></mark>
</p>

---

### Step 5.4: Geometric Basis Inspection

Let's verify where our standard basis axes landed during this transformation:

1. **Horizontal Basis Vector $\hat{\mathbf{i}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$**:


   $$
   \mathbf{W}\hat{\mathbf{i}} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}
   $$


   The unit horizontal axis was stretched by a factor of $2\times$ along the horizontal direction!

2. **Vertical Basis Vector $\hat{\mathbf{j}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$**:


   $$
   \mathbf{W}\hat{\mathbf{j}} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}
   $$


   The unit vertical axis was tilted rightward by $+1$ unit and stretched vertically by a factor of $3\times$!

<figure>
<pre>
       Original Basis Vectors                   Transformed Basis Vectors
            ▲                                        ▲
            │                                    3.00│       • W(j) = (1, 3)
        1.00│   • j = (0, 1)                         │      /
            │   │                                    │     /
            │   └───► i = (1, 0)                     │    /
        0.00└───────┴────────►                   0.00└───┴───► W(i) = (2, 0)
          0.00    1.00                             0.00 1.00 2.00
</pre>
<figcaption><strong>Figure 3.3:</strong> The transformation reshaped the original unit square into a tilted, stretched parallelogram whose edges are defined by the columns of matrix W.</figcaption>
</figure>

Reconstructing $\mathbf{W}\mathbf{x}$ directly using our basis arrows:



$$
\mathbf{W}\mathbf{x} = 2 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 1 \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 4 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 5 \\ 3 \end{bmatrix}
$$



The basis vector perspective yields the exact same answer as the row-column dot product!

---

## Step 6: Core Takeaway {: #step-6 }

!!! tip "Key Insight: Core Takeaway"
    A **weight matrix** is a programmable rubber sheet that stretches, tilts, and projects semantic vectors across multi-dimensional space, while the **bias vector** slides the coordinate frame to the optimal origin.

    Matrix multiplication is the computational engine of LLMs because it combines smooth mathematical differentiability with extreme hardware efficiency on GPU tensor cores.

    But here is a profound dilemma: **What happens if you stack two, three, or one hundred stretching sheets on top of each other?** As we will discover in Chapter 04, pure matrix multiplications collapse into a single flat stretch! To build deep intelligence, we need **Activation Functions: The One-Way Gate**.

---

