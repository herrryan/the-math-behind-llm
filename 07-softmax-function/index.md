# Chapter 07: The Fair Voting Booth (The Softmax Function)

---

## Step 1: 3-Year-Old Intuition (The Shouting Match & The Pizza Slices)

> [!INTUITION] The Shouting Match & The Pizza Slices
> Imagine three children sitting at a play table, shouting out what they want on tonight's dinner pizza:
>
> - Child 1 whispers quietly: <kbd>"Mushrooms!"</kbd> (Volume score: 4)
> - Child 2 shouts at the top of their lungs: <kbd>"Pepperoni!"</kbd> (Volume score: 7)
> - Child 3 speaks in a normal voice: <kbd>"Cheese!"</kbd> (Volume score: 6)
>
> If you just listen to the raw volume scores $[4, 7, 6]$, how do you slice a single pizza fairly?
> 
> You cannot cut a pizza into "volume units." A pizza is one whole thing (100%). Slices must always add up to exactly the whole pizza. Furthermore, no slice can be negative (you cannot take away a slice someone already ate!), and no slice can be bigger than the whole pizza.
>
> What happens if a fourth child is grumpy and grumbles with a negative volume score: $-3$? You certainly cannot give them "negative pizza slices"!
>
> The wise pizza chef invents a **Fair Voting Booth**:
>
> 1. **Turn Every Grumble Positive**: The chef puts each volume score through a magical magnifier machine ($e^z$). Even negative grumbles turn into tiny, positive sprinkles of flour ($e^{-3} \approx 0.05$), while loud shouts turn into giant heaps of toppings ($e^7 \approx 1097$). Nothing is ever negative!
> 2. **Measure the Whole Pile**: The chef dumps all the toppings into one giant bowl and weighs the total sum ($1555$ grams).
> 3. **Hand Out the Slices**: The chef gives each child a slice equal to their personal topping weight divided by the giant bowl's total weight.
>
> Instantly, everyone gets a slice between $0\%$ and $100\%$, and all slices together add up to **exactly 100% of the pizza**!

<figure>
<pre>
[Raw Shouts: Logits z]     [Magical Magnifier: e^z]     [Pizza Share: Softmax]
Child 1 ("Mushrooms"): 4  ─────────►  54.6 grams   ────────►   3.5% of pizza
Child 2 ("Pepperoni"): 7  ─────────► 1096.6 grams  ────────►  70.5% of pizza (Winner!)
Child 3 ("Cheese"):    6  ─────────►  403.4 grams  ────────►  26.0% of pizza
─────────────────────────────────────────────────────────────────────────────
Total Sum:                         1554.6 grams  ────────► 100.0% of pizza
</pre>
<figcaption><strong>Figure 7.1:</strong> Converting raw, unbounded volume scores into non-negative pizza slices that sum to exactly 100%.</figcaption>
</figure>

---

## Step 2: The Bridging Question

> [!BRIDGING] From Unbounded Scores to Probability Distributions
> In Chapter 06, we watched the ambiguous word <kbd>"bank"</kbd> fire its Query arrow against the Key arrows of <kbd>"The"</kbd>, <kbd>"river"</kbd>, and <kbd>"bank"</kbd>. The linear dot products produced raw affinity scores:
>
> $$\mathbf{z} = [4.0, 7.0, 6.0]$$
>
> But these raw dot products (called **logits**) are mathematically wild:
> 1. They can be any real number from $-\infty$ to $+\infty$.
> 2. They do not sum to $1$.
> 3. They can easily be negative if two vectors point in opposite directions.
>
> To blend the meaning of words together in Attention (and to pick the final next word out of a 100,000-word dictionary), the Transformer must answer:
>
> *"How do we convert an arbitrary list of real numbers—positive, negative, or zero—into a strictly valid probability distribution where every value is positive and the sum is strictly 100%, without destroying the relative ranking of the candidates?"*

---

## Step 3: The Exact Math & Formula

### 1. The Softmax Function

Let $\mathbf{z} = [z_1, z_2, \dots, z_N]^\top \in \mathbb{R}^N$ be an $N$-dimensional vector of raw scores (<dfn id="def-logit">logits</dfn>).

The <dfn id="def-softmax">Softmax function</dfn> transforms $\mathbf{z}$ into a probability distribution $\mathbf{s} = \operatorname{softmax}(\mathbf{z}) \in \mathbb{R}^N$:

$$
\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 7.1:</strong> Complete Mathematical Symbol Definitions for Softmax</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="22%">Symbol</th>
      <th scope="col" align="left" width="22%">Type &amp; Domain</th>
      <th scope="col" align="left" width="56%">Mathematical Meaning &amp; Physical Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left">$\mathbf{z}$</th>
      <td align="left">Vector $\in \mathbb{R}^N$</td>
      <td>The raw, unnormalized score vector (logits). In Attention, $z_j = \mathbf{q}_i^\top \mathbf{k}_j$.</td>
    </tr>
    <tr>
      <th scope="row" align="left">$z_i$</th>
      <td align="left">Scalar $\in (-\infty, +\infty)$</td>
      <td>The individual raw score for candidate $i$. Can be positive, zero, or negative.</td>
    </tr>
    <tr>
      <th scope="row" align="left">$e$</th>
      <td align="left">Constant $\approx 2.71828$</td>
      <td>Euler's number, base of the natural exponential function.</td>
    </tr>
    <tr>
      <th scope="row" align="left">$e^{z_i}$</th>
      <td align="left">Scalar $\in (0, +\infty)$</td>
      <td>The exponentiated score. Guaranteed strictly positive for every real number $z_i$.</td>
    </tr>
    <tr>
      <th scope="row" align="left">$\sum_{j=1}^N e^{z_j}$</th>
      <td align="left">Scalar $\in (0, +\infty)$</td>
      <td>The <strong>normalizing denominator</strong> (known in physics as the <em>Partition Function</em> $Z$).</td>
    </tr>
    <tr>
      <th scope="row" align="left">$\operatorname{softmax}(\mathbf{z})_i$</th>
      <td align="left">Scalar $\in (0, 1)$</td>
      <td>The normalized probability for candidate $i$, satisfying $\sum_{i=1}^N \operatorname{softmax}(\mathbf{z})_i = 1.0$.</td>
    </tr>
  </tbody>
</table>

---

### 2. The Dependency Ladder: Why Exponentiate?

Why did mathematicians choose $e^z$ instead of simpler normalization tricks? Let us walk up the ladder of failed alternatives:

<figure>
<pre>
Attempt 1: Linear Normalization
           p_i = z_i / sum(z_j)
           Fatal Flaw: If z = [2, -3, 1], sum = 0 (Division by zero!).
                       If z = [2, -1, 3], p_2 = -1/4 = -25% (Negative probability!).

Attempt 2: Absolute Value Normalization
           p_i = |z_i| / sum(|z_j|)
           Fatal Flaw: If z_1 = -10 (terrible fit) and z_2 = +10 (great fit),
                       both get |-10| = |+10| = 10. Destroys all order!

Attempt 3: Rectified Linear (ReLU) Normalization
           p_i = max(0, z_i) / sum(max(0, z_j))
           Fatal Flaw: Any negative logit is instantly crushed to 0.
                       Zero gradient flows backwards! The network cannot learn why it failed.

The Winner: The Exponential Function e^z
           1. Strictly positive: e^z &gt; 0 for all real z in (-inf, +inf).
           2. Monotonically increasing: z_a &gt; z_b &lt;==&gt; e^{z_a} &gt; e^{z_b}. Order is preserved.
           3. Smoothly differentiable: d/dz (e^z) = e^z. Perfect for gradient descent!
           4. Winner-take-most: Magnifies high scores while gently suppressing low ones.
</pre>
<figcaption><strong>Figure 7.2:</strong> The mathematical progression leading uniquely to the exponential function.</figcaption>
</figure>

---

### 3. The Numerical Stability Guard: Shift Invariance

If a neural network outputs large logits like $z_i = 1000$, computers crash. In 32-bit floating point arithmetic (<abbr title="IEEE 754 32-bit Single-Precision Float">FP32</abbr>), numbers above $e^{88.7} \approx 3.4 \times 10^{38}$ overflow to `inf`. When the computer computes `inf / inf`, it outputs `NaN` (Not a Number), instantly destroying the entire LLM training run!

Fortunately, Softmax possesses a mathematical superpower: **Shift Invariance**.

#### Mathematical Proof of Shift Invariance

Subtract any constant $c \in \mathbb{R}$ from every element of $\mathbf{z}$:

$$
\frac{e^{z_i - c}}{\sum_{j=1}^N e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_{j=1}^N \left(e^{z_j} \cdot e^{-c}\right)} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \cdot \sum_{j=1}^N e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$

The factor $e^{-c}$ cancels out in the numerator and denominator!

#### The Practical Implementation Trick

Set $c = \max_{j}(z_j)$:

$$
z'_i = z_i - \max_{j}(z_j)
$$

Because the maximum score becomes $0$ (since $\max(z) - \max(z) = 0$), all transformed values $z'_i$ are guaranteed to be $\le 0$. Consequently:

$$
e^{z'_i} \in (0, 1] \quad \text{for all } i
$$

The largest possible exponent is $e^0 = 1.0$. Overflow is mathematically impossible!

---

### 4. The Temperature Knob: Tuning Model Creativity

In modern LLMs, we frequently divide logits by a positive hyperparameter $T > 0$ called **Temperature**:

$$
\operatorname{softmax}\left(\frac{\mathbf{z}}{T}\right)_i = \frac{e^{z_i / T}}{\sum_{j=1}^N e^{z_j / T}}
$$

<figure>
<pre>
   T -&gt; 0 (Freezing Cold)           T = 1.0 (Balanced)           T -&gt; inf (Boiling Hot)
   [Argmax: Peak Sharpness]        [Natural Distribution]        [Uniform: Total Chaos]
           |                               │                            │
      1.00 ┤     █                    0.70 ┤     █                 0.33 ┤  █   █   █
           │     │                         │     │                      │  │   │   │
           └─────┴──────                   └─────┴──────                └──┴───┴───┴──
            z1   z2   z3                    z1   z2   z3                 z1  z2  z3
</pre>
<figcaption><strong>Figure 7.3:</strong> How Temperature $T$ transforms the probability landscape.</figcaption>
</figure>

- **Low Temperature ($T \to 0$, e.g., $T = 0.2$)**:
  Divides by a tiny number, blowing up differences. The highest logit dominates with nearly $100\%$ probability. The model becomes deterministic, factual, and rigid.
- **Default Temperature ($T = 1.0$)**:
  The natural mathematical distribution determined by the model's weights.
- **High Temperature ($T \to \infty$, e.g., $T = 2.0$)**:
  Divides by a large number, shrinking all logits towards $0$. Since $e^0 = 1$, every option receives equal probability $1/N$. The model becomes creative, surprising, and eventually nonsensical.

---

### 5. Calculus of Softmax: The Local Derivative (Jacobian)

To train an LLM via backpropagation, we must calculate the local derivative of every Softmax output $s_i$ with respect to every input logit $z_j$:

$$
s_i = \frac{e^{z_i}}{\sum_{k=1}^N e^{z_k}} = \frac{u}{v}
$$

Let $u = e^{z_i}$ and $v = \sum_{k=1}^N e^{z_k}$. By the quotient rule of calculus:

$$
\frac{\partial s_i}{\partial z_j} = \frac{\frac{\partial u}{\partial z_j} v - u \frac{\partial v}{\partial z_j}}{v^2}
$$

Notice that $\frac{\partial v}{\partial z_j} = \frac{\partial}{\partial z_j}\left(e^{z_1} + \dots + e^{z_j} + \dots\right) = e^{z_j}$.

Now we consider two cases:

#### Case 1: When $i = j$ (Diagonal elements of the Jacobian)

Here, $\frac{\partial u}{\partial z_i} = e^{z_i}$:

$$
\begin{aligned}
\frac{\partial s_i}{\partial z_i} &= \frac{e^{z_i} v - e^{z_i} e^{z_i}}{v^2} \\
&= \frac{e^{z_i}}{v} - \left(\frac{e^{z_i}}{v}\right)^2 \\
&= s_i - s_i^2 \\
&= s_i (1 - s_i)
\end{aligned}
$$

#### Case 2: When $i \neq j$ (Off-diagonal elements of the Jacobian)

Here, $u = e^{z_i}$ does not depend on $z_j$, so $\frac{\partial u}{\partial z_j} = 0$:

$$
\begin{aligned}
\frac{\partial s_i}{\partial z_j} &= \frac{0 \cdot v - e^{z_i} e^{z_j}}{v^2} \\
&= -\frac{e^{z_i}}{v} \cdot \frac{e^{z_j}}{v} \\
&= -s_i s_j
\end{aligned}
$$

#### The Unified Matrix Formula

Using the Kronecker delta $\delta_{ij}$ (where $\delta_{ij} = 1$ if $i = j$, and $0$ otherwise):

$$
\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j)
$$

The gradient of Softmax depends **only on its own output probabilities**! This elegant property makes backpropagation exceptionally fast to compute on GPUs.

---

## Step 4: Where Did It Come From?

<dl>
  <dt><time datetime="1868">1868</time> &mdash; <strong>Ludwig Boltzmann &amp; Thermal Equilibrium</strong></dt>
  <dd>
    Austrian physicist Ludwig Boltzmann was studying gas particles bouncing around a closed chamber. He discovered that the probability of a physical system occupying microstate $i$ with energy $E_i$ at temperature $T$ follows the canonical distribution:
    $$P(i) = \frac{e^{-E_i / (k_B T)}}{\sum_j e^{-E_j / (k_B T)}}$$
    Where $k_B$ is the Boltzmann constant. States with lower energy are exponentially more stable and probable.
  </dd>

  <dt><time datetime="1959">1959</time> &mdash; <strong>R. Duncan Luce &amp; The Choice Axiom</strong></dt>
  <dd>
    In mathematical psychology, R. Duncan Luce formulated the axiom that the probability of choosing item $A$ over a set of alternatives should depend on the ratio of positive utilities ($u(A) / \sum u(X)$). Using exponential utility $u(z) = e^z$ produces the Softmax choice model.
  </dd>

  <dt><time datetime="1989">1989</time> &mdash; <strong>John S. Bridle &amp; Softmax in Neural Networks</strong></dt>
  <dd>
    British researcher John S. Bridle introduced the formula to machine learning in his landmark paper <em>"Probabilistic Interpretation of Feedforward Classification Network Outputs"</em>. He named it <samp>"Softmax"</samp> because it provides a smooth, differentiable approximation to the discontinuous maximum function (<samp>hardmax</samp> / $\operatorname{argmax}$).
  </dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Vaswani et al. &amp; The Transformer Attention Router</strong></dt>
  <dd>
    In <em>"Attention Is All You Need"</em>, Softmax was placed at the mathematical core of Scaled Dot-Product Attention:
    $$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$
    Softmax dynamically calculates how much attention each word should pay to every other word in the sequence.
  </dd>
</dl>

---

## Step 5: Concrete Toy Example

Let us pick up the exact numerical story from **Chapter 06 Step 5**.

The ambiguous token <kbd>"bank"</kbd> calculated raw dot-product affinity scores against three words in the sentence <kbd>"The river bank"</kbd>:

$$
\mathbf{z} = [z_1, z_2, z_3] = [4.0, 7.0, 6.0]
$$

Let us convert these raw logits into attention weights, step by step by hand.

---

### Execution Checklist

<fieldset>
<legend><strong>Execution Checklist</strong></legend>
<p><input type="checkbox" checked disabled> <strong>Step 5.1:</strong> Compute standard Softmax exponentiation and probabilities</p>
<p><input type="checkbox" checked disabled> <strong>Step 5.2:</strong> Verify numerically stable Softmax ($c = \max$)</p>
<p><input type="checkbox" checked disabled> <strong>Step 5.3:</strong> Test Temperature effects ($T = 0.5$ vs $T = 2.0$)</p>
<p><input type="checkbox" checked disabled> <strong>Step 5.4:</strong> Synthesize the final Context Vector $\mathbf{c}_{\text{bank}}$ using Value vectors</p>
</fieldset>

---

### Step 5.1: Standard Softmax Calculation

- **Substep A: Calculate Exponentials ($e^{z_i}$)**:
  - $e^{z_1} = e^4 \approx 54.5982$
  - $e^{z_2} = e^7 \approx 1096.6332$
  - $e^{z_3} = e^6 \approx 403.4288$

- **Substep B: Calculate Normalizing Sum ($\sum_{j=1}^3 e^{z_j}$)**:
  $$
  \sum = 54.5982 + 1096.6332 + 403.4288 = 1554.6602
  $$

- **Substep C: Compute Probability Weights ($s_i = e^{z_i} / \sum$)**:
  - $s_1 (\text{"The"}) = \frac{54.5982}{1554.6602} \approx \mathbf{0.0351} \quad (3.51\%)$
  - $s_2 (\text{"river"}) = \frac{1096.6332}{1554.6602} \approx \mathbf{0.7054} \quad (70.54\%)$
  - $s_3 (\text{"bank"}) = \frac{403.4288}{1554.6602} \approx \mathbf{0.2595} \quad (25.95\%)$

- **Substep D: Verify Total Sum**:
  $$
  0.0351 + 0.7054 + 0.2595 = \mathbf{1.0000} \quad (100.00\%)
  $$

---

### Step 5.2: Numerically Stable Softmax ($c = \max$)

Now let us run the numerically stable version used in PyTorch and TensorFlow:

- **Substep A: Identify Maximum**:
  $c = \max([4, 7, 6]) = 7.0$.

- **Substep B: Subtract $c$ from Logits**:
  $$
  \mathbf{z}' = [4 - 7, 7 - 7, 6 - 7] = [-3.0, 0.0, -1.0]
  $$

- **Substep C: Compute Exponentials**:
  - $e^{-3.0} \approx 0.04979$
  - $e^{0.0} = 1.00000$
  - $e^{-1.0} \approx 0.36788$

- **Substep D: Compute Sum**:
  $$
  \sum = 0.04979 + 1.00000 + 0.36788 = 1.41767
  $$

- **Substep E: Compute Probabilities**:
  - $s'_1 = \frac{0.04979}{1.41767} \approx \mathbf{0.0351}$
  - $s'_2 = \frac{1.00000}{1.41767} \approx \mathbf{0.7054}$
  - $s'_3 = \frac{0.36788}{1.41767} \approx \mathbf{0.2595}$

The probabilities are **identical to 4 decimal places**, but the maximum exponent was $1.0$ instead of $1096.6$!

---

### Step 5.3: Temperature Effects on Probability

What happens when we tune the temperature knob?

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 7.2:</strong> Attention Probability Distribution Across Different Temperatures</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">Token Candidate</th>
      <th scope="col" align="center" width="25%">Cold ($T = 0.5$)</th>
      <th scope="col" align="center" width="25%">Default ($T = 1.0$)</th>
      <th scope="col" align="center" width="30%">Hot ($T = 2.0$)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><kbd>"The"</kbd></th>
      <td align="center">0.15% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0015">0.15%</meter></td>
      <td align="center">3.51% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0351">3.51%</meter></td>
      <td align="center">12.86% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1286">12.86%</meter></td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"river"</kbd> (Clue)</th>
      <td align="center"><mark><strong>87.99%</strong></mark> <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.8799">87.99%</meter></td>
      <td align="center"><strong>70.54%</strong> <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.7054">70.54%</meter></td>
      <td align="center"><strong>52.79%</strong> <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.5279">52.79%</meter></td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"bank"</kbd> (Self)</th>
      <td align="center">11.86% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1186">11.86%</meter></td>
      <td align="center">25.95% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.2595">25.95%</meter></td>
      <td align="center">34.35% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.3435">34.35%</meter></td>
    </tr>
  </tbody>
</table>

- At $T = 0.5$, <kbd>"river"</kbd> surges to **$88\%$** of the model's attention beam!
- At $T = 2.0$, attention spreads out across all words, becoming more exploratory.

---

### Step 5.4: Blending the Value Vectors into Context

Now we finish the Attention operation!

From Chapter 06 Step 3, the three tokens possess the following Value vectors $\mathbf{v}_j \in \mathbb{R}^2$:

$$
\mathbf{v}_1 (\text{"The"}) = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
\mathbf{v}_2 (\text{"river"}) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
\mathbf{v}_3 (\text{"bank"}) = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

The updated contextual representation for <kbd>"bank"</kbd> is the **weighted sum of all Value vectors** using our Softmax attention weights:

$$
\begin{aligned}
\mathbf{c}_{\text{bank}} &= \sum_{j=1}^3 s_j \mathbf{v}_j \\
&= 0.0351 \begin{bmatrix} 1 \\ 2 \end{bmatrix} + 0.7054 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.2595 \begin{bmatrix} 1 \\ 2 \end{bmatrix} \\
&= \begin{bmatrix} 0.0351 \times 1 + 0.7054 \times 0 + 0.2595 \times 1 \\ 0.0351 \times 2 + 0.7054 \times 1 + 0.2595 \times 2 \end{bmatrix} \\
&= \begin{bmatrix} 0.0351 + 0.0 + 0.2595 \\ 0.0702 + 0.7054 + 0.5190 \end{bmatrix} \\
&= \mathbf{\begin{bmatrix} 0.2946 \\ 1.2946 \end{bmatrix}}
\end{aligned}
$$

Look at what happened:
- Before attention, the static vector for <kbd>"bank"</kbd> was isolated and had no idea whether it meant a financial building or a river shoreline.
- After Softmax attention, **$70.54\%$** of the information loaded into <kbd>"bank"</kbd>'s new representation came directly from <kbd>"river"</kbd>!
- The word <kbd>"bank"</kbd> is now infused with river-like context!

---

## Step 6: Core Takeaway

<fieldset>
<legend><strong>Core Memory Card</strong></legend>
<p>
<strong>Softmax is the universal bridge between unbounded linear energy and normalized probabilistic belief:</strong><br>
Through the exponential function $e^z$, Softmax guarantees that every candidate receives a non-zero probability while accentuating the most confident options. By normalizing with the partition function $\sum e^{z_j}$, it produces a valid probability distribution that sums to strictly $100\%$. In Attention, Softmax acts as the dynamic router, determining precisely how much context each word absorbs from the rest of the sentence.
</p>
</fieldset>
