# Chapter 07: The Fair Voting Booth (The Softmax Function)


## Step 1: 3-Year-Old Intuition (The Shouting Match & The Pizza Slices) {: #step-1 }

!!! note "3-Year-Old Intuition: The Shouting Match & The Pizza Slices"
    Imagine three children sitting at a play table, shouting out what they want on tonight's dinner pizza:

    - Child 1 whispers quietly: <kbd>"Mushrooms!"</kbd> (Volume score: 4)
    - Child 2 shouts at the top of their lungs: <kbd>"Pepperoni!"</kbd> (Volume score: 7)
    - Child 3 speaks in a normal voice: <kbd>"Cheese!"</kbd> (Volume score: 6)

    If you just listen to the raw volume scores $[4, 7, 6]$, how do you slice a single pizza fairly?

    You cannot cut a pizza into "volume units." A pizza is one whole thing (100%). Slices must always add up to exactly the whole pizza. Furthermore, no slice can be negative (you cannot take away a slice someone already ate!), and no slice can be bigger than the whole pizza.

    What happens if a fourth child is grumpy and grumbles with a negative volume score: $-3$? You certainly cannot give them "negative pizza slices"!

    The wise pizza chef invents a **Fair Voting Booth**:

    1. **Turn Every Grumble Positive**: The chef puts each volume score through a magical magnifier machine ($e^z$). Even negative grumbles turn into tiny, positive sprinkles of flour ($e^{-3} \approx 0.05$), while loud shouts turn into giant heaps of toppings ($e^7 \approx 1097$). Nothing is ever negative!
    2. **Measure the Whole Pile**: The chef dumps all the toppings into one giant bowl and weighs the total sum ($1555$ grams).
    3. **Hand Out the Slices**: The chef gives each child a slice equal to their personal topping weight divided by the giant bowl's total weight.

    Instantly, everyone gets a slice between $0\%$ and $100\%$, and all slices together add up to **exactly 100% of the pizza**!

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

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: From Unbounded Scores to Probability Distributions"
    In Chapter 06, we watched the ambiguous word <kbd>"bank"</kbd> fire its Query arrow against the Key arrows of <kbd>"The"</kbd>, <kbd>"river"</kbd>, and <kbd>"bank"</kbd>. The linear dot products produced raw affinity scores:

    $$
    \mathbf{z} = [4.0, 7.0, 6.0]
    $$

    But these raw dot products (called **logits**) are mathematically wild:

    1. They can be any real number from $-\infty$ to $+\infty$.
    2. They do not sum to $1$.
    3. They can easily be negative if two vectors point in opposite directions.

    To blend the meaning of words together in Attention (and to pick the final next word out of a 100,000-word dictionary), the Transformer must answer:

    *"How do we convert an arbitrary list of real numbers—positive, negative, or zero—into a strictly valid probability distribution where every value is positive and the sum is strictly 100%, without destroying the relative ranking of the candidates?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Softmax Function

Let $\mathbf{z} = [z_1, z_2, \dots, z_N]^\top \in \mathbb{R}^N$ be an $N$-dimensional vector of raw scores (\lt dfn id="def-logit">logits</dfn>).

The \lt dfn id="def-softmax">Softmax function</dfn> transforms $\mathbf{z}$ into a probability distribution $\mathbf{s} = \operatorname{softmax}(\mathbf{z}) \in \mathbb{R}^N$:

$$
\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 7.1:</strong> Complete Mathematical Symbol Definitions for Softmax</caption>
  \lt thead>
    \lt tr bgcolor="#eae9e1">
      \lt th scope="col" align="left" width="22%">Symbol</th>
      \lt th scope="col" align="left" width="22%">Type &amp; Domain</th>
      \lt th scope="col" align="left" width="56%">Mathematical Meaning &amp; Physical Purpose</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt th scope="row" align="left">$\mathbf{z}$</th>
      \lt td align="left">Vector $\in \mathbb{R}^N$</td>
      \lt td>The raw, unnormalized score vector (logits). In Attention, $z_j = \mathbf{q}_i^\top \mathbf{k}_j$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$z_i$</th>
      \lt td align="left">Scalar $\in (-\infty, +\infty)$</td>
      \lt td>The individual raw score for candidate $i$. Can be positive, zero, or negative.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$e$</th>
      \lt td align="left">Constant $\approx 2.71828$</td>
      \lt td>Euler's number, base of the natural exponential function.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$e^{z_i}$</th>
      \lt td align="left">Scalar $\in (0, +\infty)$</td>
      \lt td>The exponentiated score. Guaranteed strictly positive for every real number $z_i$.</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\sum_{j=1}^N e^{z_j}$</th>
      \lt td align="left">Scalar $\in (0, +\infty)$</td>
      \lt td>The \lt strong>normalizing denominator</strong> (known in physics as the \lt em>Partition Function</em> $Z$).</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">$\operatorname{softmax}(\mathbf{z})_i$</th>
      \lt td align="left">Scalar $\in (0, 1)$</td>
      \lt td>The normalized probability for candidate $i$, satisfying $\sum_{i=1}^N \operatorname{softmax}(\mathbf{z})_i = 1.0$.</td>
    </tr>
  </tbody>
</table>

---

### 2. The Dependency Ladder: Why Exponentiate?

Why did mathematicians choose the natural exponential $e^z$ instead of simpler normalization formulas? A valid probability distribution must satisfy two foundational axioms of probability theory:
1. **Non-negativity**: Every probability must be non-negative: $p_i \ge 0$ for all $i$.
2. **Total Probability**: All probabilities must sum to unity: $\sum_{i=1}^N p_i = 1.0$.

Let us walk up the mathematical ladder of intuitive alternatives to see why simpler choices fail fatally.

#### Attempt 1: Direct Linear Normalization

The simplest idea is to divide each logit by the sum of all logits:

$$
p_i = \frac{z_i}{\sum_{j=1}^N z_j}
$$

Why this fails in practice:
- **Fatal Flaw 1 (Division by Zero)**: If the raw logits sum to zero, the denominator vanishes. For example, if $\mathbf{z} = [2.0, -3.0, 1.0]^\top$, then:
  $$
  \sum_{j=1}^3 z_j = 2.0 + (-3.0) + 1.0 = 0
  $$
  Computing $p_i = \frac{z_i}{0}$ results in a division-by-zero crash (`ZeroDivisionError` / `inf`), halting training immediately.
- **Fatal Flaw 2 (Negative Probabilities)**: If any logit is negative while the sum is positive, the resulting output is negative. For example, if $\mathbf{z} = [2.0, -1.0, 3.0]^\top$, then the sum is $\sum z_j = 4.0$, yielding:
  $$
  p_2 = \frac{-1.0}{4.0} = -0.25 = -25\%
  $$
  A probability of $-25\%$ is mathematically nonsensical and violates the axioms of probability theory.

#### Attempt 2: Absolute Value Normalization

To prevent negative values, one might try taking the absolute value $|z_i|$ before summing:

$$
p_i = \frac{|z_i|}{\sum_{j=1}^N |z_j|}
$$

Why this fails in practice:
- **Fatal Flaw (Destruction of Order & Symmetry Inversion)**: Absolute value destroys the directional meaning of logits. Suppose candidate 1 has $z_1 = -10.0$ (a terrible semantic match) and candidate 2 has $z_2 = +10.0$ (a perfect semantic match). Absolute value maps both to $10.0$:
  $$
  |-10.0| = |+10.0| = 10.0 \implies p_1 = p_2
  $$
  The model would assign equal probability to the worst possible word and the best possible word, completely destroying relative ordering!

#### Attempt 3: Rectified Linear (ReLU) Normalization

To eliminate negative values without folding them into positive ones, we could clamp negative logits to zero using the ReLU function $\max(0, z_i)$:

$$
p_i = \frac{\max(0, z_i)}{\sum_{j=1}^N \max(0, z_j)}
$$

Why this fails in practice:
- **Fatal Flaw 1 (All-Negative Collapse)**: If all logits happen to be negative (e.g., $\mathbf{z} = [-2.0, -5.0, -1.0]^\top$), then $\max(0, z_j) = 0$ for all $j$, causing a catastrophic division-by-zero $\frac{0}{0} = \text{NaN}$.
- **Fatal Flaw 2 (Dead Gradients / Zero Learning Signal)**: For any rejected candidate where $z_i < 0$, the derivative is strictly zero:
  $$
  \frac{\partial \max(0, z_i)}{\partial z_i} = 0
  $$
  During backpropagation, zero gradient flows backward to those tokens. The neural network receives no information about *how far off* the prediction was, leaving it incapable of learning why the candidate was rejected.

#### The Winner: The Natural Exponential Function $e^z$

The exponential function $f(z) = e^z$ elegantly solves all four problems simultaneously:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$

- **Strict Non-Negativity Everywhere**: For every real number $z \in (-\infty, +\infty)$, $e^z > 0$. The denominator $\sum e^{z_j}$ is strictly positive, guaranteeing that division by zero is impossible and every $p_i \in (0, 1)$.
- **Strict Monotonicity (Order Preserved)**: Because $\frac{d}{dz} e^z = e^z > 0$, the exponential function is strictly increasing ($z_a > z_b \iff e^{z_a} > e^{z_b} \iff p_a > p_b$). A candidate with a higher logit always receives a higher probability.
- **Smooth, Non-Zero Gradients Everywhere**: The exponential function is infinitely differentiable ($C^\infty$). Its derivative is never zero, providing clean gradient highways during backpropagation.
- **Winner-Take-Most Amplification**: Because exponential curves grow rapidly, small differences in logits translate to decisive differences in probability, helping the model focus attention sharply on the most relevant tokens.

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 7.2:</strong> Mathematical Comparison of Candidate Normalization Schemes</caption>
  \lt thead>
    \lt tr bgcolor="#eae9e1">
      \lt th scope="col" align="left" width="22%">Scheme</th>
      \lt th scope="col" align="left" width="24%">Formula</th>
      \lt th scope="col" align="center" width="18%">Non-Negative?</th>
      \lt th scope="col" align="center" width="18%">Preserves Order?</th>
      \lt th scope="col" align="left" width="18%">Backprop Gradient</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt th scope="row" align="left">Linear</th>
      \lt td align="left">$p_i = \frac{z_i}{\sum z_j}$</td>
      \lt td align="center">\lt del>No (can be &lt; 0)</del></td>
      \lt td align="center">Yes</td>
      \lt td align="left">Undefined if $\sum z_j = 0$</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">Absolute Value</th>
      \lt td align="left">$p_i = \frac{|z_i|}{\sum |z_j|}$</td>
      \lt td align="center">Yes</td>
      \lt td align="center">\lt del>No (folds -z into +z)</del></td>
      \lt td align="left">Discontinuous at $z=0$</td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">ReLU (Clamped)</th>
      \lt td align="left">$p_i = \frac{\max(0, z_i)}{\sum \max(0, z_j)}$</td>
      \lt td align="center">Yes</td>
      \lt td align="center">Partial</td>
      \lt td align="left">Zero for $z \le 0$ (dead)</td>
    </tr>
    \lt tr bgcolor="#f0f7f0">
      \lt th scope="row" align="left">\lt strong>Softmax ($e^z$)</strong></th>
      \lt td align="left">\lt strong>$p_i = \frac{e^{z_i}}{\sum e^{z_j}}$</strong></td>
      \lt td align="center">\lt strong>Yes (strictly &gt; 0)</strong></td>
      \lt td align="center">\lt strong>Yes (strictly monotonic)</strong></td>
      \lt td align="left">\lt strong>Smooth &amp; non-zero everywhere</strong></td>
    </tr>
  </tbody>
</table>

---

### 3. The Numerical Stability Guard: Shift Invariance

If a neural network outputs large logits like $z_i = 1000$, computers crash. In 32-bit floating point arithmetic (\lt abbr title="IEEE 754 32-bit Single-Precision Float">FP32</abbr>), numbers above $e^{88.7} \approx 3.4 \times 10^{38}$ overflow to `inf`. When the computer computes `inf / inf`, it outputs `NaN` (Not a Number), instantly destroying the entire LLM training run!

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

\lt figure>
\lt pre>
   T -&gt; 0 (Freezing Cold)           T = 1.0 (Balanced)           T -&gt; inf (Boiling Hot)
   [Argmax: Peak Sharpness]        [Natural Distribution]        [Uniform: Total Chaos]
           |                               │                            │
      1.00 ┤     █                    0.70 ┤     █                 0.33 ┤  █   █   █
           │     │                         │     │                      │  │   │   │
           └─────┴──────                   └─────┴──────                └──┴───┴───┴──
            z1   z2   z3                    z1   z2   z3                 z1  z2  z3
</pre>
\lt figcaption>\lt strong>Figure 7.3:</strong> How Temperature $T$ transforms the probability landscape.</figcaption>
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

## Step 4: Where Did It Come From? {: #step-4 }

\lt dl>
  \lt dt>\lt time datetime="1868">1868</time> &mdash; \lt strong>Ludwig Boltzmann &amp; Thermal Equilibrium</strong></dt>
  \lt dd>
    Austrian physicist Ludwig Boltzmann was studying gas particles bouncing around a closed chamber. He discovered that the probability of a physical system occupying microstate $i$ with energy $E_i$ at temperature $T$ follows the canonical distribution:

$$
P(i) = \frac{e^{-E_i / (k_B T)}}{\sum_j e^{-E_j / (k_B T)}}
$$

    Where $k_B$ is the Boltzmann constant. States with lower energy are exponentially more stable and probable.
  </dd>

  \lt dt>\lt time datetime="1959">1959</time> &mdash; \lt strong>R. Duncan Luce &amp; The Choice Axiom</strong></dt>
  \lt dd>
    In mathematical psychology, R. Duncan Luce formulated the axiom that the probability of choosing item $A$ over a set of alternatives should depend on the ratio of positive utilities ($u(A) / \sum u(X)$). Using exponential utility $u(z) = e^z$ produces the Softmax choice model.
  </dd>

  \lt dt>\lt time datetime="1989">1989</time> &mdash; \lt strong>John S. Bridle &amp; Softmax in Neural Networks</strong></dt>
  \lt dd>
    British researcher John S. Bridle introduced the formula to machine learning in his landmark paper \lt em>"Probabilistic Interpretation of Feedforward Classification Network Outputs"</em>. He named it \lt samp>"Softmax"</samp> because it provides a smooth, differentiable approximation to the discontinuous maximum function (\lt samp>hardmax</samp> / $\operatorname{argmax}$).
  </dd>

  \lt dt>\lt time datetime="2017">2017</time> &mdash; \lt strong>Vaswani et al. &amp; The Transformer Attention Router</strong></dt>
  \lt dd>
    In \lt em>"Attention Is All You Need"</em>, Softmax was placed at the mathematical core of Scaled Dot-Product Attention:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

    Softmax dynamically calculates how much attention each word should pay to every other word in the sequence.
  </dd>
</dl>

---

## Step 5: Concrete Toy Example {: #step-5 }

Let us pick up the exact numerical story from **Chapter 06 Step 5**.

The ambiguous token \lt kbd>"bank"</kbd> calculated raw dot-product affinity scores against three words in the sentence \lt kbd>"The river bank"</kbd>:

$$
\mathbf{z} = [z_1, z_2, z_3] = [4.0, 7.0, 6.0]
$$

Let us convert these raw logits into attention weights, step by step by hand.

---

### Execution Checklist

\lt fieldset>
\lt legend>\lt strong>Execution Checklist</strong></legend>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>Step 5.1:</strong> Compute standard Softmax exponentiation and probabilities</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>Step 5.2:</strong> Verify numerically stable Softmax ($c = \max$)</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>Step 5.3:</strong> Test Temperature effects ($T = 0.5$ vs $T = 2.0$)</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>Step 5.4:</strong> Synthesize the final Context Vector $\mathbf{c}_{\text{bank}}$ using Value vectors</p>
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

\lt table border="1" cellpadding="8" cellspacing="0" width="100%">
  \lt caption>\lt strong>Table 7.2:</strong> Attention Probability Distribution Across Different Temperatures</caption>
  \lt thead>
    \lt tr bgcolor="#eae9e1">
      \lt th scope="col" align="left" width="20%">Token Candidate</th>
      \lt th scope="col" align="center" width="25%">Cold ($T = 0.5$)</th>
      \lt th scope="col" align="center" width="25%">Default ($T = 1.0$)</th>
      \lt th scope="col" align="center" width="30%">Hot ($T = 2.0$)</th>
    </tr>
  </thead>
  \lt tbody>
    \lt tr>
      \lt th scope="row" align="left">\lt kbd>"The"</kbd></th>
      \lt td align="center">0.15% \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0015">0.15%</meter></td>
      \lt td align="center">3.51% \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0351">3.51%</meter></td>
      \lt td align="center">12.86% \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1286">12.86%</meter></td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">\lt kbd>"river"</kbd> (Clue)</th>
      \lt td align="center">\lt mark>\lt strong>87.99%</strong></mark> \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.8799">87.99%</meter></td>
      \lt td align="center">\lt strong>70.54%</strong> \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.7054">70.54%</meter></td>
      \lt td align="center">\lt strong>52.79%</strong> \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.5279">52.79%</meter></td>
    </tr>
    \lt tr>
      \lt th scope="row" align="left">\lt kbd>"bank"</kbd> (Self)</th>
      \lt td align="center">11.86% \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1186">11.86%</meter></td>
      \lt td align="center">25.95% \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.2595">25.95%</meter></td>
      \lt td align="center">34.35% \lt br>\lt meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.3435">34.35%</meter></td>
    </tr>
  </tbody>
</table>

- At $T = 0.5$, \lt kbd>"river"</kbd> surges to **$88\%$** of the model's attention beam!
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

The updated contextual representation for \lt kbd>"bank"</kbd> is the **weighted sum of all Value vectors** using our Softmax attention weights:

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

## Step 6: Core Takeaway {: #step-6 }

<fieldset>
<legend><strong>Core Memory Card</strong></legend>
<p>
<strong>Softmax is the universal bridge between unbounded linear energy and normalized probabilistic belief:</strong><br>
Through the exponential function $e^z$, Softmax guarantees that every candidate receives a non-zero probability while accentuating the most confident options. By normalizing with the partition function $\sum e^{z_j}$, it produces a valid probability distribution that sums to strictly $100\%$. In Attention, Softmax acts as the dynamic router, determining precisely how much context each word absorbs from the rest of the sentence.
</p>
</fieldset>
