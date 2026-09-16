# Chapter 04: The One-Way Gate (Activation Functions: ReLU, GELU, SwiGLU)

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

---

<h2 id="step-1">Step 1: 3-Year-Old Intuition</h2>

Imagine you are sitting at your playroom craft table with a clean sheet of construction paper.

If all you are allowed to do is slide the paper across the desk, spin it around, or stretch it with your fingers (the linear matrix multiplications we explored in Chapter 03), the paper stays completely flat.

<figure>
<pre>
    Flat Sheet Stretched               The Sharp Crease (Non-Linearity)
   ┌───────────────────────┐                    ▲
   │                       │                   / \
   │   Stays 100% Flat     │      Fold        /   \     Becomes a 3D
   │   (Stacking flat is   │    ────────►    /     \    Origami Crane!
   │    always flat)       │                /       \
   └───────────────────────┘               /         \
</pre>
<figcaption><strong>Figure 4.1:</strong> Stacking flat sheets keeps everything flat. Creating a single sharp crease (an activation function) enables infinite three-dimensional origami structures.</figcaption>
</figure>

No matter how many times you stretch or rotate flat paper, you will **never** create a 3D origami bird, an accordion, or a paper airplane. 

To turn flat paper into a dimensional sculpture, you must introduce a **sharp crease** &mdash; a fold where the rules abruptly change.

### The Mystery of the Flat Glass Sheets

Think of building a telescope:
- If you stack 100 sheets of completely flat window glass on top of each other, look through them. Everything looks the exact same size. Stacking 100 flat sheets of glass is mathematically identical to looking through **one single thick sheet of flat glass**. It cannot magnify anything.
- To magnify a distant star or bend light into a focal point, you need a **curved lens**. A curve is a surface whose slope changes as you move along it.

### The Amusement Park Turnstile

Now imagine a one-way turnstile at an amusement park:
- If children run forward with positive energy, the gate smoothly turns and lets them right through.
- If a child tries to walk backward against the gate, the ratchet clicks and stops them instantly at the line ($0$).

This one-way turnstile is an **Activation Function**. 

In an artificial neural network, matrix multiplication is the flat glass; the activation function is the **curve** that allows the model to bend space, make decisions, filter out garbage, and perform genuine reasoning.

### The Clicky Wall Switch vs. The Smooth Dimmer Knob

Before the 1980s, early AI pioneers modeled neurons like a cheap, clicky wall switch:
- Push the switch gently: nothing happens.
- Push it past a sharp threshold: *click!* It suddenly flips completely ON ($1$).

Imagine you are standing in a pitch-black room trying to adjust 10,000 hidden light switches to reach the perfect room brightness. If every switch only clicks ON or OFF, you are completely blind. When you touch a switch, you have **no clue** whether you are 1 millimeter away from flipping it or 1 mile away, because the slope is dead flat everywhere ($\text{slope} = 0$).

Now replace that clicky switch with a **smooth dimmer knob**:
- When you nudge the knob just a tiny fraction of a millimeter, the light gets just a tiny fraction brighter.
- That "tiny change in light for a tiny twist of the knob" is the **Derivative** ($\frac{dy}{dx}$). 

The derivative gives the computer tactile feedback: it tells the learning algorithm **which direction to turn the knob** and **how hard to push** to reduce its mistakes. Without a derivative, the computer is trying to learn in the dark without any sense of touch.

---

<h2 id="step-2">Step 2: The Bridging Question</h2>

In Chapter 03, we celebrated matrix multiplication as the universal engine that transforms vector spaces:

$$
\mathbf{y} = \mathbf{x} \mathbf{W}
$$

Deep learning is famous for stacking dozens or hundreds of layers together:

$$
\mathbf{x} \to \text{Layer 1} \to \text{Layer 2} \to \dots \to \text{Layer } L \to \mathbf{y}
$$

This raises an alarming mathematical problem:

> *"If Layer 1 computes $\mathbf{h}_1 = \mathbf{x} \mathbf{W}_1$ and Layer 2 computes $\mathbf{y} = \mathbf{h}_1 \mathbf{W}_2$, what prevents the entire 100-layer neural network from collapsing into a single, shallow matrix multiplication?"*

This reveals that an activation function must solve **two non-negotiable requirements** at the exact same time:
1. **Non-Linearity (Forward Pass)**: It must curve and crease space so deep layers do not collapse into a single flat matrix.
2. **Differentiability (Backward Pass)**: It must provide a clean, non-zero mathematical derivative ($\frac{d\sigma}{dz}$) so the learning algorithm knows how to adjust the weights.

> *"Why can't we just pick any arbitrary non-linear curve, like a jagged staircase or a random zigzag? Why is the mathematical derivative ($\sigma'(z)$) the literal life-support system of neural network learning?"*

---

<h2 id="step-3">Step 3: The Exact Math &amp; Formula</h2>

---

### 1. The Catastrophe of Linear Collapse

Let us prove mathematically why an artificial neural network without activation functions is completely useless.

Suppose we build an $L$-layer network where every layer is purely linear. Let $\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{in}}}$ be the input vector. Each layer applies a weight matrix $\mathbf{W}_l$ and a bias vector $\mathbf{b}_l$:

$$
\begin{aligned}
\mathbf{h}_1 &= \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1 \\
\mathbf{h}_2 &= \mathbf{h}_1\mathbf{W}_2 + \mathbf{b}_2 = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2) \\
\mathbf{h}_3 &= \mathbf{h}_2\mathbf{W}_3 + \mathbf{b}_3 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2\mathbf{W}_3) + (\mathbf{b}_1\mathbf{W}_2\mathbf{W}_3 + \mathbf{b}_2\mathbf{W}_3 + \mathbf{b}_3)
\end{aligned}
$$

Notice what happens by induction across all $L$ layers:

$$
\mathbf{W}_{\text{comb}} = \prod_{l=1}^L \mathbf{W}_l = \mathbf{W}_1 \mathbf{W}_2 \dots \mathbf{W}_L \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}
$$

$$
\mathbf{b}_{\text{comb}} = \sum_{l=1}^L \mathbf{b}_l \left(\prod_{j=l+1}^L \mathbf{W}_j\right) \in \mathbb{R}^{1 \times d_{\text{out}}}
$$

The final output is simply:

$$
\mathbf{y} = \mathbf{x}\mathbf{W}_{\text{comb}} + \mathbf{b}_{\text{comb}}
$$

<fieldset>
<legend><strong>The Linear Collapse Theorem</strong></legend>
No matter how many hidden layers, billions of parameters, or millions of dollars of compute you spend: <strong>a cascade of purely linear transformations is mathematically identical to a single linear layer</strong>. It can only draw flat hyperplanes and cannot even compute the simple <dfn id="def-xor-problem">XOR</dfn> logic gate.
</fieldset>

To prevent this collapse, we **interleave** an element-wise non-linear scalar function $\sigma(\cdot)$ after each linear matrix multiplication:

$$
\mathbf{h}_l = \sigma(\mathbf{h}_{l-1}\mathbf{W}_l + \mathbf{b}_l)
$$

Because $\sigma(\mathbf{A}\mathbf{B}) \ne \sigma(\mathbf{A})\sigma(\mathbf{B})$, the associative property of matrix multiplication is shattered. The network cannot be collapsed, and each additional layer expands the geometrical complexity of the functions the model can compute.

---

### 2. Why the Derivative is Sacred: The Chain Rule Coupler

Why do deep learning practitioners obsess over the mathematical derivative $\sigma'(z)$ of an activation function?

Because neural networks **do not learn during the forward pass**. The forward pass only calculates a prediction. The model learns exclusively during the **backward pass** via <dfn id="def-gradient-descent"><strong>Gradient Descent</strong></dfn> and <dfn id="def-backpropagation"><strong>Backpropagation</strong></dfn>.

To teach a network, we calculate how much the loss (prediction error) $\mathcal{L}$ changes when we nudge a weight $w$:

$$
w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
$$

where $\eta > 0$ is the learning rate.

#### The Chain Rule Anatomy
Consider a single neuron computing an affine combination $z = \sum_k w_k x_k + b$ followed by an activation $a = \sigma(z)$. By the calculus chain rule, the gradient of the loss with respect to weight $w_k$ is:

$$
\frac{\partial \mathcal{L}}{\partial w_k} = \underbrace{\frac{\partial \mathcal{L}}{\partial a}}_{\text{Downstream Error } \delta} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\mathbf{\sigma'(z)}} \cdot \underbrace{\frac{\partial z}{\partial w_k}}_{x_k}
$$

<figure>
<pre>
   Backward Error Signal $\frac{\partial \mathcal{L}}{\partial a}$
                     │
                     ▼
             ┌───────────────┐
             │  $\sigma'(z)$ │  ◄─── The Derivative is the Physical Valve!
             └───────┬───────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  If $\sigma'(z) = 0$     If $\sigma'(z) \approx 1$
  Gradients Extinguished   Gradients Flow Freely
  $\frac{\partial \mathcal{L}}{\partial w} = 0$ (Frozen)   Weights Update &amp; Learn
</pre>
<figcaption><strong>Figure 4.2:</strong> The activation derivative $\sigma'(z)$ acts as a physical coupler or conduit for the backpropagating error signal. If the derivative is zero, the conduit is severed and no learning can occur.</figcaption>
</figure>

Notice the pivotal role of the middle factor $\frac{\partial a}{\partial z} = \sigma'(z)$:
- **The Derivative is the Mechanical Valve**: It directly scales the incoming error signal before passing it to the weights.
- **The Disastrous Step Function**: If we used a Heaviside step function $\Theta(z)$, its derivative is $0$ everywhere (except at $z=0$ where it is undefined):

$$
\frac{\partial \mathcal{L}}{\partial w_k} = \frac{\partial \mathcal{L}}{\partial a} \cdot \mathbf{0} \cdot x_k = 0
$$

  The learning signal is instantly annihilated. The weights receive a zero gradient and remain permanently frozen.
- **The Secret 1980s Hardware Miracle**: In the 1980s, computing exponential functions on CPUs was extraordinarily slow. Sigmoid ($\sigma$) and Tanh ($\tanh$) were revered because their derivatives could be computed **directly from the already-calculated activation value $a$**, without any expensive transcendental math:

$$
\sigma'(z) = a(1 - a), \quad \tanh'(z) = 1 - a^2
$$

  A single subtraction and multiplication was all early hardware needed to backpropagate!

---

### 3. The First Era: Sigmoid and Tanh (and the Vanishing Gradient)

In early neural networks, researchers used smooth S-shaped curves inspired by biological neurons.

#### The Logistic Sigmoid Function
Maps any real number $z \in (-\infty, \infty)$ into a probability-like range $(0, 1)$:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Its derivative has an elegant algebraic form:

$$
\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))
$$

#### The Hyperbolic Tangent (Tanh)
Zero-centered variant mapping $(-\infty, \infty)$ to $(-1, 1)$:

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
$$

Its derivative is:

$$
\frac{d\tanh(z)}{dz} = 1 - \tanh^2(z)
$$

<figure>
<pre>
   Activation Value $\sigma(z)$                Derivative $\sigma'(z)$
 1.0 ┌───────────────────----┐         0.25 ┌─────────/\─────────┐  Peak = 0.25
     │                     / │              │        /  \        │  at $z = 0$
 0.5 │........./‾‾‾‾‾........│              │       /    \       │
     │        /              │              │     /        \     │  Crushed to 0
 0.0 └───----────────────────┘         0.00 └───/────────────\───┘  when $|z| > 4$
    -6  -4  -2   0   2   4   6             -6  -4  -2   0   2   4   6
</pre>
<figcaption><strong>Figure 4.3:</strong> The Vanishing Gradient crisis: For large positive or negative inputs, Sigmoid derivative drops to zero. Multiplying these small derivatives across layers extinguishes training.</figcaption>
</figure>

#### Why Sigmoid and Tanh Broke in Deep Networks
Notice the peak derivative of Sigmoid: when $z = 0$, $\sigma'(0) = 0.5 \times 0.5 = 0.25$. Everywhere else, $\sigma'(z) < 0.25$.

During backpropagation via the chain rule, gradients are multiplied backwards through each layer:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \prod_{l=1}^L \left(\mathbf{W}_l^\top \cdot \operatorname{diag}(\sigma'(\mathbf{z}_l))\right)
$$

If every layer scales the gradient by at most $0.25$, then after just $L = 10$ layers:

$$
(0.25)^{10} \approx 0.00000095
$$

The learning signal diminishes to absolute zero. The earliest layers receive no feedback and cannot update their weights. This catastrophic phenomenon is known as the <dfn id="def-vanishing-gradient"><strong>Vanishing Gradient Problem</strong></dfn>.

---

### 4. The Second Era: The ReLU Revolution

In 2010&ndash;2012, researchers realized that biological realism was holding back deep learning. They replaced the complicated exponential curve with the simplest conceivable non-linear threshold: the <dfn id="def-relu"><strong>Rectified Linear Unit (ReLU)</strong></dfn>.

$$
\operatorname{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
$$

Its derivative is piecewise constant:

$$
\frac{d\operatorname{ReLU}(z)}{dz} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

*(At $z = 0$, the derivative is mathematically undefined, but software implementations set the subgradient to $0$ or $1$.)*

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 4.1:</strong> Why ReLU Changed Deep Learning Forever</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">Property</th>
      <th scope="col" align="left" width="40%">Sigmoid / Tanh</th>
      <th scope="col" align="left" width="40%">ReLU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>Gradient for $z > 0$</strong></th>
      <td>Vanishes exponentially ($\to 0$) as $|z|$ grows</td>
      <td><strong>Exactly $1.0$</strong> &mdash; gradients flow unhindered across 100+ layers</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Compute Cost</strong></th>
      <td>Heavy floating-point exponentials ($e^{-z}$) and division</td>
      <td><strong>Single comparison instruction</strong> (<code>max(0, x)</code>) on GPU silicon</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Sparsity</strong></th>
      <td>Dense: all neurons output non-zero values</td>
      <td><strong>True Sparsity</strong>: ~50% of neurons output exactly $0$, saving memory</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>Vulnerability</strong></th>
      <td>Vanishing gradients halt all deep architectures</td>
      <td><dfn id="def-dying-relu"><strong>Dying ReLU</strong></dfn>: if a neuron outputs negative, its gradient is 0 forever</td>
    </tr>
  </tbody>
</table>

#### The "Dying ReLU" Flaw
If an unfortunate gradient update knocks a neuron's weights such that $\mathbf{x}\mathbf{W} + b < 0$ for all training samples, its output is permanently $0$ and its gradient is permanently $0$. That neuron is effectively "dead" and will never learn again.

---

### 5. The Modern Transformer Era: GELU (GPT-2, GPT-3, BERT)

When building Transformers, researchers asked: *Can we retain ReLU's non-vanishing gradient while eliminating the rigid dead-zone cliff at $z = 0$?*

In 2016, Dan Hendrycks and Kevin Gimpel introduced the <dfn id="def-gelu"><strong>Gaussian Error Linear Unit (GELU)</strong></dfn>.

Instead of deterministically gating inputs based on sign, GELU scales an input $z$ by the probability that a standard normal variable $\mathcal{N}(0, 1)$ falls below $z$:

$$
\operatorname{GELU}(z) = z \cdot \Phi(z) = z \cdot P(X \le z), \quad \text{where } X \sim \mathcal{N}(0, 1)
$$

Here, $\Phi(z)$ is the cumulative distribution function (<abbr title="Cumulative Distribution Function">CDF</abbr>) of the Gaussian distribution:

$$
\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-\frac{t^2}{2}} \, dt = \frac{1}{2} \left[1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right]
$$

<figure>
<pre>
   ReLU (Rigid Corner at 0)                 GELU (Smooth Probabilistic Dip)
 2.0 ┌                     /       2.0 ┌                     /
     │                    /            │                    /
 1.0 │                   /         1.0 │                   /
     │                  /              │                  /
 0.0 └─────────--------┌───        0.0 └─────────-.....-─/───
    -3   -2   -1   0   1   2          -3   -2   -1   0   1   2
             Strictly 0                        Smooth dip to -0.17 at z = -0.75
</pre>
<figcaption><strong>Figure 4.4:</strong> Comparison of ReLU vs. GELU. Notice GELU's smooth curvature: small negative inputs are gently suppressed rather than abruptly extinguished.</figcaption>
</figure>

#### The Fast Tanh Approximation of GELU
Because computing the Gaussian error function $\operatorname{erf}(\cdot)$ is computationally expensive on GPUs, the original Transformer papers (GPT-2, GPT-3, BERT) use Hendrycks' fast numerical approximation:

$$
\operatorname{GELU}(z) \approx 0.5 z \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715 z^3\right)\right)\right)
$$

Notice key properties of GELU:
1. **Asymptotic Behavior**: As $z \to \infty$, $\Phi(z) \to 1$, so $\operatorname{GELU}(z) \to z$ (just like ReLU).
2. **Negative Tail**: As $z \to -\infty$, $\Phi(z) \to 0$, so $\operatorname{GELU}(z) \to 0$.
3. **Smooth Local Minimum**: For small negative values ($z \approx -0.7517$), $\operatorname{GELU}(z)$ reaches a gentle trough at $\approx -0.170$. This enables gradients to flow even when inputs are slightly negative!

---

### 6. The State of the Art: SwiGLU (LLaMA-1/2/3, Mistral, Gemma, DeepSeek)

In 2020, Google researcher Noam Shazeer published a landmark paper: *"GLU Variants Improve Transformer"*. 

Shazeer asked: *Why should an activation function simply be a fixed one-input scalar curve? What if one linear projection dynamically controlled a gate for a second linear projection?*

#### The Origin: Gated Linear Units (GLU)
Dauphin et al. (2017) introduced the <dfn id="def-glu"><strong>Gated Linear Unit (GLU)</strong></dfn> for language modeling. A GLU takes an input vector and splits it into two branches via two separate weight matrices:

$$
\operatorname{GLU}(\mathbf{x}, \mathbf{W}, \mathbf{V}) = (\mathbf{x}\mathbf{W}) \odot \sigma(\mathbf{x}\mathbf{V})
$$

where:
- $\mathbf{x}\mathbf{W}$ is the **value branch** (the candidate information signal).
- $\sigma(\mathbf{x}\mathbf{V})$ is the **gating branch** (a continuous valve between $0$ and $1$ deciding how much of each feature to pass).
- $\odot$ denotes the element-wise Hadamard product.

#### The Swish / SiLU Activation
Before defining SwiGLU, Shazeer replaced the Sigmoid gate $\sigma(z)$ with <dfn id="def-swish"><strong>Swish</strong></dfn> (also known as <abbr title="Sigmoid Linear Unit">SiLU</abbr>, Ramachandran et al. 2017):

$$
\operatorname{Swish}_1(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

#### The SwiGLU Formulation
Combining Swish with Gated Linear Units yields <dfn id="def-swiglu"><strong>SwiGLU</strong></dfn>:

$$
\operatorname{SwiGLU}(\mathbf{x}) = \operatorname{Swish}_1(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})
$$

In a modern Transformer Feed-Forward Network (<abbr title="Feed-Forward Network">FFN</abbr>), the SwiGLU block projects the hidden representation up into an expanded dimension $d_{\text{ffn}}$, computes the gated interaction, and then projects it back down to the model dimension $d_{\text{model}}$:

$$
\operatorname{FFN}_{\text{SwiGLU}}(\mathbf{x}) = \left(\operatorname{Swish}_1(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}
$$

<figure>
<pre>
                        Input Vector $\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{model}}}$
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
         $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$    $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$
                 │                               │
                 ▼                               │
          $\operatorname{Swish}_1(\cdot)$                │
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     Hadamard Product $\odot$
                                 │
                                 ▼
                        $\mathbf{h} \in \mathbb{R}^{1 \times d_{\text{ffn}}}$
                                 │
                                 ▼
         $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$
                                 │
                                 ▼
                       Output $\mathbf{y} \in \mathbb{R}^{1 \times d_{\text{model}}}$
</pre>
<figcaption><strong>Figure 4.5:</strong> Architecture of the modern SwiGLU FFN block used in LLaMA-3, Gemma, Mistral, and DeepSeek. Two parallel matrices produce the gate and the value, which modulate each other multiplicatively before the down-projection.</figcaption>
</figure>

#### Parameter Balancing in SwiGLU
Notice that standard GPT-style FFNs use **two** weight matrices:
- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$
- $\mathbf{W}_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$
- Total parameters: $2 \times 4 d_{\text{model}}^2 = 8 d_{\text{model}}^2$.

Because SwiGLU introduces a **third** matrix ($\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$), setting $d_{\text{ffn}} = 4d_{\text{model}}$ would increase total parameters by $50\%$ ($3 \times 4 = 12 d_{\text{model}}^2$).

To maintain identical parameter count and compute budget, Shazeer scaled the intermediate dimension down to $\frac{8}{3}d_{\text{model}}$:

$$
d_{\text{ffn}} \approx \left\lfloor \frac{8}{3} d_{\text{model}} \right\rfloor = \left\lfloor \frac{2}{3} \times 4 d_{\text{model}} \right\rfloor
$$

In modern models like LLaMA-3, $d_{\text{ffn}}$ is further rounded to the nearest multiple of $256$ or $1024$ for optimal GPU tensor-core memory alignment.

---

### 7. Mathematical Definition of All Variables and Dimensions

<details>
<summary><strong>Click to view Formal Mathematical Symbol Catalog</strong></summary>

<dl>
  <dt><strong>$\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{model}}}$</strong></dt>
  <dd>The input activation row vector for a single token at a given layer position.</dd>

  <dt><strong>$d_{\text{model}}$</strong></dt>
  <dd>The model's internal hidden dimension (e.g., $4096$ in LLaMA-3 8B, $8192$ in LLaMA-3 70B).</dd>

  <dt><strong>$d_{\text{ffn}}$</strong></dt>
  <dd>The expanded intermediate dimension of the feed-forward network, typically $\approx \frac{8}{3}d_{\text{model}}$ ($14336$ in LLaMA-3 8B).</dd>

  <dt><strong>$\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$</strong></dt>
  <dd>The gate projection matrix that determines the modulation weights.</dd>

  <dt><strong>$\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$</strong></dt>
  <dd>The up-projection matrix that produces the candidate information signal.</dd>

  <dt><strong>$\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$</strong></dt>
  <dd>The down-projection matrix that returns the gated vector to the model dimension.</dd>

  <dt><strong>$\odot$</strong></dt>
  <dd>The Hadamard product: component-wise multiplication $(\mathbf{a} \odot \mathbf{b})_i = a_i \cdot b_i$.</dd>

  <dt><strong>$\Phi(z)$</strong></dt>
  <dd>The Gaussian Cumulative Distribution Function: $\Phi(z) = P(X \le z)$ where $X \sim \mathcal{N}(0, 1)$.</dd>
</dl>

</details>

---

<h2 id="step-4">Step 4: Where Did It Come From?</h2>

The story of activation functions is the quest to find a mathematical gate that is simultaneously **expressive**, **differentiable**, and **numerically stable** across dozens of layers.

<dl>
  <dt><time datetime="1943">1943</time> &mdash; <strong>Warren McCulloch &amp; Walter Pitts</strong>: The Binary Step Neuron</dt>
  <dd>
    The first mathematical model of a biological brain cell used the Heaviside step function:

$$
f(z) = \begin{cases} 1 & \text{if } z \ge \theta \\ 0 & \text{if } z < \theta \end{cases}
$$

    <strong>Why it broke:</strong> The derivative of a step function is zero everywhere ($f'(z) = 0$), and undefined at the threshold $\theta$. Gradient descent cannot work if the slope is always zero!
  </dd>

  <dt><time datetime="1986">1986</time> &mdash; <strong>Rumelhart, Hinton &amp; Williams</strong>: The Smooth Sigmoid Era</dt>
  <dd>
    To enable backpropagation, researchers smoothed the step function into the logistic curve $\sigma(z) = \frac{1}{1 + e^{-z}}$. Its non-zero derivative allowed gradient descent to optimize multi-layer networks for the first time.
    <cite>"Learning representations by back-propagating errors", Nature 1986</cite>.
  </dd>

  <dt><time datetime="1991">1991</time> &mdash; <strong>Sepp Hochreiter</strong>: The Vanishing Gradient Diagnosis</dt>
  <dd>
    In his German diploma thesis, Hochreiter formally proved that recurrent networks and deep cascades using Sigmoid/Tanh suffer from exponential gradient decay, stalling learning in earlier layers.
  </dd>

  <dt><time datetime="2010">2010</time>&ndash;<time datetime="2012">2012</time> &mdash; <strong>Nair, Hinton &amp; Krizhevsky</strong>: The ReLU Revolution</dt>
  <dd>
    Vinod Nair and Geoffrey Hinton introduced ReLU to restricted Boltzmann machines (2010), and Alex Krizhevsky used it to conquer ImageNet with AlexNet (2012). By delivering a permanent gradient of $1.0$ for positive inputs, ReLU allowed networks to train 6 times faster and scale to unprecedented depths.
  </dd>

  <dt><time datetime="2016">2016</time> &mdash; <strong>Dan Hendrycks &amp; Kevin Gimpel</strong>: GELU &amp; The Transformer Revolution</dt>
  <dd>
    Hendrycks formulated GELU by bridging dropout regularization with activation gating. When OpenAI trained GPT-1, GPT-2, and GPT-3, and Google trained BERT, GELU became the universal activation function of the first wave of LLMs.
    <cite>"Gaussian Error Linear Units (GELUs)", arXiv:1606.08415</cite>.
  </dd>

  <dt><time datetime="2020">2020</time> &mdash; <strong>Noam Shazeer</strong>: SwiGLU &amp; Modern Gating</dt>
  <dd>
    Former Google Brain scientist Noam Shazeer demonstrated that replacing standard FFN activations with multiplicative bilinear gates produces consistent improvements in perplexity and downstream reasoning. Today, almost every leading frontier model &mdash; <strong>LLaMA-3</strong> (Meta), <strong>Mistral</strong>, <strong>Gemma</strong> (Google), <strong>DeepSeek</strong>, and <strong>Qwen</strong> &mdash; standardizes on SwiGLU.
    <cite>"GLU Variants Improve Transformer", arXiv:2002.05202</cite>.
  </dd>
</dl>

---

<h2 id="step-5">Step 5: Concrete Toy Example</h2>

Let's trace through the exact arithmetic of these activations using tiny numbers you can verify with pencil and paper.

---

### Part A: Numerical Proof of Linear Collapse

Let input $\mathbf{x} = \begin{bmatrix} 2 & 1 \end{bmatrix}$.

Let Layer 1 and Layer 2 be pure linear transformations:

$$
\mathbf{W}_1 = \begin{bmatrix} 1 & 2 \\ 0 & 3 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} -1 & 0 \\ 2 & 1 \end{bmatrix}
$$

#### Method 1: Forward propagation through both layers step-by-step

**1. Layer 1 output:**

$$
\mathbf{h}_1 = \mathbf{x}\mathbf{W}_1 = \begin{bmatrix} 2(1) + 1(0) & 2(2) + 1(3) \end{bmatrix} = \begin{bmatrix} 2 & 7 \end{bmatrix}
$$

**2. Layer 2 output:**

$$
\mathbf{y} = \mathbf{h}_1\mathbf{W}_2 = \begin{bmatrix} 2(-1) + 7(2) & 2(0) + 7(1) \end{bmatrix} = \begin{bmatrix} -2 + 14 & 0 + 7 \end{bmatrix} = \begin{bmatrix} 12 & 7 \end{bmatrix}
$$

#### Method 2: Collapsing both matrices into one single matrix

**1. Compute $\mathbf{W}_{\text{comb}} = \mathbf{W}_1 \mathbf{W}_2$:**

$$
\mathbf{W}_{\text{comb}} = \begin{bmatrix} 1(-1) + 2(2) & 1(0) + 2(1) \\ 0(-1) + 3(2) & 0(0) + 3(1) \end{bmatrix} = \begin{bmatrix} 3 & 2 \\ 6 & 3 \end{bmatrix}
$$

**2. Single-step forward pass:**

$$
\mathbf{y} = \mathbf{x}\mathbf{W}_{\text{comb}} = \begin{bmatrix} 2(3) + 1(6) & 2(2) + 1(3) \end{bmatrix} = \begin{bmatrix} 6 + 6 & 4 + 3 \end{bmatrix} = \begin{bmatrix} 12 & 7 \end{bmatrix}
$$

Both methods yield the exact identical vector $\begin{bmatrix} 12 & 7 \end{bmatrix}$. Two linear layers collapsed completely into one!

---

### Part B: Comparing ReLU, GELU, and Swish

Let a pre-activation vector be:

$$
\mathbf{z} = \begin{bmatrix} 2.0 & 0.0 & -1.5 \end{bmatrix}
$$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table 4.2:</strong> Hand-Calculated Activation Outputs for $\mathbf{z} = [2.0, 0.0, -1.5]$</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left">Input $z$</th>
      <th scope="col" align="center">$\operatorname{ReLU}(z)$</th>
      <th scope="col" align="center">$\operatorname{GELU}(z)$</th>
      <th scope="col" align="center">$\operatorname{Swish}_1(z)$</th>
      <th scope="col" align="left">Visual Energy Gauge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>$z_1 = +2.0$</strong></th>
      <td align="center">$\max(0, 2.0) = \mathbf{2.000}$</td>
      <td align="center">$2.0 \times \Phi(2.0) \approx \mathbf{1.954}$</td>
      <td align="center">$2.0 \times \sigma(2.0) \approx \mathbf{1.762}$</td>
      <td><meter min="-0.5" max="2.0" low="0.0" high="1.5" optimum="1.8" value="1.954">1.954</meter></td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="left"><strong>$z_2 = 0.0$</strong></th>
      <td align="center">$\max(0, 0.0) = \mathbf{0.000}$</td>
      <td align="center">$0.0 \times \Phi(0.0) = \mathbf{0.000}$</td>
      <td align="center">$0.0 \times \sigma(0.0) = \mathbf{0.000}$</td>
      <td><meter min="-0.5" max="2.0" low="0.0" high="1.5" optimum="1.8" value="0.0">0.000</meter></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>$z_3 = -1.5$</strong></th>
      <td align="center">$\max(0, -1.5) = \mathbf{0.000}$ <del>(hard cutoff)</del></td>
      <td align="center">$-1.5 \times \Phi(-1.5) \approx \mathbf{-0.100}$ <ins>(soft leak)</ins></td>
      <td align="center">$-1.5 \times \sigma(-1.5) \approx \mathbf{-0.274}$ <ins>(soft leak)</ins></td>
      <td><meter min="-0.5" max="2.0" low="0.0" high="1.5" optimum="1.8" value="-0.100">-0.100</meter></td>
    </tr>
  </tbody>
</table>

Notice how ReLU completely silences $z_3 = -1.5$ to $0.0$, while GELU and Swish allow a tiny negative trickle ($-0.100$ and $-0.274$) to flow through, maintaining gradient sensitivity!

---

### Part C: A Complete SwiGLU Block Walkthrough

Let's compute an entire modern SwiGLU feed-forward layer by hand:
- Input vector: $\mathbf{x} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}$ ($d_{\text{model}} = 2$).
- Intermediate dimension: $d_{\text{ffn}} = 2$.

Weight matrices:

$$
\mathbf{W}_{\text{gate}} = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}, \quad \mathbf{W}_{\text{up}} = \begin{bmatrix} 2 & 1 \\ 0 & -1 \end{bmatrix}, \quad \mathbf{W}_{\text{down}} = \begin{bmatrix} 1 & 2 \\ 1 & 0 \end{bmatrix}
$$

<fieldset>
<legend><strong>Execution Checklist: Step-by-Step SwiGLU Forward Pass</strong></legend>

<p><input type="checkbox" checked disabled> <strong>Step 1: Compute Gate Projection</strong><br>
Project input into the gating dimension:</p>

$$
\mathbf{g} = \mathbf{x}\mathbf{W}_{\text{gate}} = \begin{bmatrix} 1(1) + 2(-1) & 1(0) + 2(1) \end{bmatrix} = \begin{bmatrix} -1.0 & 2.0 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>Step 2: Activate the Gate with $\operatorname{Swish}_1$</strong><br>
Modulate gate values smoothly with Swish:</p>
<ul>
  <li>For $g_1 = -1.0$: $\sigma(-1.0) = \frac{1}{1 + e^1} \approx 0.2689 \implies \operatorname{Swish}(-1.0) = -1.0 \times 0.2689 = \mathbf{-0.269}$</li>
  <li>For $g_2 = 2.0$: $\sigma(2.0) = \frac{1}{1 + e^{-2}} \approx 0.8808 \implies \operatorname{Swish}(2.0) = 2.0 \times 0.8808 = \mathbf{1.762}$</li>
</ul>

$$
\operatorname{Swish}(\mathbf{g}) \approx \begin{bmatrix} -0.269 & 1.762 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>Step 3: Compute Up Projection (Candidate Signal)</strong><br>
Project input into the candidate feature space:</p>

$$
\mathbf{u} = \mathbf{x}\mathbf{W}_{\text{up}} = \begin{bmatrix} 1(2) + 2(0) & 1(1) + 2(-1) \end{bmatrix} = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>Step 4: Gated Modulation (Hadamard Product $\odot$)</strong><br>
Multiply the candidate features element-wise by the active gate:</p>

$$
\mathbf{h} = \operatorname{Swish}(\mathbf{g}) \odot \mathbf{u} = \begin{bmatrix} -0.269 \times 2.0 & 1.762 \times (-1.0) \end{bmatrix} = \begin{bmatrix} -0.538 & -1.762 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>Step 5: Down Projection to Model Dimension</strong><br>
Project the intermediate representation back to $d_{\text{model}}$:</p>

$$
\mathbf{y} = \mathbf{h}\mathbf{W}_{\text{down}} = \begin{bmatrix} -0.538(1) + (-1.762)(1) & -0.538(2) + (-1.762)(0) \end{bmatrix} = \begin{bmatrix} -2.300 & -1.076 \end{bmatrix}
$$

</fieldset>

Final Output of the SwiGLU block: $\mathbf{y} = \begin{bmatrix} -2.300 & -1.076 \end{bmatrix}$.

Notice the sheer expressiveness: feature 1 was modulated by a suppressed gate ($\approx -0.269$), while feature 2 passed through an amplified, wide-open gate ($1.762$). The neural network dynamically controlled its own computational circuits!

---

<h2 id="step-6">Step 6: Core Takeaway</h2>

<fieldset>
<legend><strong>The Big Picture Punchline</strong></legend>
Linear matrix multiplications can only rotate and stretch flat space; <strong>activation functions are the sole reason an artificial neural network can bend, fold, and carve non-linear decision boundaries</strong>. By using <strong>SwiGLU</strong>, modern Large Language Models transform passive matrix layers into dynamic, self-modulating gates where tokens actively decide which concepts to amplify and which to silence.
</fieldset>

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../03-matrix-multiplication/index.html">&larr; Chapter 03: The Magic Stretching Box (Matrix Multiplication)</a> &bull;
    <a href="../index.html">Course Overview</a> &bull;
    <a href="../05-queries-keys-values/index.html">Chapter 05: The Library Clue Hunt (Queries, Keys, and Values) &rarr;</a>
  </p>
</nav>
