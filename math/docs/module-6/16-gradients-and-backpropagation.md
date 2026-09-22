# Chapter 16: Walking Down the Mountain (Gradients & Backpropagation)


## Step 1: 3-Year-Old Intuition (The Foggy Mountain & The Bucket Brigade) {: #step-1 }

!!! note "3-Year-Old Intuition: The Blindfolded Hiker and the Mountain Echo"
    Imagine you are a hiker standing on a steep, foggy mountain at dusk.

    Your goal is to reach the cozy cabin resting safely at the bottom of the deepest valley (where the error penalty is zero). But the fog is so thick that you cannot see more than two inches in front of your nose:

    1. **Feeling the Slope Under Your Boots**:
       - You cannot see the cabin in the distance.
       - But by feeling the tilt of the rocky ground beneath the soles of your hiking boots, you can instantly tell which direction tilts downward!
       - If the ground slopes steeply down to your left, you take a cautious, measured step to the left.
       - Each small step lowers your altitude. After hundreds of tiny steps, you reliably descend into the valley.
       - This tilt under your boots is the **Gradient (<dfn id="def-gradient">$\nabla \mathcal{L}$</dfn>)**!

    2. **The Bucket Brigade Relay**:
       - Now imagine that this mountain trail is staffed by a chain of mountain guides standing in a long line from the bottom of the valley all the way up to the mountain peak.
       - The scout at the very bottom checks how far off the trail the hiker landed, writes down a small note: *"Too far left by 2 feet!"*, and passes it backward to the guide behind them.
       - That guide reads the note, calculates their own adjustment, and passes an updated note further backward up the chain.
       - Within seconds, every single guide on the mountain knows exactly which way to nudge their signpost so the next hiker arrives closer to the cabin!

    In Large Language Models, this mountain descent is **Gradient Descent**, and the backward message passing is **Backpropagation**.

    Even though a model like LLaMA has 70 billion individual knob settings (weights), Backpropagation allows the model to compute the exact tilt for all 70,000,000,000 knobs simultaneously in a single, lightning-fast backward sweep!

<figure>
<pre>
Forward Pass (Making the Guess):
  [ Input Word ] ──► [ Layer 1 ] ──► [ Layer 2 ] ──► [ Prediction ]
                                                           │
                                                           ▼
                                                    [ Error Loss ]
                                                           │
Backward Pass (Passing the Blame Note):                     ▼
  [ Adj. W_1 ]  ◄── [ Adj. W_2 ] ◄── [ "You were off by +2.0!" ]
  (Each layer receives the blame note, updates its weights,
   and passes the gradient upstream to the previous layer)
</pre>
<figcaption><strong>Figure 16.1:</strong> The forward pass generates predictions; the backward pass propagates error gradients backward through the chain of layers.</figcaption>
</figure>

---

## Step 2: The Bridging Question {: #step-2 }

!!! question "The Bridging Question: How Do We Compute 70 Billion Partial Derivatives in a Split Second?"
    In Chapter 15, we discovered the miracle gradient at the very top of the network:



    $$
    \frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}
    $$



    But this gradient only tells us how to adjust the final output logits $\mathbf{z}$.

    Deep beneath those logits lie dozens of Transformer layers: unembedding matrices $\mathbf{W}_U$, feed-forward projections $\mathbf{W}_{\text{down}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{gate}}$, attention heads $\mathbf{W}_O, \mathbf{W}_V, \mathbf{W}_K, \mathbf{W}_Q$, normalization gains $\boldsymbol{\gamma}$, and input token embeddings $\mathbf{E}$.

    If we tried to compute the derivative of the scalar loss $\mathcal{L}$ with respect to each of the $P = 70,000,000,000$ parameters individually (e.g. by nudging one weight by $\epsilon = 0.0001$ and running the model forward to observe the change in loss), training a single token would require **70 billion forward passes**! A single sentence would take decades to compute.

    *"How does the multivariate Chain Rule transform this impossible billion-step bottleneck into an algorithm that computes all 70 billion gradients in the time of a single backward pass?"*

---

## Step 3: The Exact Math & Formula {: #step-3 }

### 1. The Computational Graph & Multivariate Chain Rule

A neural network is formally represented as a directed acyclic graph (<dfn id="def-comp-graph">Computational Graph</dfn>) of elementary operations.
Suppose variable $\mathbf{x}$ produces intermediate variable $\mathbf{y} = f(\mathbf{x})$, which in turn produces scalar loss $\mathcal{L} = g(\mathbf{y})$.

By the multivariate **Chain Rule of Calculus**:



$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
$$



where:
- $\frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^{1 \times d_y}$ is the incoming gradient from the downstream layer (the "blame note").
- $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{d_y \times d_x}$ is the local **Jacobian matrix** of the operation $f$.
- $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times d_x}$ is the resulting gradient propagated upstream to the previous operation.

---

### 2. The Core Building Block: Linear Layer Matrix Gradients

Modern LLMs spend over 95% of their computational FLOPs inside linear matrix multiplications: $\mathbf{y} = \mathbf{x}\mathbf{W}$.

Let:
- $\mathbf{x} \in \mathbb{R}^{B \times d_{\text{in}}}$ be the input activation matrix (batch size $B$).
- $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ be the learnable weight matrix.
- $\mathbf{y} = \mathbf{x}\mathbf{W} \in \mathbb{R}^{B \times d_{\text{out}}}$ be the output activation.
- $\mathbf{G}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^{B \times d_{\text{out}}}$ be the incoming error gradient.

During backpropagation, this single node computes two separate matrix multiplications:

#### A. Gradient with respect to Weights (for Parameter Updating):


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{x}^\top \mathbf{G}_y \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}
$$



#### B. Gradient with respect to Inputs (for Upstream Propagation):


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{G}_y \mathbf{W}^\top \in \mathbb{R}^{B \times d_{\text{in}}}
$$



<figure>
<pre>
Linear Layer Backpropagation Flow:

Forward:   x  [ B x d_in ] ──► ( * W ) ──► y  [ B x d_out ]
                                  │
                                  ▼
Backward:  dL/dx = G_y * W^T ◄── ( G_y ) ◄── dL/dy = G_y
                                  │
                                  ▼
                         dL/dW = x^T * G_y
</pre>
<figcaption><strong>Figure 16.2:</strong> A linear layer splits incoming gradient $\mathbf{G}_y$ into weight gradient $\mathbf{x}^\top \mathbf{G}_y$ and input gradient $\mathbf{G}_y \mathbf{W}^\top$.</figcaption>
</figure>

Notice the strict dimensional symmetry:
- To update $\mathbf{W}$ ($d_{\text{in}} \times d_{\text{out}}$), we multiply transposed input $\mathbf{x}^\top$ ($d_{\text{in}} \times B$) by gradient $\mathbf{G}_y$ ($B \times d_{\text{out}}$).
- To send gradients upstream ($B \times d_{\text{in}}$), we multiply gradient $\mathbf{G}_y$ ($B \times d_{\text{out}}$) by transposed weights $\mathbf{W}^\top$ ($d_{\text{out}} \times d_{\text{in}}$).

---

### 3. Gradient Descent Parameter Update Rule

Once the gradient matrix $\nabla_{\mathbf{W}} \mathcal{L}$ is computed, the parameters are updated along the negative gradient direction:



$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla_{\mathbf{W}} \mathcal{L}_t
$$



where $\eta > 0$ is the **learning rate** (the step size down the mountain).

---

### 4. Why Reverse-Mode is $70,000,000,000\times$ Faster

Why does backpropagation run *backward* instead of *forward*?

Consider a model with $P$ parameters that outputs a single scalar loss $\mathcal{L} \in \mathbb{R}$:
- **Forward-Mode Differentiation (Tangents)**: Propagates derivatives forward from inputs to outputs. Because each input parameter requires its own independent pass, computing all parameter gradients requires **$\mathcal{O}(P)$ passes** ($70 \times 10^9$ passes!).
- **Reverse-Mode Differentiation (Adjoints / Backprop)**: Starts at the single scalar output $\mathcal{L}$ and propagates gradients backward toward all parameters. Because there is only **1 scalar output**, all $P$ derivatives are obtained in **a single $\mathcal{O}(1)$ backward pass**!



$$
\frac{\text{Cost of Reverse-Mode}}{\text{Cost of Forward-Mode}} = \frac{1}{P} \approx \frac{1}{70,000,000,000}
$$



Without Reverse-Mode Automatic Differentiation, training Large Language Models would be physically impossible.

---

## Step 4: Where Did It Come From? (Linnainmaa, Rumelhart, & Hinton) {: #step-4 }

<dl>
  <dt><time datetime="1970">1970</time> &mdash; <strong>Seppo Linnainmaa</strong></dt>
  <dd>Introduced the general algorithm for reverse-mode automatic differentiation in his Master's thesis at the University of Helsinki, demonstrating that the derivatives of nested algebraic functions can be computed in time proportional to the forward evaluation.</dd>

  <dt><time datetime="1986">1986</time> &mdash; <strong>David Rumelhart, Geoffrey Hinton, & Ronald Williams</strong></dt>
  <dd>Published the landmark paper <cite>"Learning representations by back-propagating errors"</cite> in <em>Nature</em>. They showed that backpropagation enables multi-layer neural networks to learn internal representations of data, defeating the historic Minsky-Papert XOR pessimism that had frozen neural network research during the AI Winter.</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>PyTorch & Autograd</strong></dt>
  <dd>Dynamic computational graphs (tape-based autograd) popularized by Adam Paszke et al. enabled backpropagation through dynamic control flows, attention matrices, and autoregressive generation loops with zero manual gradient coding.</dd>
</dl>

---

## Step 5: Concrete Toy Example (Step-by-Step Hand Arithmetic) {: #step-5 }

Let us compute the complete forward pass, loss calculation, backward pass, and weight update for a miniature 2-layer network with tiny numbers.

### 1. Miniature Architecture & Parameters
- Input scalar: $x = 2.0$
- Layer 1 weight: $w_1 = 3.0$
- Hidden activation: $h = x \cdot w_1$
- Layer 2 weight: $w_2 = 2.0$
- Model prediction: $\hat{y} = h \cdot w_2$
- Target ground truth: $y = 10.0$
- Loss function (Squared Error): $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$
- Learning rate: $\eta = 0.01$

---

### 2. Forward Pass: Making the Prediction

<fieldset>
<legend><strong>Execution Checklist</strong></legend>
<p><input type="checkbox" checked disabled> <strong>Step A:</strong> Forward compute hidden value $h = x \cdot w_1$.</p>
<p><input type="checkbox" checked disabled> <strong>Step B:</strong> Forward compute prediction $\hat{y} = h \cdot w_2$.</p>
<p><input type="checkbox" checked disabled> <strong>Step C:</strong> Compute loss $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.</p>
<p><input type="checkbox" checked disabled> <strong>Step D:</strong> Backward compute $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ and weight gradient $\frac{\partial \mathcal{L}}{\partial w_2}$.</p>
<p><input type="checkbox" checked disabled> <strong>Step E:</strong> Backward propagate $\frac{\partial \mathcal{L}}{\partial h}$ and weight gradient $\frac{\partial \mathcal{L}}{\partial w_1}$.</p>
<p><input type="checkbox" checked disabled> <strong>Step F:</strong> Apply gradient descent update and verify loss reduction.</p>
</fieldset>

#### Step A: Layer 1 Output


$$
h = x \cdot w_1 = 2.0 \times 3.0 = 6.0
$$



#### Step B: Layer 2 Output


$$
\hat{y} = h \cdot w_2 = 6.0 \times 2.0 = 12.0
$$



#### Step C: Loss Calculation
The model predicted $12.0$, but the target was $10.0$:



$$
\mathcal{L} = \frac{1}{2}(12.0 - 10.0)^2 = \frac{1}{2}(2.0)^2 = 2.0000
$$



---

### 3. Backward Pass: Propagating the Error Upstream

#### Step D: Gradients at Layer 2
First, differentiate the loss with respect to the prediction $\hat{y}$:



$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = 12.0 - 10.0 = \mathbf{+2.0}
$$



Now, differentiate with respect to weight $w_2$:



$$
\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot h = 2.0 \times 6.0 = \mathbf{+12.0}
$$



Next, propagate the gradient backward into the hidden state $h$:



$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot w_2 = 2.0 \times 2.0 = \mathbf{+4.0}
$$



#### Step E: Gradients at Layer 1
Now compute the gradient for weight $w_1$ using the propagated signal $\frac{\partial \mathcal{L}}{\partial h}$:



$$
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot x = 4.0 \times 2.0 = \mathbf{+8.0}
$$



Summary of computed gradients:
- $\nabla_{w_2} \mathcal{L} = +12.0$ (push $w_2$ down!)
- $\nabla_{w_1} \mathcal{L} = +8.0$ (push $w_1$ down!)

---

### 4. Parameter Update & Verification

#### Step F: Apply Gradient Descent
With learning rate $\eta = 0.01$:



$$
w_{2,\text{new}} = w_2 - \eta \frac{\partial \mathcal{L}}{\partial w_2} = 2.0 - (0.01 \times 12.0) = 2.0 - 0.12 = \mathbf{1.88}
$$





$$
w_{1,\text{new}} = w_1 - \eta \frac{\partial \mathcal{L}}{\partial w_1} = 3.0 - (0.01 \times 8.0) = 3.0 - 0.08 = \mathbf{2.92}
$$



#### Step G: Verification with a New Forward Pass
Let us re-run the network with the newly updated weights:
- $h_{\text{new}} = x \cdot w_{1,\text{new}} = 2.0 \times 2.92 = 5.84$
- $\hat{y}_{\text{new}} = h_{\text{new}} \cdot w_{2,\text{new}} = 5.84 \times 1.88 = 10.9792$
- Target: $y = 10.0$

Calculate the new loss:



$$
\mathcal{L}_{\text{new}} = \frac{1}{2}(10.9792 - 10.0)^2 = \frac{1}{2}(0.9792)^2 \approx \mathbf{0.4794}
$$



<mark>The loss dropped from $2.0000$ to $0.4794$ &mdash; a dramatic $76.0\%$ error reduction in a single tiny step!</mark>

---

## Step 6: Core Takeaway {: #step-6 }

!!! tip "Key Insight: The Punchline of Backpropagation"
    **Backpropagation is reverse-mode automatic differentiation applied to the computational graph of a neural network.**

    By traversing the network backward from the single scalar loss to the billions of input weights, it computes the exact mountain slope for every parameter simultaneously in a single pass, turning an otherwise impossible trillion-hour training task into practical reality.
