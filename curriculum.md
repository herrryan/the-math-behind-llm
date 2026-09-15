# The Math Behind Large Language Models: A Master Curriculum

> [!INTUITION] How to Think About This Entire Guide
> Imagine an LLM is a giant musical music box. Inside the music box are billions of tiny wooden knobs and gears. If you whisper three words into the funnel ("The cat sat..."), the gears spin smoothly and out pops a tiny slip of paper with the next word: ("on").
> 
> Mathematics is simply the blueprint of those knobs and gears. It tells us how the box turns words into numbers, how it remembers who did what, and how it adjusts its knobs so it stops making silly mistakes.

---

## 🧭 The Teaching Philosophy: The 5-Step Pedagogy

Every single chapter in this course follows an unshakeable 5-step learning ladder:

| Step | Section Title | What You Learn |
| :--- | :--- | :--- |
| **1** | **🧸 3-Year-Old Intuition** | A tactile, physical metaphor (colored blocks, spotlights, rubber bands, pizza slices) that requires zero math background. |
| **2** | **🌉 The Bridging Question** | The exact problem that arises when we try to build this physical idea in a computer. |
| **3** | **📐 The Exact Math & Formula** | The precise mathematical equation, with every single symbol defined clearly. |
| **4** | **🔍 Where Did It Come From?** | The historical and mathematical origin: why this formula looks like this, and what went wrong with simpler ideas. |
| **5** | **🔢 Concrete Toy Example** | Step-by-step arithmetic with tiny numbers (vectors of 2 or 3 numbers), showing you every addition and multiplication. |

---

## 📚 Complete Curriculum Overview

Below is the complete roadmap of 19 chapters across 9 intuitive modules, covering the entire journey from raw words to deep alignment.

```
[Raw Words] ➔ [Word Embeddings] ➔ [Self-Attention] ➔ [Feed-Forward] ➔ [Logits] ➔ [Probabilities]
     ▲                                                                                 │
     └───────────── [Backpropagation & Adam Loss Optimization] ◄───────────────────────┘
```

---

### Module 0: The Big Picture (What is a Model, Really?)
*Foundational concept: LLMs as probability engines.*

- [Chapter 00: The Next-Word Guessing Game](00-next-word-prediction/index.html)
  - **The Metaphor**: Guessing the hidden animal in the box using clues one by one.
  - **The Math**: Discrete probability distributions $P(w_t \mid w_1, \dots, w_{t-1})$, vocabulary space $V$, and the chain rule of joint probability.
  - **Formula Origin**: Andrei Markov (1913) & Claude Shannon (1948) modeling sequences as conditional probability chains.

---

### Module 1: Representing Words with Numbers (Vector Geometry)
*How human language becomes points in geometric space.*

- [Chapter 01: The Word Map (Vectors & Embeddings)](01-vectors-and-spaces/index.html)
  - **The Metaphor**: A giant playground map where similar toys sit in neighboring sandboxes.
  - **The Math**: Vector coordinates $\mathbf{x} \in \mathbb{R}^d$, one-hot vectors $\mathbf{e}_i$, and the embedding lookup matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d}$.
  - **Formula Origin**: Distributional Semantics ("You shall know a word by the company it keeps" - J.R. Firth, 1957) and Word2Vec (Mikolov et al., 2013).

- [Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity)](02-dot-product-and-similarity/index.html)
  - **The Metaphor**: Do two toy arrows point towards the same treasure?
  - **The Math**: Dot product $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$, vector length (Euclidean $L_2$ norm $\|\mathbf{u}\| = \sqrt{\sum u_i^2}$), and Cosine similarity $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$.
  - **Formula Origin**: Vector projections from 19th-century Euclidean geometry used as the fundamental affinity metric in modern machine learning.

---

### Module 2: Transforming Spaces (Linear Algebra & Activations)
*How the network bends, stretches, and filters information.*

- [Chapter 03: The Magic Stretching Box (Matrix Multiplication)](03-matrix-multiplication/index.html)
  - **The Metaphor**: A Play-Doh press that stretches, squashes, and turns shapes in space.
  - **The Math**: Linear transformation $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$, matrix dimension compatibility $(m \times k) \times (k \times n) \to (m \times n)$.
  - **Formula Origin**: Linear combinations of basis vectors; why matrix multiplication is the fastest operation on modern GPUs.

- [Chapter 04: The One-Way Gate (Activation Functions: ReLU, GELU, SwiGLU)](04-activation-functions/index.html)
  - **The Metaphor**: A secret one-way valve; why stacking flat sheets of glass never creates a curved magnifying lens.
  - **The Math**: The collapse of linear cascades $\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = \mathbf{W}_{\text{comb}} \mathbf{x}$. Non-linear remedies: ReLU $\max(0, x)$, GELU $x \Phi(x)$, and SwiGLU $\text{Swish}(\mathbf{x}\mathbf{W}) \odot (\mathbf{x}\mathbf{V})$.
  - **Formula Origin**: Biological neuron thresholding (McCulloch-Pitts 1943) to modern probabilistic gating (Hendrycks & Gimpel 2016, Shazeer 2020).

---

### Module 3: The Secret Sauce (The Attention Mechanism)
*The engine of the Transformer: dynamic, context-aware information routing.*

- [Chapter 05: The Library Clue Hunt (Queries, Keys, and Values)](05-queries-keys-values/index.html)
  - **The Metaphor**: You hold a clue card (Query), library shelves have category labels (Keys), and inside each book is the actual treasure story (Value).
  - **The Math**: Projection matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$ and transformed matrices $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \mathbf{K} = \mathbf{X}\mathbf{W}_K, \mathbf{V} = \mathbf{X}\mathbf{W}_V$.
  - **Formula Origin**: Information retrieval systems (database query-key matching) made differentiable for neural networks.

- [Chapter 06: The Fair Voting Booth (The Softmax Function)](06-softmax-function/index.html)
  - **The Metaphor**: Turning loud shouting matches into percentages of a pizza that sum up to exactly 100%.
  - **The Math**: $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}$.
  - **Formula Origin**: Ludwig Boltzmann's statistical physics (1868) for particle energy distributions, adapted by Luce (1959) for choice probabilities.

- [Chapter 07: The Attention Formula & Why We Divide by $\sqrt{d_k}$](07-attention-formula/index.html)
  - **The Metaphor**: Why whispering in a crowded hall needs a volume damper to prevent the loudest child from drowning out everyone else.
  - **The Math**: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$.
  - **Formula Origin**: Vaswani et al. (2017). Proof of variance growth: if components have unit variance, their dot product has variance $d_k$; dividing by $\sqrt{d_k}$ keeps variance at 1 and stops softmax gradients from dying.

- [Chapter 08: Blindfolds on Future Words (Causal Masking)](08-causal-masking/index.html)
  - **The Metaphor**: Taking a reading quiz without being allowed to peek at the answer key on the next page.
  - **The Math**: Attention mask $\mathbf{M} \in \mathbb{R}^{T \times T}$ where $M_{ij} = -\infty$ for $j > i$. Since $e^{-\infty} = 0$, attention weights to future tokens strictly vanish.
  - **Formula Origin**: Autoregressive modeling constraint: $P(w_t \mid w_{<t})$ must depend only on observed history.

---

### Module 4: Order and Multi-Perspective Processing
*Teaching transformers about word order and looking at sentences through multiple lenses.*

- [Chapter 09: Where in the Sentence Am I? (Positional Encodings & RoPE)](09-positional-encodings/index.html)
  - **The Metaphor**: Number tags on runner shirts and clock hands turning at different speeds.
  - **The Math**: Sinusoidal encodings $\sin(\text{pos}/10000^{2i/d})$ and Rotary Position Embedding (RoPE) 2D rotation blocks $\mathbf{R}_{\Theta, m}$.
  - **Formula Origin**: Fourier basis encodings (Vaswani et al. 2017) and complex number geometry for relative distance preservation (Su et al. 2021).

- [Chapter 10: Looking Through Different Glasses (Multi-Head Attention)](10-multi-head-attention/index.html)
  - **The Metaphor**: A team of detectives reading the same text: one tracks grammar, one tracks feelings, one tracks timestamps.
  - **The Math**: $\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}_O$ with sub-space dimension $d_k = d_{\text{model}} / h$.
  - **Formula Origin**: Ensemble subspace projections without increasing total computation cost.

---

### Module 5: Putting the Transformer Block Together
*How modern transformers stack layers without exploding or collapsing.*

- [Chapter 11: The Shortcut Bridge (Residual Connections)](11-residual-connections/index.html)
  - **The Metaphor**: An express highway alongside a scenic mountain trail so you never get stuck.
  - **The Math**: $\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \mathcal{F}(\mathbf{x}_{\text{in}})$. Gradient flow: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{in}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} \left(\mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_{\text{in}}}\right)$.
  - **Formula Origin**: ResNet (He et al., 2015) solving the vanishing gradient dilemma in deep networks.

- [Chapter 12: Keeping Everyone Calm (LayerNorm & RMSNorm)](12-layer-norm-and-rmsnorm/index.html)
  - **The Metaphor**: A volume knob in a concert hall that keeps sound at a pleasant listening level so no speaker blows up.
  - **The Math**: $\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$ and $\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$.
  - **Formula Origin**: Ba, Kiros, & Hinton (2016) and Zhang & Sennrich (2019) proving scale invariance improves training stability.

- [Chapter 13: The Thinking Chamber (Feed-Forward Networks)](13-feed-forward-networks/index.html)
  - **The Metaphor**: Sitting quietly in your room to digest what you just heard and store facts into memory.
  - **The Math**: $\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$. Projection from $d_{\text{model}} \to 4d_{\text{model}} \to d_{\text{model}}$.
  - **Formula Origin**: Universal approximation theorem; key-value factual memory storage in MLP weights.

---

### Module 6: How the Model Learns (Training Math)
*How billions of knobs are tuned from trillions of text tokens.*

- [Chapter 14: How Wrong Was I? (Cross-Entropy Loss & Perplexity)](14-cross-entropy-loss/index.html)
  - **The Metaphor**: The penalty points you get during a guessing game when your guess was far off.
  - **The Math**: Cross-entropy $\mathcal{L} = -\sum_{i=1}^{|V|} y_i \log p_i = -\log p_{\text{correct}}$ and Perplexity $\text{PPL} = e^{\mathcal{L}}$.
  - **Formula Origin**: Information theory (Shannon entropy 1948) and Kullback-Leibler (KL) divergence minimization.

- [Chapter 15: Walking Down the Mountain (Gradients & Backpropagation)](15-gradients-and-backpropagation/index.html)
  - **The Metaphor**: Feeling the steepness of the slope with your toes while walking down a foggy mountain.
  - **The Math**: Multivariable gradients $\nabla_\theta \mathcal{L}$, chain rule $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{W}}$, gradient descent step $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.
  - **Formula Origin**: Calculus (Leibniz & Newton) meets algorithmic reverse-mode automatic differentiation (Rumelhart, Hinton, & Williams 1986).

- [Chapter 16: The Smart Walker (Momentum and the Adam Optimizer)](16-adam-optimizer/index.html)
  - **The Metaphor**: A heavy bowling ball rolling downhill (momentum) wearing shoes that adjust their grip depending on mud vs rocks (adaptive learning rates).
  - **The Math**: First moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, bias-corrected updates $\theta_{t} = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$.
  - **Formula Origin**: Kingma & Ba (2014) combining AdaGrad's coordinate scaling and Polyak's heavy-ball momentum.

---

### Module 7: Speaking to the World (Inference & Sampling)
*How the model turns probabilities into fluid sentences.*

- [Chapter 17: Turning Up the Heat (Temperature, Top-k, & Top-p Sampling)](17-sampling-and-temperature/index.html)
  - **The Metaphor**: A creativity dial: freezing cold gives the safest, most boring answer; boiling hot gives wild, unpredictable dreams.
  - **The Math**: Temperature scaling $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$, Top-$k$ restriction, and Top-$p$ (Nucleus) cumulative cutoff $\sum_{i \in V^{(p)}} p_i \ge p$.
  - **Formula Origin**: Simulated annealing in physics and Holtzman et al. (2019) observing text degeneration in the distribution tail.

---

### Module 8: Aligning and Refining (Post-Training Math)
*Teaching the model to be helpful, honest, and harmless.*

- [Chapter 18: Teaching Good Manners (RLHF, Reward Modeling, and DPO)](18-rlhf-and-dpo/index.html)
  - **The Metaphor**: Giving gold stars for kind answers, gentle penalties for rude answers, and keeping the child from forgetting who they are.
  - **The Math**: Reward objective with KL penalty $\mathbb{E}[r_\theta(x, y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$, and Direct Preference Optimization (DPO) closed-form loss:
    $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
  - **Formula Origin**: Bradley-Terry preference models (1952) and Rafailov et al. (2023) analytically bypassing the reinforcement learning actor-critic loop.

---

## 🛠️ How This Website is Built

- **Focus on Content**: Content is written in standard Markdown (`.md`).
- **Dynamic In-Browser Rendering**: Rendered on the fly in the browser using `marked.js` and `KaTeX` — zero build commands or compiler scripts needed.
- **Pure Semantic HTML**: No custom CSS stylesheets; relies on native browser elements (`<header>`, `<nav>`, `<main>`, `<fieldset>`, `<legend>`, `<table>`).
- **Version Controlled**: Every iteration is committed into Git for complete traceability.

---

> [!TIP] Ready to Begin?
> Start right away with [Chapter 00: The Next-Word Guessing Game](00-next-word-prediction/index.html) to build your first physical intuition!
