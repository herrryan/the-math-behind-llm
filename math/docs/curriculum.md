# The Math Behind Large Language Models: A Master Curriculum



!!! note "3-Year-Old Intuition: How to Think About This Entire Guide"
    Imagine an LLM is a giant musical music box. Inside the music box are billions of tiny wooden knobs and gears. If you whisper three words into the funnel (*"The cat sat..."*), the gears spin smoothly and out pops a tiny slip of paper with the next word: (*"on"*).

    Mathematics is simply the blueprint of those knobs and gears. It tells us how the box turns words into numbers, how it remembers who did what, and how it adjusts its knobs so it stops making silly mistakes.

---

<h2 id="pedagogy">The Teaching Philosophy: The 5-Step Pedagogy</h2>

Every single chapter in this course follows an unshakeable 5-step learning ladder:

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table C.1:</strong> The 6-step pedagogical learning ladder applied to every chapter.</caption>
  <thead>
    <tr>
      <th align="center">Step</th>
      <th align="left">Section Title</th>
      <th align="left">What You Learn</th>
      <th align="left">Tactile Metaphor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Step 1</strong></td>
      <td><strong>3-Year-Old Intuition</strong></td>
      <td>Core physical intuition with zero mathematical jargon</td>
      <td>Mystery boxes, playground maps, spotlight beams, Play-Doh presses</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 2</strong></td>
      <td><strong>The Bridging Question</strong></td>
      <td>How to translate the physical game into computer numbers</td>
      <td>Moving from intuitive physical concepts to numerical matrices</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 3</strong></td>
      <td><strong>The Exact Math &amp; Formula</strong></td>
      <td>The genuine formula used in modern LLMs (Vaswani, LLaMA)</td>
      <td>Explicit Greek symbols ($\sum, \prod, \exp, \nabla$), subscripts, and matrix dimensions</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 4</strong></td>
      <td><strong>Where Did It Come From?</strong></td>
      <td>Historical and mathematical origin of the equation</td>
      <td>Who invented it, why this functional form, and what failed before</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 5</strong></td>
      <td><strong>Concrete Toy Example</strong></td>
      <td>Step-by-step arithmetic with tiny numbers ($2\text{D}/3\text{D}$ vectors)</td>
      <td>Hand-calculated operations verifying every addition and multiplication</td>
    </tr>
    <tr>
      <td align="center"><strong>Step 6</strong></td>
      <td><strong>Core Takeaway</strong></td>
      <td>Punchy 1–2 sentence summary of why this formula matters</td>
      <td>The mental anchor for the overall LLM brain</td>
    </tr>
  </tbody>
</table>

---

<h2 id="evolution">The 4-Stage Python Brain Evolution Chain</h2>

<p>Parallel to the theoretical mathematics, ambitious learners follow an unbroken, hands-on engineering track: <strong>The Python Brain Evolution Chain</strong>. We build, test, and grow a single, zero-dependency Python neural language model across four major milestone labs:</p>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>Table C.0:</strong> The 4-Stage Python Brain Evolution Milestone Roadmap.</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="18%">Evolution Stage</th>
      <th align="left" width="22%">Lab Location</th>
      <th align="left" width="30%">Architecture Milestone</th>
      <th align="left" width="30%">Bottleneck Unmasked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Stage 1: The Micro-Brain</strong></td>
      <td><a href="module-2/04b-lab-micro-brain.md">Lab 01 (Post Module 2)</a></td>
      <td>80 lines pure Python: Bengio 2003 MLP with embeddings, dense projection, ReLU, and manual backpropagation.</td>
      <td><mark>Severe Short-Term Amnesia</mark>: Markovian 1-word context horizon. Demands <strong>Attention</strong>.</td>
    </tr>
    <tr>
      <td><strong>Stage 2: The Attention Brain</strong></td>
      <td><a href="#">Lab 02 (Post Module 3)</a></td>
      <td>140 lines pure Python: Full self-attention with Query, Key, Value projections, scaled dot-products, and causal masking.</td>
      <td><mark>Depth Instability</mark>: Stacking deep attention blocks triggers vanishing gradients and scale blowups. Demands <strong>Residuals &amp; RMSNorm</strong>.</td>
    </tr>
    <tr>
      <td><strong>Stage 3: The Transformer Brain</strong></td>
      <td><a href="#">Lab 03 (Post Module 5)</a></td>
      <td>220 lines pure Python: Complete modern LLaMA-style Transformer block with SwiGLU gating, Pre-RMSNorm, and Residual highways.</td>
      <td><mark>Mechanical Generation</mark>: Plain greedy decoding produces rigid, repetitive text loops. Demands <strong>Probabilistic Sampling</strong>.</td>
    </tr>
    <tr>
      <td><strong>Stage 4: The Complete LLM</strong></td>
      <td><a href="#">Lab 04 (Post Module 7)</a></td>
      <td>300 lines pure Python: Production-style autoregressive inference engine with KV Cache acceleration and Top-p (Nucleus) sampling.</td>
      <td><mark>Completed Artifact</mark>: A fully interactive, zero-dependency LLM chat engine running in your terminal!</td>
    </tr>
  </tbody>
</table>

---

<h2 id="pipeline">Complete Curriculum Overview</h2>

Below is the complete roadmap of 19 chapters across 9 intuitive modules, covering the entire journey from raw words to deep alignment:

<figure>
<pre>
[00: Next-Word Prediction] ──► [01: Vectors & Embeddings] ──► [02: Dot Product & Similarity]
                                                                        │
                                                                        ▼
[05: Transformer Blueprint] ◄── [Lab 01: Micro-Brain] ◄── [04: Activations] ◄── [03: Matrix Mult]
         │
         ▼
[06: Queries, Keys, Values] ──► [07: Softmax] ──► [08: Attention Formula] ──► [09: Causal Masking]
                                                                                      │
                                                                                      ▼
[12: Residual Connections] ◄── [11: Multi-Head Attention] ◄── [10: Positional Encodings & RoPE]
         │
         ▼
[13: RMSNorm] ──► [14: Feed-Forward Blocks] ──► [15: Cross-Entropy Loss]
                                                        │
                                                        ▼
[18: Sampling (Temp/Top-p)] ◄── [17: Adam Optimizer] ◄── [16: Backpropagation]
         │
         ▼
[19: Alignment (RLHF & DPO)]
</pre>
<figcaption><strong>Figure C.1:</strong> Master architectural dataflow pipeline from raw text to aligned model.</figcaption>
</figure>

---

<h3 id="module-0">Module 0: The Big Picture (What is a Model, Really?)</h3>
*Foundational concept: LLMs as probabilistic token prediction engines.*

- [Chapter 00: The Next-Word Guessing Game](module-0/00-next-word-prediction.md)
  - **The Metaphor**: Guessing the hidden animal in the box using clues one by one.
  - **The Math**: Discrete probability distributions $P(w_t \mid w_{\lt t})$, vocabulary space $V$, and the chain rule of joint probability $\prod_{t=1}^T P(w_t \mid w_{\lt t})$.
  - **Formula Origin**: Andrey Markov (1913) counting letters in Pushkin's poem & Claude Shannon (1948) modeling sequence communication.

---

<h3 id="module-1">Module 1: Representing Words with Numbers (Vector Geometry)</h3>
*How human language becomes points in geometric space.*

- [Chapter 01: The Word Map (Vectors & Embeddings)](module-1/01-vectors-and-spaces.md)
  - **The Metaphor**: A giant playground map where similar toys sit in neighboring sandboxes.
  - **The Math**: Vector coordinates $\mathbf{x} \in \mathbb{R}^{d}$, one-hot vectors $\mathbf{e}_i$, and the embedding lookup matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d}$.
  - **Formula Origin**: Distributional Semantics (*"You shall know a word by the company it keeps"* — J.R. Firth, 1957) and Word2Vec (Mikolov et al., 2013).

- [Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity)](module-1/02-dot-product-and-similarity.md)
  - **The Metaphor**: Do two toy arrows point towards the same treasure chest?
  - **The Math**: Dot product $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$, Euclidean $L_2$ norm $\|\mathbf{u}\| = \sqrt{\sum u_i^2}$, and Cosine similarity $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$.
  - **Formula Origin**: Euclidean vector projections used as the universal affinity metric across all neural networks.

---

<h3 id="module-2">Module 2: Transforming Spaces (Linear Algebra & Activations)</h3>
*How the network bends, stretches, and filters information.*

- [Chapter 03: The Magic Stretching Box (Matrix Multiplication)](module-2/03-matrix-multiplication.md)
  - **The Metaphor**: A Play-Doh press that stretches, squashes, and turns shapes in space.
  - **The Math**: Linear transformation $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$, matrix dimension compatibility $(m \times k) \times (k \times n) \to (m \times n)$.
  - **Formula Origin**: Basis vector transformations; why parallel GEMM operations run at trillions of FLOPS on modern GPUs.

- [Chapter 04: The One-Way Gate (Activation Functions: ReLU, GELU, SwiGLU)](module-2/04-activation-functions.md)
  - **The Metaphor**: A secret one-way valve; why stacking flat sheets of glass never creates a curved magnifying lens.
  - **The Math**: Collapse of linear cascades $\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = \mathbf{W}_{\text{comb}} \mathbf{x}$. Non-linear remedies: ReLU $\max(0, x)$, GELU $x \Phi(x)$, and SwiGLU $\text{Swish}(\mathbf{x}\mathbf{W}) \odot (\mathbf{x}\mathbf{V})$.
  - **Formula Origin**: Biological neuron thresholding (McCulloch-Pitts 1943) evolved into smooth gating (Hendrycks 2016, Shazeer 2020).

- [Hands-on Lab 01: Training Your First Brain in 80 Lines of Pure Python (Bengio 2003)](module-2/04b-lab-micro-brain.md)
  - **The Metaphor**: The clockwork punched-card music box that learns to strike tuned bells.
  - **The Math**: End-to-end integration of Chapters 00–04 in pure Python: $\mathbf{x} = \mathbf{e}_i^\top \mathbf{E}$, $\mathbf{z}_1 = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1$, $\mathbf{a}_1 = \operatorname{ReLU}(\mathbf{z}_1)$, $\mathbf{z}_2 = \mathbf{a}_1 \mathbf{W}_2 + \mathbf{b}_2$, $\mathcal{L} = -\log(\hat{y})$, and manual reverse-mode gradient updates $w \leftarrow w - \eta \nabla_w \mathcal{L}$.
  - **The Python Brain Evolution**: Stage 1 of 4. Proves a zero-dependency neural network can learn grammar and predict next words, while exposing the fatal 1-word horizon (amnesia) that necessitates Attention.

---

<h3 id="module-3">Module 3: The Secret Sauce (The Attention Mechanism)</h3>
*The engine of the Transformer: dynamic, context-aware information routing.*

- [Chapter 05: The Transformer Blueprint (The Bird's-Eye View & The Round Table)](module-3/05-transformer-architecture.md)
  - **The Metaphor**: The round-table conference where every scholar can make direct eye contact with everyone else, versus the whispering telephone game (RNNs) and horse blinkers (MLPs).
  - **The Math**: Macro-architecture of modern decoder-only LLMs: Input embeddings $\mathbf{X}^{(0)} \in \mathbb{R}^{T \times d_{\text{model}}}$, stacked Transformer blocks alternating between communication (Self-Attention) and thinking (FFN/SwiGLU) with residual streams $\mathbf{X}^{(l)} = \mathbf{X}^{(l-1)} + \text{Sublayer}(\text{RMSNorm}(\mathbf{X}^{(l-1)}))$, and unembedding projection to vocabulary logits $\mathbf{z} \in \mathbb{R}^{T \times |V|}$.
  - **Formula Origin**: The information bottleneck of Seq2Seq RNNs (Sutskever, Cho 2014) broken by *Attention Is All You Need* (Vaswani et al. 2017) and refined into the modern decoder-only standard (Radford 2018, Touvron 2023).

- [Chapter 06: The Library Clue Hunt (Queries, Keys, and Values)](module-3/06-queries-keys-values.md)
  - **The Metaphor**: You hold a clue card (Query), shelves have label cards (Keys), and books contain treasure stories (Values).
  - **The Math**: Projection matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$ and projection outputs $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \mathbf{K} = \mathbf{X}\mathbf{W}_K, \mathbf{V} = \mathbf{X}\mathbf{W}_V$.
  - **Formula Origin**: Database query-key-value retrieval made fully differentiable for gradient descent.

- [Chapter 07: The Fair Voting Booth (The Softmax Function)](module-3/07-softmax-function.md)
  - **The Metaphor**: Turning loud shouting matches into fair percentages of a pizza that sum to exactly 100%.
  - **The Math**: $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}$.
  - **Formula Origin**: Ludwig Boltzmann's statistical mechanics (1868) for particle thermal states adapted by Luce (1959).

- [Chapter 08: The Attention Formula & Why We Divide by $\sqrt{d_k}$](module-3/08-attention-formula.md)
  - **The Metaphor**: Why whispering in a crowded hall needs a volume damper so the loudest child doesn't drown out everyone else.
  - **The Math**: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$.
  - **Formula Origin**: Vaswani et al. (2017). Dot product variance scales as $d_k$; dividing by $\sqrt{d_k}$ prevents vanishing softmax gradients.

- [Chapter 09: Blindfolds on Future Words (Causal Masking)](module-4/09-causal-masking.md)
  - **The Metaphor**: Taking a reading quiz without being allowed to peek at the answers on tomorrow's page.
  - **The Math**: Mask matrix $\mathbf{M} \in \mathbb{R}^{T \times T}$ where $M_{ij} = -\infty$ for $j > i$. Since $e^{-\infty} = 0$, attention weights to future tokens strictly vanish.
  - **Formula Origin**: Enforcing the autoregressive causal condition $P(w_t \mid w_{\lt t})$ in self-attention matrices.

---

<h3 id="module-4">Module 4: Order and Multi-Perspective Processing</h3>
*Teaching transformers word order and looking at sentences through multiple lenses.*

- [Chapter 10: Where in the Sentence Am I? (Positional Encodings & RoPE)](module-4/10-positional-encodings.md)
  - **The Metaphor**: Number tags on runner shirts and clock hands spinning at different speeds.
  - **The Math**: Sinusoidal encodings $\sin(\text{pos}/10000^{2i/d})$ and Rotary Position Embedding (RoPE) 2D rotation blocks $\mathbf{R}_{\Theta, m}$.
  - **Formula Origin**: Fourier harmonic analysis (Vaswani 2017) and complex geometric rotations (Su et al. 2021).

- [Chapter 11: Looking Through Different Glasses (Multi-Head Attention)](module-4/11-multi-head-attention.md)
  - **The Metaphor**: A detective squad: one tracks grammar, one tracks feelings, one tracks timestamps.
  - **The Math**: $\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}_O$ with subspace dimension $d_k = d_{\text{model}} / h$.
  - **Formula Origin**: Subspace ensemble projections without increasing total parameter counts or FLOPs.

---

<h3 id="module-5">Module 5: Putting the Transformer Block Together</h3>
*How modern transformers stack 100+ layers without exploding or collapsing.*

- [Chapter 12: The Shortcut Bridge (Residual Connections)](module-5/12-residual-connections.md)
  - **The Metaphor**: An express highway running alongside a twisty mountain path so you never get stuck in a traffic jam.
  - **The Math**: $\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \mathcal{F}(\mathbf{x}_{\text{in}})$. Gradient propagation: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{in}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} \left(\mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_{\text{in}}}\right)$.
  - **Formula Origin**: ResNet (He et al., 2015) solving the vanishing gradient dilemma in deep architectures.

- [Chapter 13: Keeping Everyone Calm (LayerNorm & RMSNorm)](module-5/13-layer-norm-and-rmsnorm.md)
  - **The Metaphor**: A volume limiter in a concert hall that keeps sound at a pleasant listening level so speakers never pop.
  - **The Math**: $\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$ and $\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$.
  - **Formula Origin**: Ba, Kiros, & Hinton (2016) and Zhang & Sennrich (2019) demonstrating root-mean-square scale invariance.

- [Chapter 14: The Thinking Chamber (Feed-Forward Networks)](module-5/14-feed-forward-networks.md)
  - **The Metaphor**: Sitting quietly in your room to digest what you just heard and store facts into long-term memory.
  - **The Math**: $\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$. Projection from $d_{\text{model}} \to 4d_{\text{model}} \to d_{\text{model}}$.
  - **Formula Origin**: Universal approximation theorem; associative key-value memory retrieval in MLP layers.

---

<h3 id="module-6">Module 6: How the Model Learns (Training Math)</h3>
*How billions of knobs are tuned from trillions of text tokens.*

- [Chapter 15: How Wrong Was I? (Cross-Entropy Loss & Perplexity)](module-6/15-cross-entropy-loss.md)
  - **The Metaphor**: The penalty points you get during a guessing game when your guess was far off the mark.
  - **The Math**: Cross-entropy $\mathcal{L} = -\sum_{i=1}^{|V|} y_i \log p_i = -\log p_{\text{correct}}$ and Perplexity $\text{PPL} = e^{\mathcal{L}}$.
  - **Formula Origin**: Information theory (Shannon entropy 1948) and Kullback-Leibler (KL) divergence minimization.

- [Chapter 16: Walking Down the Mountain (Gradients & Backpropagation)](module-6/16-gradients-and-backpropagation.md)
  - **The Metaphor**: Feeling the steepness of the ground with your toes while walking down a foggy mountain.
  - **The Math**: Multivariable gradients $\nabla_\theta \mathcal{L}$, chain rule $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{W}}$, gradient descent step $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$.
  - **Formula Origin**: Leibniz calculus meets reverse-mode automatic differentiation (Rumelhart, Hinton, & Williams 1986).

- [Chapter 17: The Smart Walker (Momentum and the Adam Optimizer)](module-6/17-adam-optimizer.md)
  - **The Metaphor**: A heavy bowling ball rolling downhill (momentum) wearing shoes that adjust their grip depending on mud vs rocks (adaptive rates).
  - **The Math**: First moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, updates $\theta_{t} = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$.
  - **Formula Origin**: Kingma & Ba (2014) synthesizing AdaGrad coordinate scaling and Polyak heavy-ball momentum.

---

<h3 id="module-7">Module 7: Speaking to the World (Inference & Sampling)</h3>
*How the model turns probabilities into fluid sentences.*

- [Chapter 18: Turning Up the Heat (Temperature, Top-k, & Top-p Sampling)](module-7/18-sampling-and-temperature.md)
  - **The Metaphor**: A creativity knob: freezing cold gives the safest, most boring answer; boiling hot gives wild, unpredictable dreams.
  - **The Math**: Temperature scaling $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$, Top-$k$ restriction, and Top-$p$ (Nucleus) cumulative cutoff $\sum_{i \in V^{(p)}} p_i \ge p$.
  - **Formula Origin**: Statistical physics annealing and Holtzman et al. (2019) truncating the degenerated distribution tail.

---

<h3 id="module-8">Module 8: Aligning and Refining (Post-Training Math)</h3>
*Teaching the model to be helpful, honest, and harmless.*

- [Chapter 19: Teaching Good Manners (RLHF, Reward Modeling, and DPO)](module-8/19-rlhf-and-dpo.md)
  - **The Metaphor**: Giving gold stars for kind answers, gentle penalties for rude answers, and keeping the child from forgetting who they are.
  - **The Math**: Reward objective with KL penalty $\mathbb{E}[r_\theta(x, y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$, and Direct Preference Optimization (DPO) closed-form loss:
    $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
  - **Formula Origin**: Bradley-Terry preference models (1952) and Rafailov et al. (2023) analytically bypassing the reinforcement learning actor-critic loop.

---

## How This Course is Engineered

- **100% Focused on Content**: Content is authored in standard, clean Markdown (`.md`).
- **Dynamic In-Browser Engine**: Rendered on the fly in the browser using `marked.js` and `KaTeX` — zero compiler scripts or build commands needed.
- **Pure Semantic HTML (Zero Custom CSS)**: Relies strictly on native browser elements (`<header>`, `<nav>`, `<main>`, `<fieldset>`, `<legend>`, `<table border="1">`, `<footer>`).
- **Granular Git Version Control**: Every turn and chapter iteration is committed into Git for complete history.

---

!!! tip "Key Insight: Ready to Begin the Journey?"
    Dive straight into [Chapter 00: The Next-Word Guessing Game](module-0/00-next-word-prediction.md) to build your first mathematical foundation!
