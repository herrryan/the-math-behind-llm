# The Math Behind Large Language Models

<p><em>An intuitive, rigorous, and comprehensive guide to understanding the mathematical foundations of Large Language Models.</em></p>

---

## 1. Core Teaching Philosophy: The 5-Step Learning Ladder

Every single chapter in this course follows an unshakeable 5-step learning ladder, bridging physical intuition with mathematical rigor:

| Step | Section Title | What You Learn | Tactile Metaphor |
| :---: | :--- | :--- | :--- |
| **Step 1** | **3-Year-Old Intuition** | Physical intuition with zero mathematical jargon | Mystery boxes, playground maps, spotlight beams |
| **Step 2** | **The Bridging Question** | Translating physical concepts into computer matrices | Connecting physical games to linear arrays |
| **Step 3** | **Exact Math & Formulas** | Genuine equations used in modern LLMs (LLaMA, Transformers) | Explicit Greek symbols ($\sum, \prod, \exp, \nabla$), shapes & dims |
| **Step 4** | **Where Did It Come From?** | Historical origins, design rationale & failed attempts | Boltzmann (1868), Shannon (1948), Vaswani (2017) |
| **Step 5** | **Concrete Toy Example** | Step-by-step arithmetic with tiny numbers (dim 2 or 3) | Hand-calculated operations verifying every math step |
| **Step 6** | **Core Takeaway** | Punchy 1–2 sentence summary of architectural role | Mental anchor for the overall model brain |

---

## 2. Complete Curriculum Roadmap

The course is structured into **9 progressive modules** spanning from next-word probability to post-training alignment:

```
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
```

### Module Quick Navigation

- **Module 0: The Big Picture**
  - [Chapter 00: The Next-Word Guessing Game](module-0/00-next-word-prediction.md)
- **Module 1: Vector Geometry & Embeddings**
  - [Chapter 01: The Word Map (Vectors & Embeddings)](module-1/01-vectors-and-spaces.md)
  - [Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity)](module-1/02-dot-product-and-similarity.md)
- **Module 2: Transforming Spaces & Activations**
  - [Chapter 03: The Magic Stretching Box (Matrix Multiplication)](module-2/03-matrix-multiplication.md)
  - [Chapter 04: The One-Way Gate (ReLU, GELU, SwiGLU)](module-2/04-activation-functions.md)
  - [Hands-on Lab 01: The Micro-Brain in 80 Lines of Python](labs/01-micro-brain.md)
- **Module 3: The Attention Mechanism**
  - [Chapter 05: The Transformer Blueprint & 70B Parameter Census](module-3/05-transformer-architecture.md)
  - [Chapter 06: The Library Clue Hunt (Queries, Keys, Values)](module-3/06-queries-keys-values.md)
  - [Chapter 07: The Fair Voting Booth (The Softmax Function)](module-3/07-softmax-function.md)
  - [Chapter 08: The Attention Formula & Division by $\\sqrt{d_k}$](module-3/08-attention-formula.md)
- **Module 4: Positional Encodings & Multi-Head Attention**
  - [Chapter 09: Blindfolds on Future Words (Causal Masking)](module-4/09-causal-masking.md)
  - [Chapter 10: Where in the Sentence Am I? (RoPE)](module-4/10-positional-encodings.md)
  - [Chapter 11: Looking Through Different Glasses (Multi-Head Attention)](module-4/11-multi-head-attention.md)
- **Module 5: Residuals, Normalization & Feed-Forward**
  - [Chapter 12: The Shortcut Bridge (Residual Connections)](module-5/12-residual-connections.md)
  - [Chapter 13: Keeping Everyone Calm (LayerNorm & RMSNorm)](module-5/13-layer-norm-and-rmsnorm.md)
  - [Chapter 14: The Thinking Chamber (Feed-Forward Networks)](module-5/14-feed-forward-networks.md)
- **Module 6: Training, Loss & Optimization**
  - [Chapter 15: How Wrong Was I? (Cross-Entropy Loss & Perplexity)](module-6/15-cross-entropy-loss.md)
  - [Chapter 16: Walking Down the Mountain (Gradients & Backpropagation)](module-6/16-gradients-and-backpropagation.md)
  - [Chapter 17: The Smart Walker (Momentum & AdamW)](module-6/17-adam-optimizer.md)
- **Module 7: Inference & Sampling**
  - [Chapter 18: Turning Up the Heat (Temperature, Top-k, Top-p)](module-7/18-sampling-and-temperature.md)
- **Module 8: Alignment & Post-Training**
  - [Chapter 19: Teaching Good Manners (RLHF & DPO)](module-8/19-rlhf-and-dpo.md)
