# The Math Behind Large Language Models

An intuitive, rigorous guide to understanding the mathematical foundations of Large Language Models (LLMs).

## Purpose & Pedagogy

This project demystifies the mathematics behind modern transformers and generative models by bridging two worlds:
1. **Physical 3-year-old intuition**: Visual, tangible metaphors (mystery boxes, playground maps, spotlight beams, volume dials).
2. **Mathematical rigor**: Explicit formulas, origins of equations, and concrete arithmetic examples.

Every chapter is structured around the **5-Step Learning Ladder**:
1. **3-Year-Old Intuition**
2. **The Bridging Question**
3. **The Exact Math & Formula**
4. **Where Did It Come From?**
5. **Concrete Toy Example**
6. **Core Takeaway**

---

## Architecture: Dynamic Markdown + KaTeX (Zero Custom CSS)

- **Pure Semantic HTML**: No custom CSS stylesheets; relies on native browser elements (`<header>`, `<nav>`, `<main>`, `<fieldset>`, `<legend>`, `<table border="1" cellpadding="8">`).
- **Dynamic In-Browser Rendering**: Uses `marked.js` with `marked-katex-extension` and `KaTeX` to render Markdown and math equations directly in the browser on page load.
- **Zero Build Commands**: Focus 100% on authoring Markdown (`.md`) without running compiler scripts.
- **Clean Academic Typography**: Strictly zero emojis; relies on clean structure, diagrams, and mathematical clarity.
- **Organized Subdirectories**: Each chapter lives in its own dedicated folder (e.g., `00-next-word-prediction/index.md`).

---

## Quickstart

Run a local server from the project root:
```bash
python3 -m http.server 8000
# Then visit http://localhost:8000
```
Or open any `index.html` directly in your browser. Whenever you edit `index.md`, simply refresh the browser to see your changes!

---

## Curriculum Outline

- **Module 0: The Big Picture (What is a Model, Really?)**
  - [Chapter 00: The Next-Word Guessing Game](00-next-word-prediction/index.html)
- **Module 1: Representing Words with Numbers (Vector Geometry)**
  - [Chapter 01: The Word Map (Vectors & Embeddings)](01-vectors-and-spaces/index.html)
  - [Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity)](02-dot-product-and-similarity/index.html)
- **Module 2: Transforming Spaces (Linear Algebra & Activations)**
  - [Chapter 03: The Magic Stretching Box (Matrix Multiplication)](03-matrix-multiplication/index.html)
  - [Chapter 04: The One-Way Gate (Activation Functions: ReLU, GELU, SwiGLU)](04-activation-functions/index.html)
  - [Hands-on Lab 01: Training Your First Brain in 80 Lines of Pure Python (Bengio 2003)](04b-lab-micro-brain/index.html)
- **Module 3: The Secret Sauce (The Attention Mechanism)**
  - [Chapter 05: The Library Clue Hunt (Queries, Keys, and Values)](05-queries-keys-values/index.html)
  - [Chapter 06: The Fair Voting Booth (The Softmax Function)](06-softmax-function/index.html)
  - [Chapter 07: The Attention Formula & Why We Divide by $\sqrt{d_k}$](07-attention-formula/index.html)
  - [Chapter 08: Blindfolds on Future Words (Causal Masking)](08-causal-masking/index.html)
- **Module 4: Order and Multi-Perspective Processing**
  - [Chapter 09: Where in the Sentence Am I? (Positional Encodings & RoPE)](09-positional-encodings/index.html)
  - [Chapter 10: Looking Through Different Glasses (Multi-Head Attention)](10-multi-head-attention/index.html)
- **Module 5: Putting the Transformer Block Together**
  - [Chapter 11: The Shortcut Bridge (Residual Connections)](11-residual-connections/index.html)
  - [Chapter 12: Keeping Everyone Calm (LayerNorm & RMSNorm)](12-layer-norm-and-rmsnorm/index.html)
  - [Chapter 13: The Thinking Chamber (Feed-Forward Networks)](13-feed-forward-networks/index.html)
- **Module 6: How the Model Learns (Training Math)**
  - [Chapter 14: How Wrong Was I? (Cross-Entropy Loss & Perplexity)](14-cross-entropy-loss/index.html)
  - [Chapter 15: Walking Down the Mountain (Gradients & Backpropagation)](15-gradients-and-backpropagation/index.html)
  - [Chapter 16: The Smart Walker (Momentum and the Adam Optimizer)](16-adam-optimizer/index.html)
- **Module 7: Speaking to the World (Inference & Sampling)**
  - [Chapter 17: Turning Up the Heat (Temperature, Top-k, & Top-p Sampling)](17-sampling-and-temperature/index.html)
- **Module 8: Aligning and Refining (Post-Training Math)**
  - [Chapter 18: Teaching Good Manners (RLHF, Reward Modeling, and DPO)](18-rlhf-and-dpo/index.html)
