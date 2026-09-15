# The Math Behind Large Language Models

An intuitive, rigorous, zero-JavaScript guide to understanding the mathematical foundations of Large Language Models (LLMs).

## 🎯 Purpose & Pedagogy

This project demystifies the mathematics behind modern transformers and generative models by bridging two worlds:
1. **Physical 3-year-old intuition**: Visual, tangible metaphors (mystery boxes, playground maps, spotlight beams, volume dials).
2. **Mathematical rigor**: Explicit formulas, origins of equations, and concrete arithmetic examples.

Every chapter is structured around the **5-Step Learning Ladder**:
1. 🧸 **3-Year-Old Intuition**
2. 🌉 **The Bridging Question**
3. 📐 **The Exact Math & Formula**
4. 🔍 **Where Did It Come From?**
5. 🔢 **Concrete Toy Example**

---

## 🌐 Architecture: Dynamic Markdown + KaTeX (Zero Custom CSS)

- **Pure Semantic HTML**: No custom CSS stylesheets; relies on native browser elements (`<header>`, `<nav>`, `<main>`, `<fieldset>`, `<legend>`, `<table>`).
- **Dynamic In-Browser Rendering**: Uses `marked.js` with `marked-katex-extension` and `KaTeX` to render Markdown and math equations directly in the browser on page load.
- **Zero Build Commands**: Focus 100% on authoring Markdown (`.md`) without running compiler scripts.
- **Organized Subdirectories**: Each chapter lives in its own dedicated folder (e.g., `00-next-word-prediction/index.md`).

---

## 🚀 Quickstart

Run a local server from the project root:
```bash
python3 -m http.server 8000
# Then visit http://localhost:8000
```
Or open any `index.html` directly in your browser. Whenever you edit `index.md`, simply refresh the browser to see your changes!

---

## 📖 Curriculum Outline

- **Module 0: The Big Picture**
  - Chapter 00: The Next-Word Guessing Game
- **Module 1: Vector Geometry (Words to Numbers)**
  - Chapter 01: The Word Map (Vectors & Embeddings)
  - Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity)
- **Module 2: Transforming Spaces (Linear Algebra & Activations)**
  - Chapter 03: The Magic Stretching Box (Matrix Multiplication)
  - Chapter 04: The One-Way Gate (ReLU, GELU, SwiGLU)
- **Module 3: The Secret Sauce (Attention Mechanism)**
  - Chapter 05: The Library Clue Hunt (Queries, Keys, Values)
  - Chapter 06: The Fair Voting Booth (Softmax Function)
  - Chapter 07: The Attention Formula & Why We Divide by $\sqrt{d_k}$
  - Chapter 08: Blindfolds on Future Words (Causal Masking)
- **Module 4: Order & Multi-Perspective Processing**
  - Chapter 09: Where in the Sentence Am I? (Positional Encodings & RoPE)
  - Chapter 10: Looking Through Different Glasses (Multi-Head Attention)
- **Module 5: The Full Transformer Block**
  - Chapter 11: The Shortcut Bridge (Residual Connections)
  - Chapter 12: Keeping Everyone Calm (LayerNorm & RMSNorm)
  - Chapter 13: The Thinking Chamber (Feed-Forward Networks)
- **Module 6: How the Model Learns (Training Math)**
  - Chapter 14: How Wrong Was I? (Cross-Entropy Loss & Perplexity)
  - Chapter 15: Walking Down the Mountain (Gradients & Backpropagation)
  - Chapter 16: The Smart Walker (Momentum and the Adam Optimizer)
- **Module 7: Inference & Sampling**
  - Chapter 17: Turning Up the Heat (Temperature, Top-k, Top-p)
- **Module 8: Aligning and Refining (Post-Training Math)**
  - Chapter 18: Teaching Good Manners (RLHF, Reward Modeling, and DPO)
