# The Math Behind Large Language Models

An intuitive, rigorous, and comprehensive guide to understanding the mathematical foundations, engineering implementations, and core mechanisms of Large Language Models (LLMs).

---

## 1. Project Ecosystem: The Four Pillars

This repository provides a complete four-pillar learning ecosystem for mastering modern large language models, spanning from first-principles mathematics to production-grade GPU engineering:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE LLM KNOWLEDGE MATRIX                        │
├──────────────────────────────┬──────────────────────────────────────────────┤
│ 1. Core Mathematical Guide   │ 2. Engineering Implementation                │
│    Root Directory            │    engineering/                              │
│    - 5-step pedagogical ladder│    - PyTorch & Triton code implementations   │
│    - Rigorous formulas & toy │    - GPU memory math (VRAM / HBM vs SRAM)    │
│      arithmetic walkthroughs │    - Distributed parallelism & vLLM serving  │
│    - Fully bilingual (EN/ZH) │    - Interactive MkDocs Material portal      │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 3. Zero-Math Intuitive Guide │ 4. Literature & Obsolete Methods Guide       │
│    concepts/                 │    reading-guide/                            │
│    - Strictly zero formulas  │    - 7 timeless must-read foundational papers│
│    - Tangible physical models│    - 7 obsolete methods to avoid / stop      │
│    - Friendly teacher voice  │    - Architectural evolution & reasoning     │
│    - Interactive MkDocs site │    - Interactive MkDocs Material portal      │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

1. **[Core Mathematical Guide (Root)](index.html)**:
   The foundational curriculum answering two questions for every component: *What physical intuition makes this obvious to a 3-year-old child?* and *Where does the exact mathematical formula come from?*
2. **[Engineering Implementation (`engineering/`)](engineering/)**:
   Translates mathematical equations into production PyTorch and CUDA/Triton implementations, detailing GPU memory math, KV Cache bottlenecks, FlashAttention tiling, distributed sharding (TP, PP, ZeRO), and serving optimizations.
3. **[Zero-Math Intuitive Guide (`concepts/`)](concepts/)**:
   A dedicated course teaching all core transformer mechanics strictly without mathematical formulas or Greek letters, relying purely on tangible physical models, mechanical diagrams, and conversational teacher explanations.
4. **[Literature & Obsolete Methods Guide (`reading-guide/`)](reading-guide/)**:
   A high-signal roadmap categorizing essential research papers versus obsolete paradigms (Word2Vec, LSTM, BERT, Performer linear attention, 4-model PPO) and explaining the first-principles reasons why older methods were superseded.

---

## 2. Pedagogical Framework

Every core chapter follows the mandatory **5-Step Learning Ladder**:

1. **3-Year-Old Intuition**: Concrete, physical metaphors (mystery boxes, playground maps, spotlight beams, Play-Doh presses, voting booths, clock dials, volume knobs) with zero jargon.
2. **The Bridging Question**: Defining the exact computational transition: *"How do we convert this physical game into numbers that a computer can store and compute?"*
3. **The Exact Math & Formula**: Rigorous mathematical formulations used in modern models (Transformers, Attention, RoPE, RMSNorm, AdamW, DPO), explicitly defining every variable, dimension, and operation.
4. **Where Did It Come From?**: Historical origins, design rationale, and why simpler functional forms failed.
5. **Concrete Toy Example**: Step-by-step arithmetic with tiny numbers (dimension 2 or 3) so readers can verify every calculation by hand.
6. **Core Takeaway**: A concise punchline summarizing the mathematical role of the component.

---

## 3. Architecture & Technical Design

- **Static Pre-Rendering via `build.py`**:
  All root content is authored in standard Markdown (`index.md` and `index.zh.md`) and pre-rendered into self-contained HTML files (`index.html` and `index.zh.html`). Zero client-side parsing delay, zero CORS issues, and 100% offline-compatible.
- **Compact High-Density Typography**:
  Clean, academic styling with responsive typography, visual probability bars (`<meter>`, `<progress>`), tensor architecture diagrams, definition lists, and collapsible deep-dives (`<details>`).
- **KaTeX Display Isolation**:
  Math blocks are protected and compiled with strict delimiter isolation, avoiding markdown parser collisions.
- **Automated Quality Gates**:
  `build.py` enforces five automated assertions on every build: zero Unicode emojis, zero escaped HTML leaks, zero unrestored tokens, zero math tag collisions, and a full site graph link audit.

---

## 4. Quickstart & Local Preview

### Option A: Browse the Core Mathematical Course
Run a simple HTTP server from the project root:
```bash
python3 -m http.server 8000
```
Then visit `http://localhost:8000` or open any `index.html` / `index.zh.html` directly in your browser.

To rebuild the static HTML site after editing Markdown sources:
```bash
python3 build.py
```

### Option B: Browse the Engineering Implementation Course
```bash
uv run --with "mkdocs<2.0" --with mkdocs-material mkdocs serve -f engineering/mkdocs.yml
```
Then visit `http://127.0.0.1:8000` to access the interactive engineering portal.

### Option C: Browse the Zero-Math Concepts Course
```bash
uv run --with "mkdocs<2.0" --with mkdocs-material mkdocs serve -f concepts/mkdocs.yml
```
Then visit `http://127.0.0.1:8000` to access the intuitive concepts portal.

### Option D: Browse the Literature & Reading Roadmap
```bash
uv run --with "mkdocs<2.0" --with mkdocs-material mkdocs serve -f reading-guide/mkdocs.yml
```
Then visit `http://127.0.0.1:8000` to access the research literature guide.

---

## 5. Master Curriculum Matrix

The core curriculum is organized into 9 modules (Chapters 00 through 19 + Hands-on Lab):

| Module | Chapter Title | English Version | Chinese Version |
| :--- | :--- | :--- | :--- |
| **Module 0: The Big Picture** | Chapter 00: The Next-Word Guessing Game | [English](00-next-word-prediction/index.html) | [中文版](00-next-word-prediction/index.zh.html) |
| **Module 1: Vector Geometry & Embeddings** | Chapter 01: The Word Map (Vectors & Embeddings) | [English](01-vectors-and-spaces/index.html) | [中文版](01-vectors-and-spaces/index.zh.html) |
| | Chapter 02: Measuring Closeness (Dot Product & Cosine Similarity) | [English](02-dot-product-and-similarity/index.html) | [中文版](02-dot-product-and-similarity/index.zh.html) |
| **Module 2: Transforming Spaces & Activations** | Chapter 03: The Magic Stretching Box (Matrix Multiplication) | [English](03-matrix-multiplication/index.html) | [中文版](03-matrix-multiplication/index.zh.html) |
| | Chapter 04: The One-Way Gate (Activation Functions: ReLU, GELU, SwiGLU) | [English](04-activation-functions/index.html) | [中文版](04-activation-functions/index.zh.html) |
| | Hands-on Lab 01: Training Your First Brain in 80 Lines of Pure Python | [English](04b-lab-micro-brain/index.html) | [中文版](04b-lab-micro-brain/index.zh.html) |
| **Module 3: The Attention Mechanism** | Chapter 05: The Transformer Blueprint (Bird's-Eye View & Round Table) | [English](05-transformer-architecture/index.html) | [中文版](05-transformer-architecture/index.zh.html) |
| | Chapter 06: The Library Clue Hunt (Queries, Keys, and Values) | [English](06-queries-keys-values/index.html) | [中文版](06-queries-keys-values/index.zh.html) |
| | Chapter 07: The Fair Voting Booth (The Softmax Function) | [English](07-softmax-function/index.html) | [中文版](07-softmax-function/index.zh.html) |
| | Chapter 08: The Attention Formula & Why We Divide by sqrt(d_k) | [English](08-attention-formula/index.html) | [中文版](08-attention-formula/index.zh.html) |
| | Chapter 09: Blindfolds on Future Words (Causal Masking) | [English](09-causal-masking/index.html) | [中文版](09-causal-masking/index.zh.html) |
| **Module 4: Positional Encodings & Multi-Head** | Chapter 10: Where in the Sentence Am I? (Positional Encodings & RoPE) | [English](10-positional-encodings/index.html) | [中文版](10-positional-encodings/index.zh.html) |
| | Chapter 11: Looking Through Different Glasses (Multi-Head Attention) | [English](11-multi-head-attention/index.html) | [中文版](11-multi-head-attention/index.zh.html) |
| **Module 5: Residuals, Normalization & FFN** | Chapter 12: The Shortcut Bridge (Residual Connections) | [English](12-residual-connections/index.html) | [中文版](12-residual-connections/index.zh.html) |
| | Chapter 13: Keeping Everyone Calm (LayerNorm & RMSNorm) | [English](13-layer-norm-and-rmsnorm/index.html) | [中文版](13-layer-norm-and-rmsnorm/index.zh.html) |
| | Chapter 14: The Thinking Chamber (Feed-Forward Networks) | [English](14-feed-forward-networks/index.html) | [中文版](14-feed-forward-networks/index.zh.html) |
| **Module 6: Training, Loss & Optimization** | Chapter 15: How Wrong Was I? (Cross-Entropy Loss & Perplexity) | [English](15-cross-entropy-loss/index.html) | [中文版](15-cross-entropy-loss/index.zh.html) |
| | Chapter 16: Walking Down the Mountain (Gradients & Backpropagation) | [English](16-gradients-and-backpropagation/index.html) | [中文版](16-gradients-and-backpropagation/index.zh.html) |
| | Chapter 17: The Smart Walker (Momentum & The AdamW Optimizer) | [English](17-adam-optimizer/index.html) | [中文版](17-adam-optimizer/index.zh.html) |
| **Module 7: Inference & Sampling** | Chapter 18: Turning Up the Heat (Temperature, Top-k, & Top-p Sampling) | [English](18-sampling-and-temperature/index.html) | [中文版](18-sampling-and-temperature/index.zh.html) |
| **Module 8: Alignment & Post-Training** | Chapter 19: Teaching Good Manners (RLHF, Reward Modeling, and DPO) | [English](19-rlhf-and-dpo/index.html) | [中文版](19-rlhf-and-dpo/index.zh.html) |

---

## 6. Engineering Guardrails

- **Zero Unicode Emojis**: The repository enforces zero emojis across all markdown, html, yaml, and python files to maintain an elegant, distraction-free academic presentation.
- **Git Commit Protocol**: Every meaningful change is staged, verified, and committed with granular, reversible commit messages.
- **Offline Self-Contained Deliverables**: Zero external CDN runtime dependencies required for core page structure.
