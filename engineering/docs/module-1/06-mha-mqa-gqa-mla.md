# Chapter E06: Attention Architectural Variants: MHA vs. MQA vs. GQA vs. MLA

## Step 1: Hardware Intuition
Imagine a team of 32 detectives (Query heads) reading testimony.
- **MHA (Multi-Head Attention):** Every single detective insists on carrying their own separate filing cabinet of witness records (Keys and Values). That is 32 heavy filing cabinets to move every second.
- **MQA (Multi-Query Attention):** All 32 detectives share a single filing cabinet. Moving one cabinet is easy, but detectives occasionally bump into each other and miss subtle clues.
- **GQA (Grouped-Query Attention):** The 32 detectives split into 8 squads of 4. Each squad shares 1 cabinet. You move only 8 cabinets, saving 75% of the effort with virtually zero loss in investigative quality.
- **MLA (Multi-Head Latent Attention):** All records are microfilmed into a compressed code vector, taking up even less space.

## Step 2: Silicon Micro-Mechanics
- **MHA:** {\text{heads}} = n_{\text{kv\_heads}}$. KV cache traffic is  \times$ larger than necessary.
- **MQA (Shazeer 2019):** {\text{kv\_heads}} = 1$. Slashes KV cache memory by \times$.
- **GQA (Ainslie et al. 2023):** {\text{kv\_heads}} = H / G$. In LLaMA-3, 32 Q heads share 8 KV heads ( = 4$).
- **MLA (DeepSeek-V2/V3):** Low-rank compression projecting Keys and Values into a shared latent vector ^{KV} \in \mathbb{R}^{d_c}$.

## Step 3: Cross-Component Coupling
- **Decode Phase Speedup:** During single-token generation, memory bandwidth saturation from loading the KV cache dominates latency. Cutting KV heads by \times$ accelerates decode step latency by up to \times$.
- **Batch Size Scaling:** Smaller KV footprints allow expanding serving batch size without triggering OOM.

## Step 4: The Exact Performance Formula
40062\text{KV Bandwidth per Decode Step} = \frac{2 \times \text{Precision} \times n_{\text{layers}} \times n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot S}{T_{\text{step}}}40062
40062\text{Bandwidth Savings Factor} = \frac{n_{\text{heads}}}{n_{\text{kv\_heads}}}40062

## Step 5: Concrete Benchmark Walkthrough
Comparing LLaMA-2 70B (MHA: 64 KV heads) vs. LLaMA-3 70B (GQA: 8 KV heads):
- LLaMA-2 KV cache: .5	ext{ MB}$ per token.
- LLaMA-3 KV cache: zsh.3125	ext{ MB}$ per token (\times$ reduction).
- At context  = 8192$, batch  = 32$: LLaMA-2 requires 	ext{ GB}$ (impossible on a single 8-GPU node); LLaMA-3 requires .9	ext{ GB}$.

## Step 6: Core Systems Takeaway
> Grouped-Query Attention is an architectural hardware optimization that yields an 8x reduction in KV cache bandwidth during decoding, directly enabling long-context inference and multi-user concurrency.
