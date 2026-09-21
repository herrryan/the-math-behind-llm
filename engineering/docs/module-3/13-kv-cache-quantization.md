# Chapter E13: KV Cache Quantization: Compressing Context to FP8 and INT4

## Step 1: Hardware Intuition
If your office file cabinet is overflowing with historical documents, you have two choices: buy another expensive room (more GPUs), or scan every document into compressed black-and-white microfilms (quantization). As long as the microfilm text remains legible, your office stays organized and you can store 4 times as many client case files.

## Step 2: Silicon Micro-Mechanics
- **Storage Layout:** Compressing cached Key and Value vectors from 16 bits to 8 bits (FP8 E4M3 or INT8) or 4 bits.
- **Per-Channel & Per-Token Scaling:** Keys and Values have different numerical distributions. Key vectors benefit from per-head dynamic scaling; Value vectors benefit from per-token scaling.
- **Dequantization in FlashAttention:** Dequantizing INT8/FP8 KV blocks directly inside on-chip SRAM before computing attention tiles.

## Step 3: Cross-Component Coupling
- **Serving Concurrency Explosion:** Cutting KV cache size by $50\%$ (FP8) or $75\%$ (INT4) immediately doubles or quadruples the maximum concurrent batch size in serving engines.
- **Needle-in-a-Haystack Accuracy:** Poorly calibrated KV quantization degrades long-range retrieval accuracy.

## Step 4: The Exact Performance Formula
$$\text{KV Memory Savings} = \left(1 - \frac{\text{Quantized Bits}}{16}\right) \times 100\%$$
For FP8: 50% memory reduction; for INT4: 75% memory reduction.

## Step 5: Concrete Benchmark Walkthrough
Serving 128 concurrent users at 8K context on LLaMA-3 70B:
- FP16 KV cache: $128 \times 8192 \times 0.3125\text{ MB} = 327.6\text{ GB}$ (requires at least 5x 80GB GPUs just for cache).
- FP8 KV cache: $163.8\text{ GB}$ ($2\times$ fewer GPUs required).
- INT4 KV cache: $81.9\text{ GB}$ ($4\times$ fewer GPUs required).

## Step 6: Core Systems Takeaway
> Quantizing the KV cache is the most cost-effective architectural lever for unlocking massive concurrent serving capacity without degrading reasoning accuracy.
