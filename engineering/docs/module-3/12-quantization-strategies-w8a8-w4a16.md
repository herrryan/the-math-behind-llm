# Chapter E12: Quantization Strategies: Weight-Only vs. Weight-Activation

## Step 1: Hardware Intuition
Imagine packing luggage for a flight.
- **Weight-Only Quantization (W4A16 / W8A16):** You vacuum-seal your heavy winter coats (weights) into tiny bags to fit more into the suitcase. Once you reach the hotel (SRAM), you open the bag and let the coat expand before wearing it. This saves luggage space (HBM bandwidth) during decode.
- **Weight-Activation Quantization (W8A8):** Both your clothes and your shoes are manufactured in lightweight mini sizes, allowing you to walk directly through security without unpacking.

## Step 2: Silicon Micro-Mechanics
- **Weight-Only (AWQ, GPTQ):** Weights are stored in INT4 or INT8 in HBM. During GEMV, weights are streamed across the memory bus at reduced size and dequantized to FP16 in registers before compute. Ideal for **memory-bound decode**.
- **Weight-Activation (SmoothQuant, FP8):** Both weights and activations are quantized to 8 bits, enabling native INT8/FP8 Tensor Core MMA instructions. Ideal for **compute-bound prefill**.
- **Activation Outliers:** 0.1% of activation channels exhibit extreme values ($100\times$ normal magnitude); SmoothQuant mathematically migrates the outlier difficulty from activations to weights.

## Step 3: Cross-Component Coupling
- **Prefill vs. Decode Trade-offs:** W4A16 accelerates batch size 1 decode, but slows down large-batch prefill due to dequantization overhead.
- **Quantization Granularity:** Per-tensor vs. per-channel vs. block-wise (group size 128) scaling factors.

## Step 4: The Exact Performance Formula
$$W_{\text{quant}} = \text{round}\left(\frac{W}{\text{scale}}\right) + \text{zero\_point}$$
$$\text{HBM Bandwidth Speedup} = \frac{\text{FP16 Bytes}}{\text{Quantized Bytes}} = \frac{16}{\text{bits}}$$

## Step 5: Concrete Benchmark Walkthrough
Evaluating a 70B model under INT4 vs FP16 on a single 80GB GPU:
- FP16 model: $140\text{ GB}$ (does not fit on one GPU; requires 2 GPUs).
- INT4 model (AWQ): $35\text{ GB}$ (fits easily on a single 80GB GPU with room for a 40GB KV cache).
- Single-token decode latency drops from $84\text{ ms}$ (distributed over 2 GPUs) to $22\text{ ms}$ on 1 GPU.

## Step 6: Core Systems Takeaway
> Choose your quantization strategy based on your operational regime: Weight-Only quantization accelerates memory-bound decoding; Weight-Activation quantization accelerates compute-bound prefill.
