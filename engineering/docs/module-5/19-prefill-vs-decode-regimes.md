# Chapter E19: The Two Inference Regimes: Prefill vs. Decode

## Step 1: Hardware Intuition
Imagine a post office sorting mail:
- **Prefill (Prompt Ingestion):** A delivery truck unloads 4,000 letters all at once. The sorting machine runs at full power, batching and scanning hundreds of envelopes per second.
- **Decode (Generation):** A customer stands at the counter asking one question every 30 seconds, forcing the clerk to walk back to the storage vault each time to retrieve one single document.

## Step 2: Silicon Micro-Mechanics
- **Prefill Phase:** High operational intensity ($I > 100$), highly compute-bound, saturates Tensor Cores. Primary metric: **Time to First Token (TTFT)**.
- **Decode Phase:** Low operational intensity ($I \approx 1-5$), severely memory-bandwidth bound. Primary metric: **Inter-Token Latency (ITL)**.
- **Prefill-Decode Interference:** Co-scheduling a heavy prefill job inside an ongoing decode batch causes latency spikes for interactive streaming users.

## Step 3: Cross-Component Coupling
- **Architecture Disaggregation:** Splitwise & Mooncake systems physically separate prefill worker nodes from decode worker nodes to isolate interference.

## Step 4: The Exact Performance Formula
$$\text{TTFT} \approx \frac{2 \times P \times S_{\text{prompt}}}{\text{Attainable FLOPS}}$$
$$\text{ITL} \approx \frac{\text{Model Weights} + \text{KV Cache}}{\text{HBM Bandwidth}}$$

## Step 5: Concrete Benchmark Walkthrough
Comparing prefill vs. decode execution on LLaMA-3 8B ($16\text{ GB}$ FP16 weights) on an H100 ($3.35\text{ TB/s}$, 1000 TFLOPS):
- Prefill (2048 prompt tokens): Compute takes $\approx 0.065\text{ ms}$; bandwidth takes $0.005\text{ ms}$. Compute-bound!
- Decode (1 token, batch 1): Compute takes $0.00003\text{ ms}$; memory bandwidth takes $4.77\text{ ms}$. 99.9% memory-bound!

## Step 6: Core Systems Takeaway
> An LLM engine is two completely different beasts: a compute-bound GEMM cruncher during prefill, and a memory-bound bandwidth streamer during decode. Production serving architectures must optimize both regimes independently.
