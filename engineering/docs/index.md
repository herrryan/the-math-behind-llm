# The Engineering Behind Large Language Models

Welcome to the **Systems & Engineering Subproject** of *The Math Behind Large Language Models*.

While mathematical theory explains *what* transformations an LLM performs, this course explains *how* those transformations physically execute on modern silicon (NVIDIA Hopper/Blackwell, AMD CDNA, Google TPU) and how hardware limits shape the design of real-world AI systems.

---

## Why Systems Engineering Matters

```
                      [COMPUTE]
              Tensor Cores / Systolic Arrays
                  (Peak FLOPS: e.g. 2,000 TFLOPS)
                            ▲
                           / \
                          /   \
                         /     \
                        /       \
                       /         \
                      ▼           ▼
               [MEMORY] ◄───────► [INTERCONNECT]
             HBM / SRAM            NVLink / PCIe / RoCE
      (Bandwidth: 3.35 TB/s)      (Bandwidth: 900 GB/s)
```

In production LLM infrastructure, no algorithm runs in a vacuum. Every design decision couples compute, memory, and networking:

- **The Roofline Bottleneck:** A kernel with low operational intensity spends 95% of its execution time stalled, waiting for bytes to trickle from High Bandwidth Memory (HBM) into on-chip registers.
- **The Attention Memory Wall:** Naive quadratic attention materializes $S \times S$ attention matrices in HBM, generating gigabytes of redundant traffic. Tiling algorithms like **FlashAttention** fuse the entire attention loop inside on-chip SRAM, cutting memory traffic by an order of magnitude.
- **The KV Cache Conundrum:** Autoregressive generation requires caching Keys and Values for every prior token. Without **PagedAttention** and **Grouped-Query Attention (GQA)**, memory fragmentation and bandwidth saturation limit serving concurrency and cause Out-Of-Memory crashes.
- **Distributed Scaling Overheads:** Splitting models across GPUs via **Tensor Parallelism**, **Pipeline Parallelism**, and **ZeRO / FSDP** requires continuous cross-GPU communication. Understanding latency hiding and collective primitives is the difference between linear scaling and communication collapse.

---

## Course Roadmap

The curriculum is structured into 8 comprehensive modules, spanning the hardware substrate to distributed clusters:

1. **[Module 0: Hardware Landscape & Physics of Execution](module-0/00-silicon-anatomy.md)**  
   Silicon hierarchy (Registers, SRAM, HBM), the Roofline Model, memory coalescing, and DRAM burst mechanics.
2. **[Module 1: The Attention Subsystem in Silicon](module-1/03-naive-attention-memory-wall.md)**  
   Naive attention memory traffic, FlashAttention 1/2/3 tiling, KV cache footprint, and MHA vs. GQA vs. MLA.
3. **[Module 2: Linear Projections, Activations & Memory Bandwidth](module-2/07-gemm-tensor-cores.md)**  
   Tensor Core GEMM tiling, SwiGLU 3-matrix footprints, fused RMSNorm kernels, and on-the-fly RoPE rotations.
4. **[Module 3: Numerical Formats, Precision & Quantization](module-3/11-number-formats-fp8-bf16.md)**  
   FP32, FP16, BF16, and FP8 (E4M3/E5M2); weight-only (AWQ/GPTQ) vs. weight-activation (SmoothQuant) quantization.
5. **[Module 4: Distributed Systems & Parallelism](module-4/14-distributed-communication-primitives.md)**  
   All-Reduce, Megatron Tensor Parallelism, 1F1B Pipeline scheduling, ZeRO-1/2/3 & FSDP, and Ring Attention.
6. **[Module 5: High-Throughput Serving & Inference](module-5/19-prefill-vs-decode-regimes.md)**  
   Prefill vs. decode dynamics, continuous batching, speculative decoding, and chunked prefill with prefix caching.
7. **[Module 6: Mixture of Experts (MoE) Engineering](module-6/23-moe-routing-capacity-factor.md)**  
   Top-K routing, expert capacity factors, and All-to-All distributed dispatch/combine bottlenecks.
8. **[Module 7: Auditing, Profiling & Hardware Efficiency](module-7/25-model-memory-budget.md)**  
   Complete model memory budget accounting, Model FLOPs Utilization (MFU), and Nsight profiling.

---

## The 4 Hands-On Systems Labs

Along with the theoretical deep-dives, you will build and profile 4 production-grade systems artifacts:

- **[Lab E1: The Roofline Profiler](labs/01-roofline-profiler.md):** Measure operational intensity and plot real PyTorch kernels against GPU hardware ceilings.
- **[Lab E2: The Minimal FlashAttention Kernel](labs/02-flash-attention-kernel.md):** Build a tiled attention kernel with online softmax in Triton / C++.
- **[Lab E3: The Paged KV Cache Engine](labs/03-paged-kv-cache.md):** Implement virtual memory page-table KV cache management from scratch.
- **[Lab E4: 2-GPU Tensor Parallel Engine](labs/04-tensor-parallel-engine.md):** Construct distributed column-parallel and row-parallel linear layers with PyTorch collective primitives.

---

Ready to begin? Start with **[Chapter E00: The Silicon Anatomy](module-0/00-silicon-anatomy.md)**!
