# The Engineering Behind Large Language Models

Welcome to the **Systems & Engineering Subproject** of *The Math Behind Large Language Models*.

While the core curriculum focuses on the mathematical foundations (linear algebra, probability, loss landscapes, and optimization geometry), this subproject answers the critical systems questions:
1. **How do these mathematical operations physically run on modern silicon (GPUs/TPUs)?**
2. **How do components interact and influence each other across memory hierarchies, bandwidth constraints, and compute schedules?**
3. **What determines real-world performance (TTFT, ITL, Tokens/Sec/GPU, and MFU)?**

---

## Subproject Structure

- **[Master Engineering Curriculum](curriculum.md)**: The complete 27-chapter curriculum spanning hardware architecture, attention engines, linear projection systems, numerical quantization, distributed parallelism, high-throughput serving, and Mixture-of-Experts engineering.
- **Hands-On Systems Labs (`labs/`)**:
  - `01-roofline-profiler/`: Measure arithmetic intensity and plot real kernels against GPU hardware rooflines.
  - `02-flash-attention-kernel/`: Implement tiled attention with online softmax in Triton / C++.
  - `03-paged-kv-cache/`: Build a virtual memory page-table KV cache allocator.
  - `04-tensor-parallel-engine/`: Multi-GPU column and row parallel linear layers with PyTorch collective primitives.

---

## The Core Interaction Triangle

Every LLM implementation is governed by three tightly coupled physical dimensions:

<figure>
<pre>
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
</pre>
<figcaption><strong>Figure E.0:</strong> The Physical Triad of LLM Hardware Systems.</figcaption>
</figure>

- **Compute Bound (Prefill Phase):** Saturated by dense matrix multiplies (GEMM). Arithmetic intensity is high ($I > 100$). The bottleneck is Tensor Core clock cycles.
- **Memory Bandwidth Bound (Decode Phase):** Generating one token at a time requires loading all model weights and KV caches from HBM to SRAM for tiny matrix-vector products ($I \approx 1-5$). The bottleneck is HBM memory bus width.
- **Interconnect Bound (Distributed Scaling):** Splitting layers across GPUs (Tensor Parallelism) or nodes (Expert Parallelism) requires continuous collective communications (`all-reduce`, `all-to-all`). The bottleneck is network latency and wire bandwidth.

---

## Navigating the Curricula

- Mathematical Foundations: [Curriculum Overview](../curriculum.md)
- Systems & Engineering Implementation: [Engineering Curriculum](curriculum.md)
- Course Root Homepage: [Main Index](../index.html)
