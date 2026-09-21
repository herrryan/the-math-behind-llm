# Chapter E14: Distributed Communication Primitives: All-Reduce, All-Gather, and Reduce-Scatter

## Step 1: Hardware Intuition
Imagine 8 accountants spread across 8 tables. Each accountant has calculated a partial total.
- **All-Reduce:** The accountants quickly pass summaries around the circle until every accountant has the grand global total.
- **All-Gather:** Each accountant has a unique chapter of a book; they make copies and share so everyone holds the full book.
- **Reduce-Scatter:** They combine partial results, but each accountant keeps only their assigned final chapter.
The time spent talking and passing papers across the room is pure overhead where no accounting math gets done.

## Step 2: Silicon Micro-Mechanics
- **Collective Algorithms:**
  - Ring All-Reduce: Transmits $2 \frac{N-1}{N} S$ bytes per GPU in $2(N-1)$ steps.
  - Tree All-Reduce: Optimizes for low latency at small message sizes.
- **Network Bandwidth:** Intra-node NVLink (900 GB/s) vs. Inter-node InfiniBand/RoCE (400-800 Gbps $\approx 50-100\text{ GB/s}$).
- **CUDA Streams & Overlap:** Executing communication on dedicated communication streams while Tensor Cores execute matrix math on default compute streams.

## Step 3: Cross-Component Coupling
- **Parallelism Strategy Selection:** Fast NVLink enables high-bandwidth Tensor Parallelism (TP); slow inter-node networks force engineers to use Pipeline (PP) or Data Parallelism (DP).

## Step 4: The Exact Performance Formula
$$T_{\text{Ring-AllReduce}} = 2(N-1)\alpha + 2\left(\frac{N-1}{N}\right)\frac{S}{\text{Bus Bandwidth}}$$
where $\alpha$ is network latency, $N$ is number of GPUs, and $S$ is tensor size in bytes.

## Step 5: Concrete Benchmark Walkthrough
Benchmarking an All-Reduce of a 70B activation tensor ($S = 4096, d = 8192, \text{FP16} = 67.1\text{ MB}$) across 8 GPUs:
- Over NVLink 4 ($900\text{ GB/s}$): $T \approx 0.13\text{ ms}$.
- Over 100 Gbps Ethernet ($12.5\text{ GB/s}$): $T \approx 9.4\text{ ms}$ ($72\times$ slower!).

## Step 6: Core Systems Takeaway
> Collective communication overhead dictates parallel scaling efficiency. Match the communication volume of your parallelism strategy to the physical bandwidth of the underlying network interconnect.
