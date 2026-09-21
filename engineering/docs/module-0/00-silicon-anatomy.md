# Chapter E00: The Silicon Anatomy: HBM, SRAM, Tensor Cores, and Interconnects

## Step 1: Hardware Intuition

Imagine you are a brilliant chef working in a high-speed professional kitchen. You have three distinct places to store ingredients:

1. **Your Cutting Board (Registers):** What is in your hands right now. You can chop, slice, and dice instantly with zero delay. But your cutting board is tiny; it holds only two onions at a time.
2. **The Kitchen Counter (SRAM / Shared Memory):** A table right next to your elbow. Reaching for an ingredient takes two seconds. It can hold a few bowls of pre-chopped ingredients, but it cannot store the whole restaurant's weekly inventory.
3. **The Distant Basement Walk-in Freezer (High Bandwidth Memory - HBM):** A giant refrigerated room down two flights of stairs. It holds 80 gigabytes of ingredients. But every time you need something from the freezer, you must stop cooking, walk down the stairs, unlock the door, haul the crate back up, and unpack it. Even with a wide elevator (high memory bandwidth), that round-trip walk takes an eternity compared to chopping on your cutting board.
4. **The Supplier Delivery Truck (PCIe / NVLink):** When you run out of ingredients entirely and must borrow crates from the restaurant next door across an alleyway.

In modern AI accelerators (such as the NVIDIA H100, AMD MI300X, or Google TPU v5), **the compute units (Tensor Cores) are lightning-fast chefs starved for ingredients**. The fundamental challenge of LLM systems engineering is not making the chef chop faster; it is keeping the chef supplied from the distant basement freezer without ever letting their hands go idle.

---

## Step 2: Silicon Micro-Mechanics

Modern GPUs are massively parallel hierarchical computers. An NVIDIA H100 SXM GPU contains 132 Streaming Multiprocessors (SMs). Understanding execution speed requires examining the memory and compute tiers inside each SM and across the board:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               NVIDIA H100 GPU (132 SMs)                          │
│                                                                                  │
│  ┌─────────────────────────────────┐       ┌─────────────────────────────────┐  │
│  │ Streaming Multiprocessor (SM 0) │  ...  │ Streaming Multiprocessor (SM131)│  │
│  │                                 │       │                                 │  │
│  │  [Registers: 64K x 32-bit]      │       │  [Registers: 64K x 32-bit]      │  │
│  │  Latency: 1 cycle               │       │  Latency: 1 cycle               │  │
│  │  Bandwidth: ~33 TB/s aggregate  │       │  Bandwidth: ~33 TB/s aggregate  │  │
│  │                                 │       │                                 │  │
│  │  [4x 4th-Gen Tensor Cores]      │       │  [4x 4th-Gen Tensor Cores]      │  │
│  │  Peak FP16: ~15 TFLOPS / SM     │       │  Peak FP16: ~15 TFLOPS / SM     │  │
│  │                                 │       │                                 │  │
│  │  [Shared Memory / L1 SRAM]      │       │  [Shared Memory / L1 SRAM]      │  │
│  │  Capacity: 228 KB / SM          │       │  Capacity: 228 KB / SM          │  │
│  │  Latency: ~20-30 cycles         │       │  Latency: ~20-30 cycles         │  │
│  │  Bandwidth: ~17 TB/s aggregate  │       │  Bandwidth: ~17 TB/s aggregate  │  │
│  └────────────────┬────────────────┘       └────────────────┬────────────────┘  │
│                   │                                         │                   │
│                   └────────────────────┬────────────────────┘                   │
│                                        │                                        │
│                         [L2 Cache: 50 MB, ~5 TB/s]                              │
│                                        │                                        │
│                         [HBM3: 80 GB, 3.35 TB/s]                                │
│                         Latency: 200-400 cycles                                 │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                 [NVLink 4: 900 GB/s Bidirectional to Peer GPUs]
                 [PCIe Gen 5: 64 GB/s Bidirectional to Host CPU]
```

### 1. Registers (Zero Latency Workspace)
- Each SM has $65,536$ 32-bit registers ($256\text{ KB}$ per SM, totaling $33.8\text{ MB}$ across 132 SMs).
- Access latency: $1$ clock cycle.
- Operands must reside in registers for Tensor Cores to execute arithmetic operations. If a kernel uses too many variables per thread ("register pressure"), excess variables spill into local memory residing in HBM, causing catastrophic latency penalties ("register spilling").

### 2. Shared Memory / L1 SRAM (Software-Managed Scratchpad)
- Each SM has up to $228\text{ KB}$ of configurable shared memory / L1 data cache.
- Access latency: $\approx 20\text{--}30$ clock cycles.
- Aggregate on-chip SRAM bandwidth across all SMs: over $15\text{--}18\text{ TB/s}$.
- This is where modern fused kernels (e.g., FlashAttention, Fused RMSNorm) store intermediate matrix tiles to avoid round-trips to HBM.

### 3. High Bandwidth Memory (HBM3)
- The main physical device storage: $80\text{ GB}$ (H100) or $141\text{ GB}$ (H200) stacked vertically on silicon interposers alongside the GPU die.
- Bus width: 5120 bits wide (compared to 128-384 bits on consumer GPUs).
- Peak bandwidth: $3.35\text{ TB/s}$.
- Access latency: $\approx 200\text{--}400$ clock cycles. If data is not in cache, the SM stalls for hundreds of cycles waiting for DRAM memory lines to arrive.

### 4. Interconnects: NVLink vs. PCIe
- **PCIe Gen 5:** Connects the GPU to the host CPU motherboard at $64\text{ GB/s}$ bidirectional.
- **NVLink 4:** Connects GPUs directly to each other via NVSwitch at $900\text{ GB/s}$ bidirectional ($14\times$ faster than PCIe Gen 5).
- Why this matters: Distributed Tensor Parallelism requires GPUs to exchange activation tensors at every single layer; running Tensor Parallelism over PCIe stalls the entire model, whereas NVLink sustains high-throughput matrix sharding.

---

## Step 3: Cross-Component Coupling

The physical hardware hierarchy dictates how software components must be designed and coupled:

1. **Attention Tile Sizing vs. SRAM Capacity:**  
   In FlashAttention, the Query, Key, and Value blocks must fit simultaneously inside the $228\text{ KB}$ SRAM of a single SM. If the attention head dimension is $d = 128$, a tile of $64 \times 128$ in FP16 takes $16\text{ KB}$. Query, Key, and Value tiles plus intermediate accumulation buffers occupy $\approx 100\text{ KB}$, fitting comfortably. If $d = 256$, tile sizes must be halved, increasing loop iterations and register pressure.

2. **Kernel Fusion vs. HBM Traffic:**  
   When calculating $\mathbf{y} = \text{RMSNorm}(\mathbf{x} + \text{residual})$, an unfused implementation writes the sum back to HBM, then reads it again for RMSNorm. Because HBM latency is 300 cycles and bandwidth is only $3.35\text{ TB/s}$, unfused operations spend 90% of their wall-clock time waiting on memory bus transfers. Fusing both operations keeps data in registers and L1 SRAM, reducing latency by $3\times$.

3. **Decode Concurrency vs. HBM Bus Saturation:**  
   During autoregressive generation, generating 1 token requires reading the full model parameter weights ($140\text{ GB}$ for a 70B model) from HBM into SRAM once per step. If the batch size is 1, you transfer $140\text{ GB}$ of data to perform only $140\text{ billion}$ FLOPs, achieving less than $1\%$ Tensor Core utilization.

---

## Step 4: The Exact Performance Formula

The time required to execute any GPU kernel is determined by the maximum of its compute time and its memory transfer time:

$$T_{\text{kernel}} = \max\left(T_{\text{compute}}, \; T_{\text{memory}}\right) + T_{\text{latency\_overhead}}$$

Where:

$$T_{\text{compute}} = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Attainable Compute Throughput (FLOPS)}}$$

$$T_{\text{memory}} = \frac{\text{Total Bytes Transferred from HBM}}{\text{Attainable HBM Bandwidth (Bytes/second)}}$$

Latency hiding through warp scheduling: A Streaming Multiprocessor executes threads in groups of 32 called **warps**. When Warp 0 issues a high-latency load from HBM (taking 300 cycles), the hardware warp scheduler instantly switches execution to Warp 1, Warp 2, etc., in zero cycles:

$$\text{Active Warps per SM} = \frac{\text{Total Allocated Threads per SM}}{32}$$

To fully hide the 300-cycle HBM latency, an SM requires enough concurrent active warps (high **occupancy**) to perform independent math instructions while other warps wait for memory requests.

---

## Step 5: Concrete Benchmark Walkthrough

Let us calculate the exact time required to run a single decode step for a **LLaMA-3 70B** model in FP16 ($140\text{ GB}$ weights) running on a single theoretical GPU with $80\text{ GB}$ HBM3 ($3.35\text{ TB/s}$) at batch size $B = 1$. (Assume weights are sharded or quantized to fit):

### Step 5.1: Calculate Compute Time
For a 70B parameter model, generating 1 token requires:

$$\text{FLOPs} = 2 \times 70 \times 10^9 = 1.4 \times 10^{11} \text{ FLOPs} = 140 \text{ GFLOPs}$$

On an H100 with peak FP16 Tensor Core throughput of $2,000\text{ TFLOPS} = 2 \times 10^{15}\text{ FLOPs/s}$:

$$T_{\text{compute}} = \frac{1.4 \times 10^{11}}{2 \times 10^{15}} = 0.00007 \text{ seconds} = 0.07 \text{ ms}$$

### Step 5.2: Calculate Memory Transfer Time
To execute those $140\text{ GFLOPs}$, every single weight matrix parameter must be fetched from HBM into SRAM registers:

$$\text{Bytes} = 70 \times 10^9 \times 2 \text{ bytes (FP16)} = 140 \times 10^9 \text{ bytes} = 140 \text{ GB}$$

At peak HBM3 bandwidth of $3.35\text{ TB/s} = 3,350\text{ GB/s}$:

$$T_{\text{memory}} = \frac{140 \text{ GB}}{3,350 \text{ GB/s}} = 0.0418 \text{ seconds} = 41.8 \text{ ms}$$

### Step 5.3: Calculate Hardware Efficiency
Comparing the two numbers:
- Compute time: $0.07\text{ ms}$
- Memory transfer time: $41.8\text{ ms}$

$$\text{Ratio} = \frac{T_{\text{memory}}}{T_{\text{compute}}} = \frac{41.8}{0.07} \approx 597\times$$

**Result:** The Tensor Cores spend **$41.73\text{ ms}$ out of $41.8\text{ ms}$ ($99.83\%$ of the time) completely idle**, stalled waiting for weights to arrive from HBM. The model achieves less than $0.2\%$ Model FLOPs Utilization (MFU).

---

## Step 6: Core Systems Takeaway

> In Large Language Model systems, FLOPs are virtually free; moving bytes across the memory bus is expensive. Every high-performance LLM engineering breakthrough—FlashAttention, GQA, PagedAttention, and Quantization—is an architectural mechanism designed to prevent bytes from traveling across the HBM bus.
