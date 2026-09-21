# Chapter E02: Memory Layouts, Strides, and Memory Coalescing

## Step 1: Hardware Intuition

Imagine you are a postal delivery worker delivering letters to a row of 32 houses numbered 1 through 32 on Maple Street:

- **The Coalesced Scenario:** Every house from 1 to 32 has a letter today. You drive down the street once, open your delivery bag, hand 1 letter to each house in a single continuous sweep, and finish in 10 seconds.
- **The Strided / Uncoalesced Scenario:** Every house has a letter, but the addresses were printed in strides of 32 (House 1, then House 33, then House 65, across different neighborhoods). For every single letter, you must drive to a completely different part of town, park your truck, deliver one envelope, get back in, and drive to the next neighborhood. Delivering 32 letters now takes 30 minutes!

In GPU silicon, High Bandwidth Memory (HBM) does not fetch individual 2-byte numbers. Whenever a thread asks for 2 bytes, **the physical DRAM bus transfers a wide 32-byte or 128-byte chunk (a cache line)**.

If all 32 threads in a warp access consecutive, adjacent memory addresses, their 32 individual requests **coalesce** into one or two clean 128-byte cache-line transfers ($100\%$ bus efficiency). If threads access scattered addresses with large strides, the GPU must execute up to 32 separate 128-byte transfers to fetch only 64 bytes of real data ($1.5\%$ bus efficiency!).

---

## Step 2: Silicon Micro-Mechanics

A GPU executes threads in lockstep groups of 32 called a **warp**. When threads in a warp execute a load instruction (`LDG.E`), the hardware Memory Management Unit (MMU) inspects the 32 requested target memory addresses:

```
COALESCED MEMORY ACCESS (Consecutive Addresses):
Warp Threads:  T0   T1   T2   T3   ...   T30  T31
               │    │    │    │           │    │
Addresses:    [00] [02] [04] [06]  ...  [60] [62]  (in bytes)
               └────┴────┴────┴───────────┴────┘
               ▼
[Single 128-byte HBM Memory Transaction]  -->  100% Efficiency (0 wasted bytes)

UNCOALESCED / STRIDED ACCESS (Stride = 128 bytes):
Warp Threads:  T0           T1           T2          ...   T31
               │            │            │                 │
Addresses:    [0000]       [0128]       [0256]       ...  [3968]
               ▼            ▼            ▼                 ▼
             [128B]       [128B]       [128B]             [128B]
Total: 32 separate 128-byte transactions = 4096 bytes transferred from HBM!
Useful Data: 32 x 2 bytes = 64 bytes.
Efficiency: 64 / 4096 = 1.56% (98.4% of HBM bandwidth is thrown away!)
```

### Tensor Strides in Multidimensional Arrays
In PyTorch and C++, a tensor is a contiguous 1D block of physical memory paired with a `shape` and a `stride` tuple:
$$\text{Memory Offset}(\mathbf{x}_{i, j, k}) = \text{base\_ptr} + i \times \text{stride}_0 + j \times \text{stride}_1 + k \times \text{stride}_2$$

For a 2D tensor of shape $(M, N)$ stored in **Row-Major format (C-contiguous)**:
- `shape` = `(M, N)`
- `stride` = `(N, 1)`
- Accessing along row dimension $j$ has $\text{stride}_1 = 1$: adjacent threads access adjacent bytes (perfect coalescing).
- Accessing along column dimension $i$ has $\text{stride}_0 = N$: adjacent threads access bytes spaced $N$ elements apart. If $N$ is large, access becomes completely uncoalesced!

---

## Step 3: Cross-Component Coupling

Memory layouts dictate the performance of key LLM components:

1. **The Transpose Penalty in Attention ($Q K^\top$):**  
   In textbook multi-head attention, computing the dot product between Query $\mathbf{Q} \in \mathbb{R}^{B \times H \times S \times d}$ and Key $\mathbf{K} \in \mathbb{R}^{B \times H \times S \times d}$ requires transposing $\mathbf{K}$ into shape $(B, H, d, S)$.  
   Calling `K.transpose(-1, -2)` does not move bytes in memory; it merely swaps the tensor strides from `(d, 1)` to `(1, d)`.  
   When a standard GEMM reads transposed $\mathbf{K}$, threads read across column strides of size $d$. If $d = 128$ (256 bytes in FP16), every single thread load hits a different cache line, wasting up to $75\%$ of memory bandwidth unless an explicit SRAM transpose buffer is used.

2. **KV Cache Layouts for Decoding:**  
   During autoregressive generation, the KV cache can be arranged in two primary dimension layouts:
   - Layout A: `[batch, num_heads, max_seq_len, head_dim]`
   - Layout B: `[max_seq_len, batch, num_heads, head_dim]`
   In Layout A, as sequence length advances, reading past tokens for a single request reads contiguous rows along `head_dim` ($stride = 1$), yielding perfect memory coalescing.

3. **The Hidden Cost of `.contiguous()`:**  
   When PyTorch operations require a contiguous tensor buffer, calling `.contiguous()` triggers an explicit memory copy kernel that reads the strided data and writes a newly packed buffer in HBM. For large activation tensors in 70B models, calling `.contiguous()` after tensor sharding or slicing consumes gigabytes of HBM bandwidth and introduces measurable latency stalls.

---

## Step 4: The Exact Performance Formula

Memory Coalescing Efficiency $\eta_{\text{coalesce}}$ is defined as:

$$\eta_{\text{coalesce}} = \frac{\text{Requested Useful Bytes}}{\text{Number of Cache Lines Fetched} \times L_{\text{line}}}$$

Where $L_{\text{line}}$ is the hardware memory transaction size ($32\text{ bytes}$ for L1 / Shared Memory, $128\text{ bytes}$ for L2 / HBM DRAM bursts).

If 32 threads in a warp access FP16 data (2 bytes per thread), the total requested useful data is:

$$\text{Useful Bytes} = 32 \times 2 = 64 \text{ bytes}$$

- If access is perfectly coalesced (all 32 addresses fall inside a single $128\text{-byte}$ aligned boundary):
  $$\eta_{\text{coalesce}} = \frac{64}{1 \times 128} = 50\% \quad (\text{or } 100\% \text{ if using 32-byte sectors})$$
- If access has stride $S \ge 64\text{ elements}$ ($128\text{ bytes}$), each thread touches a distinct $128\text{-byte}$ line:
  $$\eta_{\text{coalesce}} = \frac{64}{32 \times 128} = \frac{64}{4096} \approx 1.56\%$$

The Effective Memory Bandwidth $\text{BW}_{\text{eff}}$ experienced by the kernel is:

$$\text{BW}_{\text{eff}} = \eta_{\text{coalesce}} \times \text{BW}_{\text{peak}}$$

On an H100 with $3.35\text{ TB/s}$ peak bandwidth, an uncoalesced kernel with $\eta = 1.56\%$ experiences an effective memory bandwidth of only:

$$\text{BW}_{\text{eff}} = 0.0156 \times 3350 \text{ GB/s} \approx 52.3 \text{ GB/s}$$

**The memory system is throttled down to the speed of a standard DDR5 laptop bus!**

---

## Step 5: Concrete Benchmark Walkthrough

Let us trace what happens when 32 threads in a warp read a column from a Key tensor stored in row-major order:
- Tensor dimensions: $S = 4096$ tokens, $d = 128$ dimensions
- Datatype: FP16 (2 bytes per element)
- Memory buffer: Row-major, so $\text{stride}_0 = d \times 2 = 256\text{ bytes}$, $\text{stride}_1 = 2\text{ bytes}$.

### Scenario A: Reading a Row (Consecutive Dimensions)
- Thread $k$ reads element $(i, k)$ for $k \in [0, 31]$:
$$\text{Address}(k) = \text{base} + i \times 256 + k \times 2$$
The 32 threads request addresses spanning from offset $0$ to offset $62$.
- All requested addresses fit inside **one single $128\text{-byte}$ DRAM transaction line** (from offset 0 to 127).
- Total DRAM data fetched: $128\text{ bytes}$.
- Useful data delivered: $64\text{ bytes}$.
- **Bus Efficiency: $50\%$** (or $100\%$ on architectures with 32-byte sub-sectoring).

### Scenario B: Reading a Column (Consecutive Sequence Tokens)
- Thread $k$ reads element $(k, j)$ for $k \in [0, 31]$ at fixed dimension $j$:
$$\text{Address}(k) = \text{base} + k \times 256 + j \times 2$$
- Thread 0 requests offset $0$.
- Thread 1 requests offset $256$.
- Thread 2 requests offset $512$.
- ...
- Thread 31 requests offset $7936$.

Notice that each consecutive thread address is spaced $256\text{ bytes}$ apart. Because the cache line size is $128\text{ bytes}$, **no two threads share a cache line!**
- The memory controller must issue **32 separate $128\text{-byte}$ transactions**.
- Total DRAM data fetched: $32 \times 128 = 4096\text{ bytes}$.
- Useful data delivered: $32 \times 2 = 64\text{ bytes}$.
- **Bus Efficiency: $\frac{64}{4096} = 1.56\%$**.
- **Performance Impact:** Scenario B runs **$32\times$ slower** than Scenario A, solely due to memory stride layout.

---

## Step 6: Core Systems Takeaway

> Tensor dimensions are not abstract coordinates; they are physical memory pointers. In high-performance LLM engineering, accessing data with non-unit strides causes up to 98% of HBM bandwidth to be discarded on unused bytes. Always ensure the fastest-changing thread index maps to the contiguous ($stride = 1$) dimension of physical memory.
