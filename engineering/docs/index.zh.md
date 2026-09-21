# 大语言模型背后的工程实现：底层系统与硬件物理

欢迎来到《大语言模型背后的数学原理》的**系统与工程实现子项目**。

如果说数学理论揭示了 LLM 到底在执行*什么样*的代数变换与概率投影，那么本课程则深入物理现实，全面解析这些数学算子如何在现代硅基芯片（NVIDIA Hopper/Blackwell、AMD CDNA、Google TPU）上真实执行，以及物理硬件极限如何塑造了当今前沿 AI 系统的架构设计。

---

## 为什么系统工程至关重要？

```
                      [算力 COMPUTE]
                Tensor Core 脉动阵列 / 矩阵单元
                  (峰值浮点算力: 如 2,000 TFLOPS)
                            ▲
                           / \
                          /   \
                         /     \
                        /       \
                       /         \
                      ▼           ▼
               [显存 MEMORY] ◄──► [互联 INTERCONNECT]
                 HBM / SRAM         NVLink / PCIe / RoCE
          (带宽: 3.35 TB/s)        (带宽: 900 GB/s)
```

在真实的生产级 LLM 基础设施中，没有任何算法能脱离物理硬件运行。算力、显存带宽与互联网络三者紧密交织：

- **屋顶模型瓶颈（Roofline Bottleneck）：** 一个算术强度过低的算子，即使配备了全球最先进的 Tensor Core，95% 以上的时钟周期也会被迫处于停滞挂起状态，苦苦等待字节从远端高带宽显存（HBM）运入片上寄存器。
- **注意力机制显存墙（Attention Memory Wall）：** 朴素的二次方复杂度注意力机制会在 HBM 中显式分配并物化 $S \times S$ 的完整分数矩阵，产生数十吉字节（GB）的冗余读写开销。以 **FlashAttention** 为代表的分块平铺（Tiling）算法，将整个计算与在线 Softmax 循环完全熔合在片上 SRAM 中，将访存流量削减了一个数量级。
- **KV Cache 显存碎片困境：** 自回归生成要求为上下文中的每一个历史 Token 缓存键（Key）和值（Value）投影向量。如果没有 **PagedAttention** 虚拟内存分页管理与 **分组查询注意力（GQA）** 架构，巨大的显存碎片化与显存带宽饱和将直接扼杀系统并发，频繁引发 OOM 崩溃。
- **分布式扩展损耗：** 无论是跨卡张量并行（Tensor Parallelism）、流水线并行（Pipeline Parallelism）还是 ZeRO / FSDP 数据并行，模型切分都会引入持续的跨卡通信。深刻理解通信隐藏机制与集合通信原语，是决定集群能达到接近线性的扩展效率，还是陷入通信瘫痪的关键。

---

## 课程体系大纲

本课程分为 8 大核心模块，完整覆盖从单芯片微架构到大规模分布式集群的全栈体系：

1. **[模块 0：硬件图景与物理执行](module-0/00-silicon-anatomy.zh.md)**  
   深入硅基存储层级（寄存器、SRAM、HBM）、屋顶模型（Roofline Model）、合并访存机制与 DRAM 突发传输物理原理。
2. **[模块 1：芯片层级注意力子系统](module-1/03-naive-attention-memory-wall.zh.md)**  
   朴素注意力的 HBM 显存墙瓶颈、FlashAttention 1/2/3 架构演进、KV Cache 内存解剖与 MHA / GQA / MLA 架构变体权衡。
3. **[模块 2：线性投影、激活函数与显存带宽](module-2/07-gemm-tensor-cores.zh.md)**  
   Tensor Core 矩阵乘法分块对齐、SwiGLU 激活函数的 3 矩阵显存开销、RMSNorm 残差熔合算子与 RoPE 旋转位置编码寄存器级即时计算。
4. **[模块 3：数值格式、计算精度与量化技术](module-3/11-number-formats-fp8-bf16.zh.md)**  
   FP32、FP16、BF16 与 FP8（E4M3/E5M2）数值分布解析；仅权重量化（AWQ/GPTQ）与权重-激活量化（SmoothQuant）在不同阶段的硬件表现；KV Cache 压缩至 FP8/INT4。
5. **[模块 4：分布式系统与并行扩展](module-4/14-distributed-communication-primitives.zh.md)**  
   All-Reduce 环状与树状通信；Megatron-LM 张量并行（TP）；1F1B 流水线并行（PP）气泡消除；ZeRO-1/2/3 与 FSDP 显存分片；以及百万长上下文的环状注意力（Ring Attention）。
6. **[模块 5：高吞吐推理引擎与在线服务](module-5/19-prefill-vs-decode-regimes.md)**  
   Prefill 首字延迟（TTFT，计算密集型）与 Decode 逐字生成（ITL，访存密集型）两大工况解耦；动态连续批处理（Continuous Batching）；推测解码（Speculative Decoding）；分块预填与前缀缓存。
7. **[模块 6：混合专家模型（MoE）工程架构](module-6/23-moe-routing-capacity-factor.zh.md)**  
   Top-K 门控路由；专家容量因子（Capacity Factor）；分布式专家并行中的 All-to-All 跨机交换机通信瓶颈与计算通信重叠优化。
8. **[模块 7：显存审计、性能分析与硬件利用率](module-7/25-model-memory-budget.zh.md)**  
   训练 16P 显存完整预算公式推导；激活值重计算权衡；模型浮点利用率（MFU）与硬件利用率（HFU）精准度量；Nsight 工具链性能剖析实战。

---

## 4 大动手系统实战实验室

配合深入的系统理论剖析，课程配套 4 个工业级核心系统原型开发：

- **[实验 E1：GPU 屋顶模型性能分析器](labs/01-roofline-profiler.zh.md)：** 编写轻量级分析工具，测量任意 PyTorch 算子的算术强度，并绘制其在 H100/A100 硬件天花板下的真实运行轨迹。
- **[实验 E2：极简 FlashAttention 算子实现](labs/02-flash-attention-kernel.zh.md)：** 使用 Triton / C++ 编写基于片上 SRAM 分块与在线 Softmax 的注意力算子，彻底消除中间状态在 HBM 的显存物化。
- **[实验 E3：PagedAttention 分页 KV Cache 引擎](labs/03-paged-kv-cache.zh.md)：** 从零手写基于虚拟内存页表思想的显存管理引擎，实现零显存碎片化与动态写时复制（CoW）。
- **[实验 E4：双卡张量并行（Tensor Parallelism）引擎](labs/04-tensor-parallel-engine.zh.md)：** 纯手工使用 PyTorch 集合通信原语构建 Megatron 风格的列并行与行并行线性层，完成两卡协同前向传播并验证数值绝对对齐。

---

准备好开启系统底层的探索了吗？从 **[第 E00 章：芯片物理拓扑架构](module-0/00-silicon-anatomy.zh.md)** 正式启程！
