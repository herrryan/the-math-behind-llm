# 大语言模型背后的工程实现：系统级大师课程体系

<nav aria-label="Table of Contents">
  <p>
    <strong>课程导航：</strong>
    <a href="#thesis">系统论点</a> &bull;
    <a href="#pedagogy">六步工程教学法</a> &bull;
    <a href="#interaction-graph">组件依赖与硬件瓶颈图</a> &bull;
    <a href="#evolution-chain">动手系统实验室</a> &bull;
    <a href="#module-0">模块 0</a> &bull;
    <a href="#module-1">模块 1</a> &bull;
    <a href="#module-2">模块 2</a> &bull;
    <a href="#module-3">模块 3</a> &bull;
    <a href="#module-4">模块 4</a> &bull;
    <a href="#module-5">模块 5</a> &bull;
    <a href="#module-6">模块 6</a> &bull;
    <a href="#module-7">模块 7</a>
  </p>
</nav>

<hr>

<h2 id="thesis">1. 系统核心论点：LLM 执行的物理定律</h2>

在纯数学理论中，大语言模型是一系列在离散概率空间中进行的优美矩阵变换与非线性投影。但在硅基芯片中，大语言模型本质上是一个**受物理硬件极限严格约束的访存密集型数据流调度系统**。这些物理瓶颈包括：片上 SRAM 容量极限、高带宽显存（HBM）总线物理位宽、Tensor Core 浮点算力峰值，以及多卡加速器之间的高速互联带宽（NVLink、PCIe、InfiniBand）。

每一次数学运算都对应着明确的硬件成本：

- **显存带宽 vs. 浮点算力：** 一个硬件即使算力达到巅峰，如果字节无法足够快地从 HBM 传输至片上寄存器，其实际计算单元的利用率仍可能低于 5%。
- **跨组件相互影响：** 任何微小的模型架构设计变动（例如将多头注意力 MHA 改为分组查询注意力 GQA，或者由普通 MLP 改为 SwiGLU）都会在全系统引发链式反应：它改变了反向传播中间激活值的显存占用量，决定了片上 SRAM 算子分块尺寸，直接增减了分布式张量并行的通信体积，并根本性地改变了自回归生成阶段是处于访存受限还是计算受限状态。
- **系统核心指标：** 一个优秀的工程实现，其成果最终体现为具体的物理度量：**首字延迟（TTFT）**、**逐字生成延迟（ITL）**、**单卡每秒处理 Token 数（Tokens/s/GPU）**、**模型浮点利用率（MFU）** 以及 **总体拥有成本（TCO）**。

---

<h2 id="pedagogy">2. 六步系统工程教学法</h2>

本课程的每一章都严格遵循环环相扣的六步进阶学习阶梯：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="center" width="8%">步骤</th>
      <th align="left" width="22%">板块名称</th>
      <th align="left" width="35%">学习目标</th>
      <th align="left" width="35%">具体物理类比</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>步骤 1</strong></td>
      <td><strong>物理直觉</strong></td>
      <td>脱离软件抽象，感受底层真实的物理机械限制</td>
      <td>水桶传递队、工厂流水传送带、厨师案板与远端地下冷冻库</td>
    </tr>
    <tr>
      <td align="center"><strong>步骤 2</strong></td>
      <td><strong>芯片微观机制</strong></td>
      <td>解析算子在硬件部件上的物理执行细节（寄存器、SRAM、HBM、ALU）</td>
      <td>显存跨步与合并访存、Tensor Core 矩阵乘分块对齐、Warp 线程束调度</td>
    </tr>
    <tr>
      <td align="center"><strong>步骤 3</strong></td>
      <td><strong>跨组件相互耦合</strong></td>
      <td>解析该组件如何影响并受制于模型其他子系统</td>
      <td>注意力头维度如何限制 SRAM 瓦片大小；KV Cache 容量如何限制在线并发批大小</td>
    </tr>
    <tr>
      <td align="center"><strong>步骤 4</strong></td>
      <td><strong>精确性能数学公式</strong></td>
      <td>给出计算量（FLOPs）、访存量（Bytes）、算术强度与物理耗时的推导公式</td>
      <td>严谨推导 FLOPs 计算公式、HBM 往返数据量、前向与反向激活值显存计算</td>
    </tr>
    <tr>
      <td align="center"><strong>步骤 5</strong></td>
      <td><strong>具体微基准数字推导</strong></td>
      <td>在主流硬件（H100 SXM）与模型架构（LLaMA-3 8B/70B）上进行真实数据手算</td>
      <td>纯手工计算显存占用大小、总线带宽饱和度，以及理论与实测延迟对比</td>
    </tr>
    <tr>
      <td align="center"><strong>步骤 6</strong></td>
      <td><strong>核心系统工程铁律</strong></td>
      <td>提炼为 1-2 句简洁且关键的工程实践准则</td>
      <td>每一位系统工程师必须深刻铭记在心的架构决策经验法则</td>
    </tr>
  </tbody>
</table>

---

<h2 id="interaction-graph">3. 系统组件交互与硬件瓶颈全景图</h2>

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     物理加速芯片存储层级结构                                     │
│  [片上寄存器: ~64 KB/SM] ◄──► [片上共享内存 / SRAM: ~228 KB/SM] ◄──► [高带宽显存 (HBM)]          │
│  访问延迟: ~1 周期            访问延迟: ~20-30 周期                  │  访问延迟: ~200-400 周期  │
│  聚合带宽: ~33 TB/s           聚合带宽: ~15-20 TB/s                  │  总线带宽: ~2-3.35 TB/s   │
└───────────────────────────────────────────────┬──────────────────────┴───────────────────────────┘
                                                │
       ┌────────────────────────────────────────┴────────────────────────────────────────┐
       ▼                                                                                 ▼
┌────────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────┐
│                    Prefill 阶段 (提示词预填)           │  │                Decode 阶段 (自回归逐字生成)│
│  - 提示词输入: 并发批量处理 S 个 Token                │  │  - 每次仅生成 1 个新 Token                 │
│  - 物理工况: 计算密集型 (高算术强度)                   │  │  - 物理工况: 访存密集型 (低算术强度)       │
│  - 硬件瓶颈: Tensor Core 矩阵乘峰值算力                │  │  - 硬件瓶颈: HBM 至片上 SRAM 的带宽传输吞吐│
│  - 核心指标: 首字延迟 (TTFT)                           │  │  - 核心指标: 逐字生成延迟 (ITL)            │
└──────────────────────────┬─────────────────────────────┘  └─────────────────────┬──────────────────────┘
                           │                                                      │
                           ▼                                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             跨组件连锁影响机制                                         │
├───────────────────────────────────┬───────────────────────────────────┬────────────────────────────────┤
│       组件 A：注意力计算算子      │       组件 B：前馈网络 / SwiGLU   │       组件 C：显存与 KV Cache  │
├───────────────────────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ - FlashAttention: 在片上 SRAM 中  │ - 3 矩阵结构投影: W_gate,         │ - KV Cache 显存容量公式:       │
│   完成 QK^T 与 Softmax 循环熔合， │   W_up, W_down。占模型 66% 参数量 │   2 * 2 * n_layers * n_kv_heads│
│   HBM 访存由 O(S^2) 降至 O(S)。   │   和绝大部分前向反向 FLOPs。      │   * d_head * seq_len * batch。 │
│ - MHA vs GQA: GQA 将 KV 头数压缩  │ - 中间隐藏维度: 8/3 * d_model。   │ - PagedAttention 彻底消除内存  │
│   至原来的 1/8。显著降低自回归    │ - 反向传播激活值占据显存大头。    │   虚拟空间碎片化浪费。         │
│   阶段带宽压力，并发提升 4-8 倍。 │ - TP 拆分为列切与行切 (1 AllReduce│ - 解除最大序列长度静态预分配。 │
└───────────────────────────────────┴───────────────────────────────────┴────────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       分布式集群系统与编译引擎                                         │
│  [张量并行 Tensor Parallelism] ◄──► [流水线并行 Pipeline Parallelism] ◄──► [ZeRO-3 / FSDP 显存分片]   │
│  [算子级芯片深度熔合]          ◄──► [动态连续批处理 Continuous Batch]  ◄──► [投机解码验证加速]         │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

<h2 id="evolution-chain">4. 动手系统实验室</h2>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="22%">实验项目</th>
      <th align="left" width="22%">文档链接</th>
      <th align="left" width="28%">核心攻关目标</th>
      <th align="left" width="28%">攻克的物理瓶颈</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>实验 E1：GPU 屋顶模型分析器</strong></td>
      <td><a href="labs/01-roofline-profiler.zh.md">屋顶模型分析器</a></td>
      <td>编写 Python/PyTorch 性能分析器，自动提取算子算术强度，并绘制其在硬件天花板下的运行轨迹。</td>
      <td><strong>盲目优化盲区</strong>：精确判断算子是处于显存带宽受限还是计算受限区间。</td>
    </tr>
    <tr>
      <td><strong>实验 E2：极简 FlashAttention 算子</strong></td>
      <td><a href="labs/02-flash-attention-kernel.zh.md">FlashAttention 算子</a></td>
      <td>使用 Triton / C++ 实现分块平铺与在线 Softmax 循环，使中间注意矩阵始终保留在片上 SRAM 中。</td>
      <td><strong>O(S^2) 显存墙</strong>：杜绝注意力中间大矩阵在 HBM 中的显式分配与读写。</td>
    </tr>
    <tr>
      <td><strong>实验 E3：PagedAttention 分页引擎</strong></td>
      <td><a href="labs/03-paged-kv-cache.zh.md">分页 KV Cache 引擎</a></td>
      <td>纯 Python/NumPy 实现虚拟内存页表管理系统，支撑高并发自回归 Token 缓存的按需分配。</td>
      <td><strong>显存静态碎片化</strong>：避免连续静态缓冲区预分配造成的显存浪费，并发提升 3 倍。</td>
    </tr>
    <tr>
      <td><strong>实验 E4：双卡张量并行引擎</strong></td>
      <td><a href="labs/04-tensor-parallel-engine.zh.md">张量并行引擎</a></td>
      <td>基于 Megatron-LM 架构手工构建列并行与行并行线性层，结合 PyTorch 集合通信完成数值对齐。</td>
      <td><strong>单卡显存容量壁垒</strong>：将巨量权重参数与前向计算切分并均摊至多个物理 GPU 上。</td>
    </tr>
  </tbody>
</table>

---

<h2 id="module-0">模块 0：硬件图景与物理执行</h2>

- **[第 E00 章：芯片物理拓扑架构](module-0/00-silicon-anatomy.zh.md)**：HBM、SRAM、Tensor Core 与互联拓扑。
- **[第 E01 章：屋顶模型与算术强度](module-0/01-roofline-model.zh.md)**：计算密集型 vs 访存密集型工况判定。
- **[第 E02 章：内存物理布局、跨步与合并访存](module-0/02-memory-layouts-and-coalescing.zh.md)**：DRAM 突发传输机制与张量跨步对齐。

---

<h2 id="module-1">模块 1：芯片层级注意力子系统</h2>

- **[第 E03 章：朴素注意力的致命缺陷](module-1/03-naive-attention-memory-wall.zh.md)**：$O(S^2)$ 显存带宽墙与中间矩阵物化灾难。
- **[第 E04 章：FlashAttention (1, 2, 3) 演进](module-1/04-flash-attention.zh.md)**：SRAM 瓦片划分、在线 Softmax 与 Hopper TMA 异步拷贝。
- **[第 E05 章：KV Cache 显存解剖与 PagedAttention](module-1/05-kv-cache-and-paged-attention.zh.md)**：虚拟内存页表思想在 Token 缓存管理中的应用。
- **[第 E06 章：注意力架构变体深度权衡 (MHA, GQA, MLA)](module-1/06-mha-mqa-gqa-mla.md)**：多查询、分组查询与多头潜在注意力机制的硬件开销对比。

---

<h2 id="module-2">模块 2：线性投影、激活函数与显存带宽</h2>

- **[第 E07 章：规模化 GEMM 与 Tensor Core 矩阵乘](module-2/07-gemm-tensor-cores.zh.md)**：脉动阵列、Warp 级分块对齐与显存对齐规则。
- **[第 E08 章：SwiGLU 与前馈网络在芯片上的执行](module-2/08-swiglu-ffn-in-silicon.zh.md)**：3 矩阵投影架构、反向传播激活值占用与算子大分块熔合。
- **[第 E09 章：熔合 RMSNorm 与残差流工程](module-2/09-fused-rmsnorm-residuals.zh.md)**：算子深度熔合技术，彻底消除 2 次完整的 HBM 往返传输。
- **[第 E10 章：RoPE 旋转位置编码寄存器级即时计算](module-2/10-rope-kernel-fusion.zh.md)**：摆脱预计算查找表，在片上寄存器中即时求解三角变换。

---

<h2 id="module-3">模块 3：数值格式、计算精度与量化技术</h2>

- **[第 E11 章：硅基芯片中的数值格式 (FP32, FP16, BF16, FP8)](module-3/11-number-formats-fp8-bf16.zh.md)**：阶码与尾数的物理权衡，FP8 E4M3 与 E5M2 的分工。
- **[第 E12 章：量化策略实战 (W8A8, W4A16, INT4)](module-3/12-quantization-strategies-w8a8-w4a16.zh.md)**：仅权重量化 vs 权重-激活量化在 Prefill 与 Decode 阶段的适用场景。
- **[第 E13 章：KV Cache 显存量化压缩](module-3/13-kv-cache-quantization.zh.md)**：将上下文缓存压缩至 FP8 和 INT4 的硬件收益与精度保全。

---

<h2 id="module-4">模块 4：分布式系统与并行扩展</h2>

- **[第 E14 章：分布式集合通信原语](module-4/14-distributed-communication-primitives.zh.md)**：All-Reduce 环状与树状算法、Reduce-Scatter 与 All-Gather 通信代价。
- **[第 E15 章：张量并行 (Megatron-LM) 底层实现](module-4/15-tensor-parallelism-megatron.zh.md)**：列切与行切拆分策略，每层仅两次 All-Reduce 的通信约束。
- **[第 E16 章：流水线并行 (PP) 与 1F1B 调度算法](module-4/16-pipeline-parallelism-1f1b.zh.md)**：流水线气泡因子分析，微批次切分与激活值缓存上限。
- **[第 E17 章：数据并行与零冗余优化 (FSDP / ZeRO-1, 2, 3)](module-4/17-zero-fsdp-data-parallelism.zh.md)**：显存 16P 负担切分，以 50% 额外通信换取无界显存扩展。
- **[第 E18 章：上下文与序列并行 (Ring Attention)](module-4/18-ring-attention-context-parallelism.zh.md)**：环形拓扑通信与注意力计算完美重叠，突破单卡百万长文本瓶颈。

---

<h2 id="module-5">模块 5：高吞吐推理引擎与在线服务</h2>

- **[第 E19 章：两大推理工况深度解耦 (Prefill vs. Decode)](module-5/19-prefill-vs-decode-regimes.zh.md)**：首字延迟（计算密集）与逐字生成（访存密集）的物理本质。
- **[第 E20 章：连续批处理与动态调度引擎](module-5/20-continuous-batching-scheduling.zh.md)**：基于 Iteration 粒度的动态调度，消灭静态批处理填充气泡。
- **[第 E21 章：投机解码：利用过剩算力加速生成](module-5/21-speculative-decoding.zh.md)**：草稿模型推测与大模型并行验证，打破单 Token 访存墙。
- **[第 E22 章：分块预填与前缀缓存技术](module-5/22-chunked-prefill-prefix-caching.zh.md)**：将长提示词切片调度以平抑 ITL 抖动，利用 Radix Tree 实现前缀零开销复用。

---

<h2 id="module-6">模块 6：混合专家模型（MoE）工程架构</h2>

- **[第 E23 章：MoE 系统架构：Top-K 门控与容量因子](module-6/23-moe-routing-capacity-factor.zh.md)**：激活参数与总参数解耦，专家容量超载丢弃策略。
- **[第 E24 章：All-to-All 跨机调度与通信瓶颈](module-6/24-all-to-all-distributed-moe.zh.md)**：分布式专家并行中的交换网络压力与双流水线（DualPipe）通信重叠。

---

<h2 id="module-7">模块 7：显存审计、性能分析与硬件利用率</h2>

- **[第 E25 章：大模型显存预算全景公式](module-7/25-model-memory-budget.zh.md)**：训练 16P 静态显存、优化器状态与前向反向激活值显存的精准计算。
- **[第 E26 章：真实硬件利用率度量 (MFU, HFU 与 Profiling)](module-7/26-measuring-mfu-hfu-profiling.zh.md)**：从理论 Peak 算力到实际模型浮点利用率（MFU）的科学测量与 Nsight 实战。
