# 第 E18 章：上下文与序列并行（Ring Attention）：环形拓扑通信与注意力计算完美重叠，突破单卡百万长文本瓶颈

## 步骤 1：物理直觉

想象一场横跨 8 座城市的超级国际接力阅读马拉松，要一口气研读一本长达 100 万页的巨著：
- 任何一座城市的图书馆都装不下整本书；
- 于是，每座城市各自保管其中的 12.5 万页（**上下文序列切分**）；
- **环状注意力（Ring Attention）：**
  - 城市 0 的研究员们手头拿着自己的问题清单（Query 分块），首先研读自己桌上的前 12.5 万页（Key/Value 分块）；
  - 研读完毕后，城市 0 把这 12.5 万页复印件通过高铁发给下座城市，同时从上游城市接收对方的 12.5 万页；
  - 高铁在铁轨上疾驰运送书页的同时，研究员们**一秒钟也不停歇**，立刻埋头研读刚刚送到的新书页！
  - 只要高铁运送书页的耗时，短于研究员研读一箱书的时间，**通信所产生的等待延迟就彻底变成了 0**！

---

## 步骤 2：芯片微观底层执行机制

标准 FlashAttention 能够解决单卡内部 $O(S)$ 的显存问题，但当序列长度达到 $S = 100\text{ 万}$ 时，即使是切分后的 KV Cache 本身也会突破单卡 $80\text{ GB}$ 的物理上限。  
**Ring Attention（Liu et al., 2023）** 将序列维度切分给 $N$ 个 GPU，并组织为环形拓扑：

```
GPU 环形调度 (Ring Topology, N = 4):
GPU 0: 拥有 Q0, 初始拥有 K0, V0
GPU 1: 拥有 Q1, 初始拥有 K1, V1
GPU 2: 拥有 Q2, 初始拥有 K2, V2
GPU 3: 拥有 Q3, 初始拥有 K3, V3

在 Step k (共 N 步迭代):
1. 在非阻塞通信流 (Stream 1) 中: 异步向右侧 GPU 发送当前持有的 K_curr, V_curr, 并从左侧接收 K_next, V_next (P2P Send/Recv)
2. 在核心计算流 (Stream 0) 中: 同步在 Tensor Core 上执行 FlashAttention 局部块计算:
   FlashAttn_Step(Q_local, K_curr, V_curr, 在线 Softmax 状态更新)
3. 同步两个 CUDA 流: 此时下一块 K, V 已经无缝送达 SRAM!
```

只要局部 FlashAttention 的计算耗时 $T_{\text{compute}}$ 大于点对点 P2P 通信耗时 $T_{\text{comm}}$，跨卡通信就被**完全隐蔽在计算阴影之中**！

---

## 步骤 3：跨组件相互耦合机制

1. **因果掩码（Causal Mask）的非对称负载均衡：**  
   在自回归因果注意力中，矩阵是一个下三角。如果简单切分，排在后面的 GPU 计算量远大于排在前面的 GPU。工业级实现（如 Zig-Zag Ring Attention）通过交错交织分配 Token 序列块，使得所有 GPU 的计算负载完全拉平。
2. **与张量并行（TP）的正交组合：**  
   长文本处理中，序列并行（Context Parallelism, CP）通常与张量并行正交结合：机内 8 卡跑 TP，跨机多节点通过 InfiniBand 环路跑 CP，轻松支撑起数百万 Token 的端到端训练与推理。

---

## 步骤 4：精确性能数学公式

设序列总长度为 $S$，上下文并行度为 $P_{\text{cp}}$，单头维度为 $d$。每个 GPU 负责的局部序列长度为：

$$S_{\text{local}} = \frac{S}{P_{\text{cp}}}$$

在单次环迭代中，每个 GPU 执行局部分块注意力的浮点运算量为：

$$\text{FLOPs}_{\text{step}} = 4 \times H \times S_{\text{local}}^2 \times d$$

单步点对点传输的 Key 和 Value 数据字节数为：

$$\text{Bytes}_{\text{step}} = 2 \times (2 \times H \times S_{\text{local}} \times d) = 4 H S_{\text{local}} d \text{ 字节 (FP16)}$$

通信被完全隐藏的充要物理条件是计算耗时覆盖通信耗时：

$$T_{\text{compute}} \ge T_{\text{comm}} \implies \frac{4 H S_{\text{local}}^2 d}{P_{\text{tensor\_core}}} \ge \frac{4 H S_{\text{local}} d}{\text{BW}_{\text{interconnect}}}$$

两端约去公共因子，得到极其优雅的**无感长文本物理临界公式**：

$$S_{\text{local}} \ge \frac{P_{\text{tensor\_core}}}{\text{BW}_{\text{interconnect}}}$$

只要每个 GPU 上切分到的局部序列长度 $S_{\text{local}}$ 大于当前互联网络的硬件平衡比，**通信开销就在物理上完全隐形**！

---

## 步骤 5：具体微基准数字推导

以单卡配备 400 Gbps（单向有效带宽 $\text{BW} = 45\text{ GB/s}$）InfiniBand 网卡的 H100 节点（实际有效算力取 $600\text{ TFLOPS}$）为例：

1. **计算零延迟通信所需的单卡局部序列下界：**
   $$S_{\text{local}} \ge \frac{600 \times 10^{12} \text{ FLOPs/s}}{45 \times 10^9 \text{ Bytes/s}} \approx 13,333 \text{ 个 Token}$$
2. **工程含义：**
   只要每张卡分到的 Token 数不少于约 **14K**：
   - 跨 8 张 GPU 组网，即可实现 **11 万 Token** 的无通信开销极限长文本；
   - 跨 64 张 GPU 组网，即可实现 **100 万 Token**（1M 上下文）的超长文本全速无感计算！
   - 通信开销被 100% 隐藏，集群算力利用率（MFU）与处理短序列时完全相同！

---

## 步骤 6：核心系统工程铁律

> 真正的系统扩展性不是消灭通信，而是让通信与计算完美重叠。Ring Attention 利用双缓冲流水线将环形 P2P 传输完全掩盖在 FlashAttention 分块计算的阴影之中，让百万级长上下文的计算复杂度在多卡集群中重返优雅的线性扩展轨道。
