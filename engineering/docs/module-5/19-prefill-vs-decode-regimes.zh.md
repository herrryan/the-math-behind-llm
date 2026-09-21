# 第 E19 章：两大推理工况深度解耦：Prefill（计算密集型）与 Decode（访存密集型）的物理本质

## 步骤 1：物理直觉

想象一家超级餐厅面对两类截然不同的做菜任务：
- **第一类：宴席前总备菜（Prefill 预填）：** 厨房收到一张长长的宴席菜单（4000 字提示词）。主厨搬来一整车食材，大锅大灶同时猛火翻炒，所有的锅炉、大勺和刀具火力全开满负荷做功，香气冲天（Tensor Core 算力被 100% 榨干）。虽然很累，但出菜极具规模效应。
- **第二类：给包厢客人逐粒喂瓜子（Decode 逐字生成）：** 客人每次只嚼一颗瓜子，嚼完了才慢悠悠点下一颗。为了伺候这位客人，服务员每次都必须大老远跑去地下总冷库，把整座冷库里沉重无比的大箱子扛出来，小心翼翼撕开包装，只取出一颗瓜子递给客人，然后再把整箱推回冷库。主厨和猛火大灶在旁边闲得发慌打哈欠，服务员的腿却快要跑断了！

在大语言模型推理中，**Prefill（处理输入提示词）**与 **Decode（自回归逐字生成）**在物理底层根本就是两种完全不同性质的计算机工作流！

---

## 步骤 2：芯片微观底层执行机制

对比两者的芯片微观执行特征：

```
工况 A: 提示词预填阶段 (Prefill Phase)
- 输入形状: 批量输入 S 个已存在的 Token 序列 (如 S = 2048, 4096)
- 核心算子: 大规模矩阵乘 (GEMM)
- 算术强度 I: 极高 (I = 100 ~ 500+ FLOPs/Byte)
- 物理状态: 计算密集型 (Compute-Bound), 昂贵的 Tensor Core 满负荷高效轰鸣!
- 关键用户指标: 首字延迟 TTFT (Time to First Token)

工况 B: 自回归生成阶段 (Decode Phase)
- 输入形状: 每次输入仅仅 1 个全新 Token (B 个并发序列各输入 1 个)
- 核心算子: 矩阵向量乘 (GEMV) 与长 KV Cache 聚合
- 算术强度 I: 极低 (当 B 较小时, I = 1 ~ 5 FLOPs/Byte)
- 物理状态: 极度访存受限 (Memory-Bound), Tensor Core 95% 以上的时间处于闲置等待!
- 关键用户指标: 逐字生成延迟 ITL (Inter-Token Latency) 与系统吞吐量 (Tokens/s/GPU)
```

### 传统混部调度的灾难性冲突：
若将 Prefill 与 Decode 简单混合在同一个 GPU 实例中执行：
- 一个突然涌入的长提示词 Prefill 请求（需耗时数百毫秒计算），会霸占整个 GPU 的 Tensor Core；
- 导致正在以 30ms 匀速吐字的几十个 Decode 任务瞬间发生严重的**生成停顿卡顿（ITL 抖动 Jitter）**，极端损害交互体验。

---

## 步骤 3：跨组件相互耦合机制

1. **解耦式推理架构（Disaggregated Serving）：**  
   工业界最前沿的推理系统（如 Splitwise、DistServe）开始在物理上将硬件集群拆分为专职的 **Prefill 节点池** 与 **Decode 节点池**。Prefill 节点配置高算力卡，跑完后通过高速网络直接将生成的 KV Cache 传递给专攻超高带宽的 Decode 节点池，彻底根除两者互相抢占资源的冲突。
2. **批大小与带宽利用率的救赎：**  
   在 Decode 节点，提高系统吞吐的唯一办法是尽可能提升并发批大小 $B$，使更多的 Token 共享同一次全量权重的 HBM 读取。

---

## 步骤 4：精确性能数学公式

单步物理执行延迟分解公式：

$$T_{\text{prefill}} \approx \frac{2 \cdot P_{\text{model}} \cdot S}{P_{\text{peak\_compute}}} + T_{\text{flash\_attn}}(S)$$

$$T_{\text{decode}} \approx \frac{2 \cdot P_{\text{model}}}{\text{BW}_{\text{mem}}} + \frac{\text{KV\_Cache\_Bytes}(B, S)}{\text{BW}_{\text{mem}}}$$

在 Decode 阶段，生成每个 Token 的系统边际耗时随着并发批大小 $B$ 扩大而大幅摊薄：

$$\text{单 Token 边际耗时} = \frac{T_{\text{decode}}}{B} \approx \frac{2 P_{\text{model}}}{B \cdot \text{BW}_{\text{mem}}} + \frac{\text{单请求 KV 字节}}{\text{BW}_{\text{mem}}}$$

当 $B$ 足够大时，静态模型参数权重的搬运时间被完全平摊，系统吞吐量逼近物理理论极限！

---

## 步骤 5：具体微基准数字推导

以 LLaMA-3 70B 模型在 H100 SXM（算力 $1000\text{ TFLOPS}$，带宽 $3.35\text{ TB/s}$）上对比两种工况：

### 1. Prefill 阶段（处理一条 4096 长度的 Prompt）：
- 计算量：$F = 2 \times 70 \times 10^9 \times 4096 \approx 5.73 \times 10^{14} \text{ FLOPs} = 573 \text{ TFLOPs}$
- 物理耗时（按 60% 算力利用率即 600 TFLOPS 计算）：
  $$T_{\text{prefill}} \approx \frac{573 \times 10^{12}}{600 \times 10^{12}} \approx 0.955 \text{ 秒} \approx 955 \text{ 毫秒}$$
- 这一千毫秒内，GPU 算力被压榨到极致。

### 2. Decode 阶段（为该单请求生成 1 个新 Token，批大小 $B=1$）：
- 计算量：仅为 $140\text{ GFLOPs}$
- 但必须把全量 $140\text{ GB}$ 权重从 HBM 读取一遍：
  $$T_{\text{decode}} \approx \frac{140 \times 10^9}{3.35 \times 10^{12}} \approx 41.8 \text{ 毫秒}$$
- 此时单步计算实际仅需 $0.14\text{ ms}$，其余 $41.66\text{ ms}$（占比 $99.6\%$）全是在纯等待显存搬家！

---

## 步骤 6：核心系统工程铁律

> 混淆 Prefill 与 Decode 是大模型推理系统设计的大忌。前者是 Tensor Core 算力的巅峰碰撞，追求最高算术强度与低 TTFT；后者是 HBM 显存带宽的无休止折磨，追求最大并发摊销与极速 ITL。只有从物理层面对二者实施解耦与差异化调度，才能构筑最高效的生产级推理引擎。
