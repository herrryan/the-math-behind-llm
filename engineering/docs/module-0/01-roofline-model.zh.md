# 第 E01 章：屋顶模型与算术强度：计算密集型 vs 访存密集型工况判定

## 步骤 1：物理直觉

想象你在景区开了一家网红鲜榨果汁摊，摊位上有两套核心装备：
1. **备料工人（内存搬运带宽）：** 负责跑去地下冷库搬出一箱箱冻草莓，并将草莓倒进榨汁机杯子里。这个工人每分钟最多只能拆箱倒进 **3 公斤草莓**。
2. **大功率破壁机（计算核心 / Tensor Core）：** 一台工业级高速破壁机，转速高达每分钟 **2,000 转**。

现在，第一位顾客点了一杯**清爽草莓水**（只需放 1 颗草莓，加满 1 升纯净水，破壁机轻轻转动 2 圈搅拌均匀即可）。备料工人为了搬来大桶纯净水和这 1 颗草莓，满头大汗忙碌了 20 秒；而破壁机只工作了 0.1 秒就立刻停机了。在剩下的 19.9 秒里，昂贵的破壁机都在干等着工人去搬下一桶水。此时，果汁摊的瓶颈在于搬运工，属于**搬运受限（访存密集型 Memory-Bound）**。

接着，第二位顾客点了一碗**超浓草莓果泥酱**（不需要加水，但要求破壁机对每颗草莓高速研磨 1,000 转打碎成微米级细糊）。这时，工人只需搬来一小碗草莓倒进去，破壁机就必须全功率高速轰鸣运转整整 30 秒。在这个过程中，工人甚至可以在旁边喝茶休息，破壁机依然在全力输出。此时，果汁摊的瓶颈在于机器打磨能力，属于**破壁机受限（计算密集型 Compute-Bound）**。

**机器做功的次数（FLOPs）**与**搬运食材的重量（Bytes）**之比，在计算机体系结构中被称为**算术强度（Operational Intensity）**。而**屋顶模型（Roofline Model）**，就是告诉你这行代码到底是在等破壁机还是在等搬运工的物理诊断金标准。

---

## 步骤 2：芯片微观底层执行机制

屋顶模型（Williams, Waterman, & Patterson, 2009）是一个将硬件峰值算力、显存物理带宽与算法软件本身的算术强度有机统一的二维理论性能边界图：

```
可达到的算力性能 [TFLOPS]
         ▲
峰值     │                          /──────────────────────── (算力天花板: P_peak)
算力     │                         /
天花板   │                        /
         │                       /
         │                      / ◄── 显存带宽倾斜天花板
         │                     /      斜率 = 物理显存带宽 (BW_mem)
         │                    /
         │                   /
         │                  /
         │                 /
         │                / │
         └───────────────┴──┴────────────────────────────────► 算术强度 I
                         0  I* (硬件平衡临界点)                  [FLOPs / Byte]
                          ▲
                          │
          [访存受限区间]  │  [计算受限区间]
          I < I*          │  I > I*
```

### 核心物理参数解析：
1. **$P_{\text{peak}}$（硬件峰值计算吞吐）：** 芯片在当前数据精度下的理论最高浮点计算能力（例如 NVIDIA H100 SXM 在 FP16 密集计算下为 $1,000\text{ TFLOPS}$，在 2:4 结构化稀疏下为 $2,000\text{ TFLOPS}$）。
2. **$\text{BW}_{\text{mem}}$（硬件峰值显存带宽）：** 字节能够从 HBM 运入片上缓存的最大物理速率（例如 H100 SXM 的 HBM3 带宽为 $3.35\text{ TB/s}$）。
3. **算术强度（Operational Intensity，记作 $I$）：** 算法执行过程中，每跨越显存总线搬运 1 个字节所能完成的浮点运算次数：
   $$I = \frac{\text{算法总浮点运算量 (FLOPs)}}{\text{算法往返于 HBM 的总访存量 (Bytes)}} \quad \left[\frac{\text{FLOP}}{\text{Byte}}\right]$$
4. **$I^*$（硬件平衡拐点 / 屋檐转折点）：** 区分访存密集型与计算密集型的关键分水岭：
   $$I^* = \frac{P_{\text{peak}}}{\text{BW}_{\text{mem}}}$$

对于一台 NVIDIA H100 SXM GPU（取稠密 FP16 峰值 $P_{\text{peak}} \approx 1,000\text{ TFLOPS}$，HBM3 带宽 $\text{BW}_{\text{mem}} = 3.35\text{ TB/s}$）：

$$I^* = \frac{1,000 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx 298.5 \text{ FLOPs/Byte}$$

这表明：**在 H100 上，一个算子每从 HBM 加载或写回 1 个字节，必须在片上对其进行至少 298 次浮点乘加运算，才有可能让 Tensor Core 发挥出 100% 的全部算力！**

---

## 步骤 3：跨组件相互耦合机制

LLM 架构中的每一个算子，都严格处于屋顶模型的某一个特定区间：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="22%">LLM 核心算子</th>
      <th align="left" width="20%">典型算术强度 $I$</th>
      <th align="left" width="18%">物理运行工况</th>
      <th align="left" width="40%">硬件瓶颈与工业优化手段</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Prefill 矩阵乘 (提示词预填)</strong></td>
      <td>$I = 100\text{--}500+$</td>
      <td>计算密集型</td>
      <td>受 Tensor Core 峰值算力限制。优化策略：增大分块尺寸，维度对齐至 64/128 整数倍。</td>
    </tr>
    <tr>
      <td><strong>Decode 矩阵向量乘 (逐字生成)</strong></td>
      <td>$I = 1\text{--}5$ (当 $B=1$)</td>
      <td>深度访存受限</td>
      <td>受 HBM 显存带宽限制。优化策略：增大批大小、采用 W4/W8 权重量化、采用 GQA 削减 KV 头数。</td>
    </tr>
    <tr>
      <td><strong>LayerNorm / RMSNorm</strong></td>
      <td>$I \approx 2$</td>
      <td>深度访存受限</td>
      <td>受 HBM 读写带宽限制。优化策略：必须与残差相加熔合为单 CUDA 算子。</td>
    </tr>
    <tr>
      <td><strong>Softmax 归一化</strong></td>
      <td>$I \approx 2.5$</td>
      <td>深度访存受限</td>
      <td>受 HBM 带宽限制。优化策略：采用 FlashAttention 在片上 SRAM 中做在线归一化。</td>
    </tr>
    <tr>
      <td><strong>逐元素激活函数 (GELU/Swish)</strong></td>
      <td>$I \approx 1\text{--}2$</td>
      <td>深度访存受限</td>
      <td>受 HBM 带宽限制。优化策略：直接熔合到上一层 GEMM 的输出尾声（Epilogue）中执行。</td>
    </tr>
  </tbody>
</table>

### 批大小（Batch Size）如何扭转受限状态？
对比矩阵乘向量（GEMV）与矩阵乘矩阵（GEMM）：
- 当批大小 $B = 1$（自回归单请求解码）：输入向量 $\mathbf{x} \in \mathbb{R}^{1 \times K}$ 乘以权重矩阵 $\mathbf{W} \in \mathbb{R}^{K \times N}$。
  $$\text{总计算量} = 2 K N \text{ FLOPs}$$
  $$\text{HBM 访存量} = 2 K N \text{ (权重)} + 2 K \text{ (输入)} + 2 N \text{ (输出)} \approx 2 K N \text{ 字节 (FP16)}$$
  $$\text{算术强度 } I = \frac{2 K N}{2 K N} = 1.0 \text{ FLOP/Byte}$$
  因为 $1.0 \ll 298.5$，单个用户的自回归生成深陷在**显存带宽受限区**的最底部。

- 当批大小 $B = 128$ 时（多并发批处理解码或提示词预填）：输入矩阵 $\mathbf{X} \in \mathbb{R}^{B \times K}$ 乘以 $\mathbf{W} \in \mathbb{R}^{K \times N}$。
  $$\text{总计算量} = 2 B K N \text{ FLOPs}$$
  $$\text{HBM 访存量} \approx 2 K N \text{ (同一份权重被并发复用了 } B \text{ 次!)} + 2 B K + 2 B N$$
  $$\text{算术强度 } I \approx \frac{2 B K N}{2 K N} = B \text{ FLOPs/Byte} = 128 \text{ FLOPs/Byte}$$
  随着并发批大小 $B$ 提升至 300 以上，算术强度跨越了硬件临界拐点 $I^*$，算子成功由访存密集型蜕变为**计算密集型**！

---

## 步骤 4：精确性能数学公式

在屋顶模型框架下，算子所能达到的实际最大浮点计算吞吐量 $P(I)$ 由两条折线共同约束：

$$P(I) = \min\left(P_{\text{peak}}, \; I \times \text{BW}_{\text{mem}}\right)$$

对于总浮点运算量为 $F$、往返显存流量为 $M$ 字节的算子，其实际物理执行耗时下界为：

$$T_{\text{exec}} = \max\left(\frac{F}{P_{\text{peak}}}, \; \frac{M}{\text{BW}_{\text{mem}}}\right)$$

由此，模型浮点利用率（Model FLOPs Utilization，MFU）可精确表达为：

$$\text{MFU} = \frac{P(I)}{P_{\text{peak}}} = \min\left(1, \; \frac{I \times \text{BW}_{\text{mem}}}{P_{\text{peak}}}\right) = \min\left(1, \; \frac{I}{I^*}\right)$$

这意味着，当算子的算术强度低于硬件临界点时（$I < I^*$）：

$$\text{MFU} = \frac{I}{I^*}$$

这个公式揭示了一条冷酷的物理规律：**只要一个算子处于访存受限区间，其最高硬件算力利用率在物理上就被 $\frac{I}{I^*}$ 彻底锁死，芯片上堆叠再多的 Tensor Core 对其提速也毫无意义！**

---

## 步骤 5：具体微基准数字推导

我们来手算一段在大模型中频繁调用的 **RMSNorm** 算子在真实 NVIDIA H100 SXM 硬件上的执行指标：
- 隐藏层维度：$d_{\text{model}} = 8192$（LLaMA-3 70B）
- 批大小：$S = 2048$ 个 Token
- 数据精度：BF16（每元素 2 字节）
- 硬件规格：NVIDIA H100（$P_{\text{peak}} = 1,000\text{ TFLOPS}$, $\text{BW}_{\text{mem}} = 3.35\text{ TB/s}$, $I^* = 298.5\text{ FLOPs/Byte}$）

### 步骤 5.1：计算该算子的总计算量
RMSNorm 对每个 Token 向量 $\mathbf{x}$ 计算：
$$y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$$
1. 向量各元素平方：$d$ 次乘法
2. 求和累加：$d$ 次加法
3. 除以 $d$、加 $\epsilon$、开方：3 次标量运算
4. 各元素除以均方根并乘以缩放因子 $\gamma_i$：$2d$ 次运算
每个 Token 的计算量约为 $4d$ FLOPs。

$$\text{总计算量 } F = 2048 \times (4 \times 8192) \approx 6.71 \times 10^7 \text{ FLOPs} = 67.1 \text{ MFLOPs}$$

### 步骤 5.2：计算该算子的未熔合 HBM 访存量
- 从 HBM 读取输入向量 $\mathbf{x}$：$2048 \times 8192 \times 2 \text{ 字节} = 33.55\text{ MB}$
- 从 HBM 读取缩放权重 $\boldsymbol{\gamma}$：$8192 \times 2 \text{ 字节} \approx 0.016\text{ MB}$
- 向 HBM 写回输出向量 $\mathbf{y}$：$2048 \times 8192 \times 2 \text{ 字节} = 33.55\text{ MB}$
$$\text{总显存流量 } M \approx 67.12 \text{ MB}$$

### 步骤 5.3：求解算术强度与硬件利用率
$$I = \frac{67.1 \times 10^6 \text{ FLOPs}}{67.12 \times 10^6 \text{ Bytes}} \approx 1.0 \text{ FLOP/Byte}$$

该算子能达到的最大计算吞吐为：
$$P(1.0) = \min(1000\text{ TFLOPS}, \; 1.0 \times 3.35\text{ TB/s}) = 3.35 \text{ TFLOPS}$$

实际可达到的 MFU：
$$\text{MFU} = \frac{1.0}{298.5} \approx 0.33\%$$

### 步骤 5.4：计算物理耗时
$$T_{\text{exec}} = \frac{67.12 \times 10^6 \text{ Bytes}}{3.35 \times 10^{12} \text{ Bytes/s}} \approx 20.0 \times 10^{-6} \text{ 秒} = 20.0 \;\mu\text{s}$$

而按 Tensor Core 算力计算，$67.1\text{ MFLOPs}$ 理论上仅需 $0.067\;\mu\text{s}$ 即可算完。由于显存搬运耗时高达 $20.0\;\mu\text{s}$，算子实际运行速度比纯计算能力慢了整整 **300 倍**。

---

## 步骤 6：核心系统工程铁律

> 当算子的算术强度处于硬件平衡拐点 $I^*$ 以下时，对数学逻辑本身的优化毫无效果。在访存受限区间，性能提升的唯一途径是减少 HBM 搬运总量（通过算子深度熔合或量化压缩），或者大幅提升数据复用率（通过扩大并发批大小或片上 SRAM 缓存）。
