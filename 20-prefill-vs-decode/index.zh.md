# 第 20 章：双阶引擎（预填充 vs 解码与屋顶模型）

> [!INTUITION] 步骤 1：3 岁小孩直觉
> 想象你开了一家玩具快递公司，旗下拥有一辆能装载 50 吨货物的巨型黄色重卡，以及一位骑自行车的快递小哥。
> 
> 当一所学校一次性订购 1,000 块积木时，你把 1,000 块积木严丝合缝地塞满整辆重型卡车。卡车的澎湃引擎轰鸣，轮胎紧咬地面，只跑一次高速公路就把所有积木全部运达。卡车里的每一立方分米空间都被充分利用——这就是 **预填充（Prefill）**。
> 
> 但是，如果学校老师要求你每隔一小时送一颗弹珠：*“先送一颗红弹珠……等一小时……再送一颗蓝弹珠……”*
> 
> 为了把这一颗小弹珠送到学校，你不能靠手扔过去。你必须发动那辆 50 吨重的巨型卡车，在高速公路上呼啸驶过，卸下一颗弹珠，再大费周章地开回仓库，然后下一小时再来一次。卡车的千匹马力根本没有用在弹珠上，99% 的时间与燃油都白白消耗在巨型卡车本身的来回奔波上——这就是 **解码（Decode）**。
> 
> 大语言模型的推理绝不是单一形态的任务，它在硬件底层拥有两幅截然相反的面孔：一个是处理完整提示词的重载货运火车，另一个是逐字吐字、被卡车自重拖累的单程快递员。

---

## 步骤 2：承前启后的关键过渡

我们如何把“一次性运送 1,000 块积木”与“为了送一颗弹珠而频繁跑空车”的物理直觉，转化成计算机芯片可精确度量的数学语言？

在计算机硬件底层，有两大物理构件直接决定着系统的运行极限：
1. **算力引擎（计算核心 / 张量核心 Tensor Cores）**：芯片每秒能完成多少次浮点加法与乘法运算（单位：<abbr title="每秒浮点运算次数">FLOP/s</abbr>）。
2. **显存高速路（高带宽显存 HBM / DRAM）**：芯片每秒能将多少吉字节（GB）的数字从显存搬运到算力引擎的寄存器中（单位：Bytes/s）。

这里的核心过渡问题是：
$$\text{从显存仓库每搬运 1 字节的数据，我们的算力引擎能对它执行多少次数学计算？}$$

如果每搬运 1 字节数据，就能对它反复计算数百次，算力引擎就能全速运转；如果每搬运 1 字节数据，却仅仅计算一次就丢弃，算力引擎就只能长时间停工闲置，苦苦等待显存数据的低速输送。

---

## 步骤 3：严谨数学公式与推导

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        GPU 屋顶模型（ROOFLINE MODEL）与算术强度                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 实际性能 (TFLOPS)                                                                      │
│       ▲                                                                                │
│ P_peak│───────────────────────────────┐ &lt;── 平顶天花板：算力受限区（Compute-Bound）    │
│       │                              /      （预填充 Prefill / 矩阵-矩阵 GEMM）        │
│       │                             /                                                  │
│       │                            /                                                   │
│       │                           / &lt;────── 倾斜斜坡：显存带宽受限区（Memory-Bound）   │
│       │                          /          （逐字解码 Decode / 矩阵-向量 GEMV）       │
│       │                         /                                                      │
│       │                        /                                                       │
│      0└───────────────────────┴──────────────────────────────────────►                 │
│       0                       I* (性能拐点)            算术强度 / 计算访存比 (FLOPs/B) │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 20.1:</strong> GPU 屋顶模型。在临界拐点 I* 左侧，运行吞吐被显存带宽严格卡死；在 I* 右侧，计算吞吐达到芯片硬件的峰值算力天花板。</figcaption>
</figure>

### 1. 算术强度（Operational Intensity）

设 $\text{FLOPs}$ 为某一计算步骤所执行的浮点运算总次数，$\text{Bytes}$ 为该步骤在主显存（<abbr title="高带宽显存">HBM</abbr>）与片上缓存（SRAM/寄存器）之间搬运的实际数据字节数。

<dfn id="def-intensity-zh">算术强度（计算访存比）</dfn> $I$ 定义为：

$$
I = \frac{\text{FLOPs}}{\text{Bytes Transfers}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
$$

### 2. 屋顶模型公式（The Roofline Model）

设：
- $P_{\text{peak}}$ 为芯片的理论峰值浮点计算吞吐量（例如 $\text{TFLOPS} = 10^{12} \text{ FLOP/s}$）。
- $B_{\text{peak}}$ 为显存总线的峰值带宽吞吐量（例如 $\text{TB/s} = 10^{12} \text{ Byte/s}$）。

任意算法在硬件上所能达到的实际最大计算性能 $P(I)$ 由算力上限与访存上限的交集界定：

$$
P(I) = \min\left(P_{\text{peak}}, \; I \times B_{\text{peak}}\right)
$$

### 3. 临界性能拐点（The Critical Ridge Point, $I^*$）

当显存带宽所能供给的算力恰好等于硬件的绝对算力上限时，对应的算术强度称为**拐点**：

$$
I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}}
$$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 20.1:</strong> 现代主流 AI 加速芯片在 16 位半精度（BF16/FP16）下的硬件参数与拐点 $I^*$。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">加速卡型号</th>
      <th align="right">峰值张量算力（$P_{\text{peak}}$）</th>
      <th align="right">峰值显存带宽（$B_{\text{peak}}$）</th>
      <th align="right">临界算术强度拐点（$I^*$）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>NVIDIA A100 (SXM4 80GB)</strong></td>
      <td align="right">312 TFLOPS (BF16)</td>
      <td align="right">2.039 TB/s</td>
      <td align="right"><strong>153.0 FLOPs/Byte</strong></td>
    </tr>
    <tr>
      <td><strong>NVIDIA H100 (SXM5 80GB)</strong></td>
      <td align="right">989 TFLOPS (BF16)</td>
      <td align="right">3.350 TB/s</td>
      <td align="right"><strong>295.2 FLOPs/Byte</strong></td>
    </tr>
    <tr>
      <td><strong>NVIDIA B200 (SXM 192GB)</strong></td>
      <td align="right">2,250 TFLOPS (BF16)</td>
      <td align="right">8.000 TB/s</td>
      <td align="right"><strong>281.3 FLOPs/Byte</strong></td>
    </tr>
  </tbody>
</table>

### 4. 阶段 1：预填充阶段（Prefill Phase - GEMM）

在 <dfn id="def-prefill-zh">预填充阶段</dfn>（提示词处理阶段），用户输入长度为 $T_{\text{prompt}}$ 的提示词。模型并行处理所有这 $T_{\text{prompt}}$ 个词元。

对于模型中的任意一个线性投影权重矩阵 $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 与输入激活矩阵 $\mathbf{X} \in \mathbb{R}^{T_{\text{prompt}} \times d_{\text{in}}}$：
- 矩阵乘法为通用矩阵-矩阵乘（<abbr title="General Matrix-Matrix Multiplication">GEMM</abbr>）：
  $$\mathbf{Y} = \mathbf{X} \mathbf{W} \in \mathbb{R}^{T_{\text{prompt}} \times d_{\text{out}}}$$
- **计算量**：每个输出元素包含 $d_{\text{in}}$ 次乘加运算（$2 d_{\text{in}}$ FLOPs），总浮点操作为：
  $$\text{FLOPs}_{\text{prefill}} = 2 \times T_{\text{prompt}} \times d_{\text{in}} \times d_{\text{out}}$$
- **显存访存量**：采用 16 位半精度存储权重，每个参数占 $p = 2$ 字节。当模型层数较深时，权重读取占主导地位：
  $$\text{Bytes}_{\text{prefill}} \approx 2 \times d_{\text{in}} \times d_{\text{out}}$$
- **算术强度**：
  $$
  I_{\text{prefill}} = \frac{2 \times T_{\text{prompt}} \times d_{\text{in}} \times d_{\text{out}}}{2 \times d_{\text{in}} \times d_{\text{out}}} = T_{\text{prompt}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
  $$

若用户提示词长度 $T_{\text{prompt}} = 1024$，在 NVIDIA H100（$I^* \approx 295$）上：
$$I_{\text{prefill}} = 1024 > 295$$
预填充阶段稳稳落在**算力受限区（Compute-Bound）**！GPU 的 Tensor Cores 被充分喂饱，实际执行性能无限逼近理论峰值 $P_{\text{peak}}$。

### 5. 阶段 2：逐字解码阶段（Decode Phase - GEMV）

在 <dfn id="def-decode-zh">解码阶段</dfn>（自回归文本生成阶段），模型在时间步 $t$ 只能自回归地生成**单个**最新词元。

输入为当前步单一词元的行向量 $\mathbf{x}_t \in \mathbb{R}^{1 \times d_{\text{in}}}$：
- 矩阵计算退化为通用矩阵-向量乘（<abbr title="General Matrix-Vector Multiplication">GEMV</abbr>）：
  $$\mathbf{y}_t = \mathbf{x}_t \mathbf{W} \in \mathbb{R}^{1 \times d_{\text{out}}}$$
- **计算量**：
  $$\text{FLOPs}_{\text{decode}} = 2 \times 1 \times d_{\text{in}} \times d_{\text{out}}$$
- **显存访存量**：为了计算这个单一行向量，GPU 必须将权重矩阵 $\mathbf{W}$ 的全部参数完整从显存搬运进计算寄存器：
  $$\text{Bytes}_{\text{decode}} \approx 2 \times d_{\text{in}} \times d_{\text{out}}$$
- **算术强度**：
  $$
  I_{\text{decode}} = \frac{2 \times 1 \times d_{\text{in}} \times d_{\text{out}}}{2 \times d_{\text{in}} \times d_{\text{out}}} = 1.0 \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
  $$

对比 H100 的拐点：
$$I_{\text{decode}} = 1.0 \ll 295.2$$

由于 $I_{\text{decode}} \ll I^*$，逐字生成阶段深陷于**显存带宽受限区（Memory-Bandwidth-Bound）**。此时硬件能达到的实际吞吐仅为：
$$
P_{\text{decode}} = I_{\text{decode}} \times B_{\text{peak}} = 1.0 \times 3.35 \times 10^{12} = 3.35 \text{ TFLOPS}
$$

在一张理论算力高达 989 TFLOPS 的 H100 GPU 上，单并发逐字生成的算力利用率仅为：
$$\frac{3.35}{989} \approx 0.34\%！$$
硬件芯片上超过 99% 的算力晶体管完全处于空闲状态，绝大部分时间都在枯等权重参数穿越带宽有限的显存总线！

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2009">2009 年</time> &mdash; <strong>屋顶模型创立</strong>（<cite>Williams, Waterman, &amp; Patterson, 《ACM 通讯》</cite>）</dt>
  <dd>加州大学伯克利分校的 Sam Williams 等人提出了 Roofline 模型，为计算机体系结构提供了一个兼具高度物理直觉与严密数学量化的可视化分析框架。</dd>
  <dt><time datetime="2020">2020 年</time> &mdash; <strong>自回归推理服务危机</strong></dt>
  <dd>随着语言模型参数量从 GPT-2（15 亿）跨越到 GPT-3（1750 亿），工程师们痛苦地发现：训练阶段的算力利用率极高，但单用户交互聊天时，GPU 利用率惨跌至个位数。学术界与工业界深刻意识到，大模型服务必须将并行计算问题拆解为访存流式调度问题。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

我们用一个极简微型神经网络，手算两种阶段的真实访存量与计算量。

### 微型模型配置
- 维度：$d_{\text{in}} = 4, d_{\text{out}} = 4$。
- 权重矩阵 $\mathbf{W} \in \mathbb{R}^{4 \times 4}$（共 16 个参数）。
- 精度：16 位浮点数（每参数 $p = 2$ 字节）。
- 权重占用显存：$16 \times 2 = 32\text{ Bytes}$。

设权重矩阵为：
$$
\mathbf{W} = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 2 & 0 & 1 \\
1 & 1 & 0 & 0 \\
0 & 0 & 2 & 1
\end{bmatrix}
$$

---

### 部分 A：预填充阶段（处理 $T = 3$ 个词元的提示词）

设输入为 3 个词元的嵌入向量矩阵：
$$
\mathbf{X} = \begin{bmatrix}
1 & 0 & 2 & 1 \\
0 & 1 & 1 & 0 \\
2 & 0 & 0 & 1
\end{bmatrix} \in \mathbb{R}^{3 \times 4}
$$

#### 1. 浮点计算量（FLOPs）
计算 $\mathbf{Y} = \mathbf{X} \mathbf{W}$：
- 输出矩阵维度为 $3 \times 4 = 12$ 个元素。
- 每个输出元素经历 4 次乘加运算（$\approx 8$ FLOPs）。
$$\text{总计算量} = 2 \times 3 \times 4 \times 4 = 96 \text{ FLOPs}$$

#### 2. 显存搬运量（Bytes）
芯片从显存中完整读取权重矩阵 $\mathbf{W}$ **一次**：
$$\text{读取字节} = 16 \text{ 个参数} \times 2 \text{ 字节} = 32 \text{ Bytes}$$

#### 3. 算术强度
$$
I_{\text{prefill}} = \frac{96 \text{ FLOPs}}{32 \text{ Bytes}} = 3.0 \text{ FLOPs/Byte}
$$
每个从显存读出的字节，在 3 个提示词并行计算中被复用了 3 次！

---

### 部分 B：逐字解码阶段（生成 1 个新词元）

现在模型必须生成下一个新词元，当前输入退化为单行向量：
$$
\mathbf{x}_4 = \begin{bmatrix} 1 & 1 & 0 & 2 \end{bmatrix} \in \mathbb{R}^{1 \times 4}
$$

#### 1. 浮点计算量（FLOPs）
计算 $\mathbf{y}_4 = \mathbf{x}_4 \mathbf{W}$：
- 输出元素仅为 $1 \times 4 = 4$ 个。
$$\text{总计算量} = 2 \times 1 \times 4 \times 4 = 32 \text{ FLOPs}$$

#### 2. 显存搬运量（Bytes）
为了计算这短短 1 个新词元，芯片仍必须将全部 16 个权重搬进寄存器：
$$\text{读取字节} = 16 \text{ 个参数} \times 2 \text{ 字节} = 32 \text{ Bytes}$$

#### 3. 算术强度
$$
I_{\text{decode}} = \frac{32 \text{ FLOPs}}{32 \text{ Bytes}} = 1.0 \text{ FLOP/Byte}
$$
算术强度直接崩塌至原来的三分之一！

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 20.2:</strong> 玩具模型中预填充与解码阶段的全面定量对比。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">对比指标</th>
      <th align="right">预填充阶段（$T=3$）</th>
      <th align="right">解码阶段（$T=1$）</th>
      <th align="center">比值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>处理词元数</td>
      <td align="right">3</td>
      <td align="right">1</td>
      <td align="center">$3\times$</td>
    </tr>
    <tr>
      <td>总浮点运算量（FLOPs）</td>
      <td align="right">96 FLOPs</td>
      <td align="right">32 FLOPs</td>
      <td align="center">$3\times$</td>
    </tr>
    <tr>
      <td>权重显存搬运量</td>
      <td align="right">32 Bytes</td>
      <td align="right">32 Bytes</td>
      <td align="center"><strong>$1\times$（完全一致！）</strong></td>
    </tr>
    <tr>
      <td><strong>算术强度（$I$）</strong></td>
      <td align="right"><strong>3.0 FLOPs/Byte</strong></td>
      <td align="right"><strong>1.0 FLOP/Byte</strong></td>
      <td align="center"><strong>预填充高出 $3\times$</strong></td>
    </tr>
  </tbody>
</table>

---

## 步骤 6：核心精髓总结

<fieldset>
<legend><strong>核心教学要点</strong></legend>
<p>大语言模型的推理绝非单一任务，而是由屋顶模型性能拐点严格割裂开的两个极端物理世界：</p>
<p><strong>预填充（Prefill）</strong> 是并行的矩阵-矩阵乘（<abbr title="General Matrix-Matrix Multiplication">GEMM</abbr>），它能够充分喂饱 GPU 计算核心，处于算力受限区；而 <strong>解码（Decode）</strong> 是串行的矩阵-向量乘（<abbr title="General Matrix-Vector Multiplication">GEMV</abbr>），它被显存数据搬运速率死死卡住，处于显存带宽受限区。</p>
<p>现代大模型推理服务的所有硬核优化技术（如 KV Cache、连续批处理、推测解码、量化压缩），其本质使命完全相同——通过数学重构与调度创新，跨越解码阶段这道残酷的显存带宽之墙。</p>
</fieldset>
