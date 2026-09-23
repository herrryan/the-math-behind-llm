# 第 28 章：分布式推理服务（张量并行与预填-解码物理分离架构）

> [!INTUITION] 步骤 1：3 岁小孩直觉
> 想象一家大饭店的后厨里，有两种性格截然相反的工作人员：
> 
> 1. **切菜大厨（预填充 Prefill 节点）**：
>    身形魁梧、手握重型大砍刀。每次一旦接到新订单，大厨就会在 5 秒钟内雷霆万钧地将 50 斤土豆和胡萝卜瞬间切成整齐的细丝（爆发性高算力需求）。
> 2. **奉茶侍者（逐字解码 Decode 节点）**：
>    步伐轻盈、手托银质茶盘。每隔整整 2 秒钟，侍者优雅地走到餐桌前，给客人的小茶杯里倒上一小勺热茶（持续平稳、低计算密度、高频往返）。
> 
> 在传统的 **一体化混合服务（Monolithic Serving）** 模式下：
> 饭店老板偏偏强迫同一个工作人员，在同一张狭窄的小桌子上边挥刀剁土豆、边给客人倒热茶！结果如何？大砍刀震得桌子乱晃，滚烫的茶水溅了一地；每当大厨开始剁土豆，客人就得干渴地苦等 10 分钟喝不上一口茶；而大厨也被频繁的倒茶打断，心烦意乱差点切到手指！
> 
> 后来，一位聪明的管理专家发明了 **预填-解码分离架构（Disaggregated Serving）**：
> - 把狂野切菜的大厨专门请进后院的重切车间（**专属预填充集群**）；
> - 把倒茶侍者专门安排在宁静雅致的迎宾大厅（**专属解码集群**）；
> - 切好的蔬菜食材，通过一条极速传送带（**超高速 RDMA 内部网络**）在零点零几秒内闪电般滑入前厅！
> 
> 如果一份菜谱实在太大，单个切菜案板根本摆不下怎么办？
> 专家采用了 **张量切分并行（Tensor Parallelism，切拼图法）**：大厨甲切左半张菜谱，大厨乙切右半张菜谱，切完后击个掌（All-Reduce 求和合并），完美上菜！

---

## 步骤 2：承前启后的关键过渡

我们如何将巨型的百亿、千亿权重矩阵在多张 GPU 之间拆分切块并最小化跨卡通信，又如何在集群层级将算力绑定的 Prefill 与显存带宽绑定的 Decode 在物理上拆分隔离？

当运行开源旗舰大模型（例如 Llama-3-405B 或 DeepSeek-V3 671B）时：
- 仅仅存放 FP16 原始模型参数，就需要 **810 GB 到 1.3 TB** 的显存天量空间。
- 当今世界上没有任何单张独立显卡能够独自吃下。

更关键的是，正如第 20 章所揭示，Prefill 与 Decode 对底层硬件资源的索求存在天然的**水火不容**：
- **Prefill（预填充）**：算术强度极高（$I \gg I^*$），深陷算力饱和区，能够彻底喂饱 GPU 的 Tensor Cores，极度渴望多卡并行切片算力。
- **Decode（自回归解码）**：算术强度极低（$I \ll I^*$），深陷显存带宽墙，Tensor Cores 大量闲置，极度渴望超高的 HBM 读取带宽与极低的单步延迟。

将这两个阶段强行糅合在同一张显卡上，会导致长文本 Prefill 随时粗暴打断正在生成的 Decode 序列，引发灾难性的尾部延迟抖动（$P_{99}$ TPOT 暴涨）。

承前启后的核心过渡问题是：
$$\text{Megatron-LM 如何通过共轭的列-行矩阵切分将每层前向传播的跨卡通信压缩至仅仅两次 All-Reduce，而预填-解码解耦架构又如何通过纳秒级 RDMA 网络实现 KV 缓存的物理迁移？}$$

---

## 步骤 3：严谨数学公式与推导

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MEGATRON 张量并行（MLP 前向传播切分流）                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 输入特征激活 X [B x d_model] 广播复制到两张 GPU：                                      │
│                                                                                        │
│   GPU 0（持有 W_1 左半列与 W_2 上半行）：                                              │
│     h_1 = GeLU( X @ W_1,1 )  ──►  y_1 = h_1 @ W_2,1                                    │
│                                          │                                             │
│   GPU 1（持有 W_1 右半列与 W_2 下半行）：│                                             │
│     h_2 = GeLU( X @ W_1,2 )  ──►  y_2 = h_2 @ W_2,2                                    │
│                                          │                                             │
│   NVLink 高速互连 All-Reduce（求和）：   ▼                                             │
│     Y = y_1 + y_2  （数学上严格等于 X @ W_1 @ W_2，中间激活层完全无需跨卡同步！）     │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 28.1:</strong> Megatron-LM 共轭列-行张量并行。中间隐层激活完全保留在各自 GPU 内部，整个 MLP 块仅在末尾执行单次 All-Reduce。</figcaption>
</figure>

### 1. Megatron 共轭列-行矩阵乘并行分解

考虑标准 Transformer 内部的两层前馈神经网络（FFN/MLP）：
$$
\mathbf{Y} = \operatorname{GeLU}(\mathbf{X} \mathbf{W}_1) \mathbf{W}_2
$$
其中输入 $\mathbf{X} \in \mathbb{R}^{B \times d}$，第一层升维权重 $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$，第二层降维权重 $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$。

若要将该计算均摊到 $N$ 张物理 GPU 上：

#### 第一层：列并行 GEMM（Column Parallel）
我们将权重 $\mathbf{W}_1$ 沿着**列方向**（输出维度）竖直切分为 $N$ 份：
$$
\mathbf{W}_1 = \begin{bmatrix} \mathbf{W}_{1}^{(1)} & \mathbf{W}_{1}^{(2)} & \cdots & \mathbf{W}_{1}^{(N)} \end{bmatrix}, \quad \mathbf{W}_{1}^{(i)} \in \mathbb{R}^{d \times \frac{4d}{N}}
$$
每张 GPU $i$ 独立计算局部中间激活表征：
$$
\mathbf{H}^{(i)} = \operatorname{GeLU}\left(\mathbf{X} \mathbf{W}_{1}^{(i)}\right) \in \mathbb{R}^{B \times \frac{4d}{N}}
$$
<mark>关键架构发现：</mark> 由于 GeLU 激活函数是严格的逐元素独立非线性映射：
$$
\operatorname{GeLU}\left(\begin{bmatrix} \mathbf{A} & \mathbf{B} \end{bmatrix}\right) = \begin{bmatrix} \operatorname{GeLU}(\mathbf{A}) & \operatorname{GeLU}(\mathbf{B}) \end{bmatrix}
$$
**在第一层计算完毕后，GPU 之间完全不需要发生任何网络通信！** 各张卡将切片结果妥善保留在本地片上寄存器即可。

---

#### 第二层：行并行 GEMM（Row Parallel）
我们将第二层权重 $\mathbf{W}_2$ 沿着**行方向**（输入维度）横向切分为 $N$ 份：
$$
\mathbf{W}_2 = \begin{bmatrix} \mathbf{W}_{2}^{(1)} \\ \mathbf{W}_{2}^{(2)} \\ \vdots \\ \mathbf{W}_{2}^{(N)} \end{bmatrix}, \quad \mathbf{W}_{2}^{(i)} \in \mathbb{R}^{\frac{4d}{N} \times d}
$$
每张 GPU $i$ 用自己手头的中间切片乘上手头的行权重切片：
$$
\mathbf{Y}^{(i)} = \mathbf{H}^{(i)} \mathbf{W}_{2}^{(i)} \in \mathbb{R}^{B \times d}
$$

将所有 GPU 的局部部分和相加，即可严格还原全局矩阵乘：
$$
\mathbf{Y} = \sum_{i=1}^N \mathbf{Y}^{(i)} = \sum_{i=1}^N \operatorname{GeLU}\left(\mathbf{X} \mathbf{W}_{1}^{(i)}\right) \mathbf{W}_{2}^{(i)} = \operatorname{GeLU}(\mathbf{X} \mathbf{W}_1) \mathbf{W}_2
$$

所有 GPU 通过片间高速 NVLink 总线调用一次底层的硬件 <dfn id="def-allreduce-zh">All-Reduce（Sum 规约求和）</dfn> 算子。

---

### 2. 多头自注意力块切分与通信开销

对于 Self-Attention 自注意力层：
- $W_Q, W_K, W_V$ 采取**列切分**（将注意力头数 $H$ 均分到各卡）；
- 输出投影矩阵 $W_O$ 采取**行切分**；
- 在残差相加前执行单次 All-Reduce。

整套 Transformer 基础层（Attention + MLP）在前向传播中**总共仅需触发 2 次 All-Reduce 通信**。
对于包含 $N$ 张 GPU、张量尺寸为 $[B, T, d]$、存储精度为 $p$ 字节的环形 All-Reduce：

$$
\text{单层通信数据量} = 2 \times \left(2 \cdot \frac{N - 1}{N} \cdot B \cdot T \cdot d \cdot p\right) \text{ 字节}
$$

由于每层都必须发生同步阻塞，张量并行（TP）通常被严格限制在拥有超高带宽 NVLink（H100 达 $900\text{ GB/s}$）的单机 8 卡节点内展开。

---

### 3. 预填-解码物理分离架构（PD Separation）

在现代海量并发工业架构（DistServe、Mooncake、Splitwise）中，系统在物理机群层级被切分为两组专精集群：

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        预填-解码物理分离服务拓扑（PD DISAGGREGATION）                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 用户请求 ──► [ 全局智能负载网关 Global Router ]                                        │
│                     │                                                                  │
│                     ▼                                                                  │
│ ┌──────────────────────────────────────────┐                                           │
│ │ 预填 Prefill 节点集群（算力密集型工场）： │                                           │
│ │   - 专精算力吞吐（GEMM 密集阵列）         │                                           │
│ │   - 大张量并行度（TP=4 或 TP=8）         │                                           │
│ │   - 巨型批处理并发，瞬间拉满 Tensor Cores │                                           │
│ └──────────────────────────────────────────┘                                           │
│                     │                                                                  │
│                     ▼  超低时延 RDMA 网络（RoCE v2 / InfiniBand 高速总线）             │
│   KV CACHE 跨网传输 ──► [ 零拷贝直通：将计算出的键值缓存高速注入解码节点显存 ]          │
│                     │                                                                  │
│                     ▼                                                                  │
│ ┌──────────────────────────────────────────┐                                           │
│ │ 解码 Decode 节点集群（显存带宽型客栈）： │                                           │
│ │   - 专精显存带宽（GEMV 极致利用）        │                                           │
│ │   - 小张量并行度（TP=1 或 TP=2，低同步） │                                           │
│ │   - 丝滑逐字吐词，彻底杜绝尾部卡顿       │                                           │
│ └──────────────────────────────────────────┘                                           │
│                     │                                                                  │
│                     ▼                                                                  │
│             流式输出用户答案（稳定低时延）                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 28.2:</strong> 预填-解码物理分离拓扑。将算力受限的 Prefill 节点与显存带宽受限的 Decode 节点物理物理隔离，通过 RDMA 高速搬运 KV 缓存。</figcaption>
</figure>

设历史上下文长度为 $T_{\text{ctx}}$，模型拥有 $L$ 层，分组注意力 KV 头数为 $H_{\text{kv}}$，单头维度为 $d$。
预填完成后需要在网络上传输的 KV Cache 总大小为：
$$
\text{Size}_{\text{KV}} = 2 \times L \times H_{\text{kv}} \times d \times T_{\text{ctx}} \times p \text{ 字节}
$$

在带宽为 $B_{\text{RDMA}}$（例如 $400 \text{ Gbps} = 50 \text{ GB/s}$）的高速集群网络上，网络传输纯耗时为：

$$
t_{\text{transfer}} = \frac{\text{Size}_{\text{KV}}}{B_{\text{RDMA}}}
$$

以一个拥有 4,000 词元 Prompt 的 70B 模型（GQA $H_{\text{kv}} = 8, d=128, L=80, p=2$）为例，其 $\text{Size}_{\text{KV}} \approx 655 \text{ MB}$。
$$
t_{\text{transfer}} = \frac{0.655 \text{ GB}}{50 \text{ GB/s}} \approx 13 \text{ 毫秒}
$$

仅仅付出了区区 13 毫秒的网络直传代价，便彻底换来了**解码节点 100% 不受外部大 Prompt 冲击干扰**的宁静环境，将长尾解码延迟波动（$P_{99}$ TPOT）生生降低了整整 **10 倍以上**！

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2019 年">2019 年</time> &mdash; <strong>Megatron-LM 张量并行奠基</strong>（<cite>Mohammad Shoeybi 等，NVIDIA</cite>）</dt>
  <dd>首创共轭列-行矩阵乘法分解，将 Transformer 层内跨卡通信次数极限压缩至 2 次 All-Reduce，成为现代一切超大模型多卡训练与推理的黄金基石。</dd>
  <dt><time datetime="2024 年">2024 年</time> &mdash; <strong>DistServe 与 Splitwise 提出服务解耦</strong>（<cite>Hao Zhong 等，OSDI 2024；Pratyush Patel 等，ISCA 2024</cite>）</dt>
  <dd>从排队论角度深刻证明了同卡混跑预填与解码必然引发严重的排队阻塞，在学术界确立了物理分离服务的理论范式。</dd>
  <dt><time datetime="2024 年">2024 年</time> &mdash; <strong>Mooncake（Kimi 架构）工业化验证</strong>（<cite>Qin 等，月之暗面 Moonshot AI</cite>）</dt>
  <dd>在超长文本（百万级上下文）生产环境中大规模部署了预填-解码分离系统，利用自研高速 KV 缓存池与分块传输，支撑起了数千万人并发的超长推理服务。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

我们以一个 $2 \times 2$ 的极简矩阵乘法，手算推导 2 张 GPU 协同执行 Megatron 张量并行的全流程。

### 输入向量与网络权重
设单个输入词元特征向量为（$B=1, d=2$）：
$$
\mathbf{x} = [1.0, \; 2.0]
$$

两层前馈网络的权重矩阵分别为：
$$
\mathbf{W}_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$
（为简化手算演示，激活函数设为恒等映射 $\sigma(z) = z$）。

---

### 单卡单机标准计算（真值对照）
$$
\mathbf{h} = \mathbf{x} \mathbf{W}_1 = [1, 2] \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = [1(1) + 2(3), \; 1(2) + 2(4)] = [7, \; 10]
$$
$$
\mathbf{y} = \mathbf{h} \mathbf{W}_2 = [7, 10] \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = [7(5) + 10(7), \; 7(6) + 10(8)] = [35 + 70, \; 42 + 80] = [\mathbf{105}, \; \mathbf{122}]
$$

---

### 2 张 GPU 张量并行执行轨迹

#### 第 1 步：$W_1$ 列并行切分
- GPU 0 分得第 1 列：$\mathbf{W}_{1,1} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$
- GPU 1 分得第 2 列：$\mathbf{W}_{1,2} = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$

每张 GPU 独立用全局广播的 $\mathbf{x} = [1, 2]$ 执行本地相乘：
- **GPU 0 本地计算**：$h_1 = [1, 2] \begin{bmatrix} 1 \\ 3 \end{bmatrix} = 1(1) + 2(3) = 7$
- **GPU 1 本地计算**：$h_2 = [1, 2] \begin{bmatrix} 2 \\ 4 \end{bmatrix} = 1(2) + 2(4) = 10$

两张卡独立得到了 $[h_1, h_2] = [7, 10]$。**期间发生了 0 次网络通信！**

---

#### 第 2 步：$W_2$ 行并行切分
- GPU 0 分得第 1 行：$\mathbf{W}_{2,1} = \begin{bmatrix} 5 & 6 \end{bmatrix}$
- GPU 1 分得第 2 行：$\mathbf{W}_{2,2} = \begin{bmatrix} 7 & 8 \end{bmatrix}$

每张 GPU 用本地的标量 $h_i$ 乘以本地的行权重向量：
- **GPU 0 本地输出**：$\mathbf{y}_1 = 7 \times [5, 6] = [35, \; 42]$
- **GPU 1 本地输出**：$\mathbf{y}_2 = 10 \times [7, 8] = [70, \; 80]$

---

#### 第 3 步：硬件级 All-Reduce 求和
两张 GPU 在 NVLink 总线上将各自的 $\mathbf{y}_i$ 执行规约加法：
$$
\mathbf{y} = \mathbf{y}_1 + \mathbf{y}_2 = [35, 42] + [70, 80] = [35 + 70, \; 42 + 80] = [\mathbf{105}, \; \mathbf{122}]
$$

计算结果与单卡完全一致，**分毫不差**，而在整个过程中，巨型权重矩阵被完整切分，跨卡通信被压缩到了理论极值！

---

## 步骤 6：核心精髓总结

<fieldset>
<legend><strong>核心教学要点</strong></legend>
<p>在单机节点内部，<strong>Megatron 张量并行（Tensor Parallelism）</strong> 利用共轭的“列切分-行切分”代数分解，使庞大的注意力与前馈矩阵得以在多卡间并行切分，将跨卡同步压缩至每层仅需两次 All-Reduce。</p>
<p>在数据中心集群层级，<strong>预填-解码分离架构（PD Disaggregation）</strong> 彻底化解了推理两阶段的硬件对抗，将计算密集型 Prompt 与带宽受限型 Decode 物理隔离，通过纳秒级 RDMA 网络搬运 KV 缓存，构筑了现代万亿参数大模型服务的高可靠护城河。</p>
</fieldset>
