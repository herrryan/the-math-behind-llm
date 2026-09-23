# 第 27 章：数值压缩的艺术（量化技术：AWQ、SmoothQuant、FP8 与 INT4 数学）

> [!INTUITION] 步骤 1：3 岁小孩直觉
> 想象你要出门远行，行李箱里塞满了 16 磅重的实心铁保龄球。
> 
> 你根本拎不动箱子，托运超重要交高昂罚款，拖着它在机场狂奔更是慢得像蜗牛。
> 
> 此时，一位聪明的玩具工匠走来对你说：
> *“为什么一定要带 16 磅重的沉铁球呢？我们换成只有 4 磅重的轻质木球好不好？它们滚起来的轨迹几乎一模一样！”*
> 
> 你把 16 磅的铁球换成了 4 磅的木球（**INT4 4位量化**）。顷刻间，你的行李箱轻了足足四分之三！你在机场奔跑的速度提升了 4 倍，同样的后备箱能塞下原来 4 倍多的玩具！
> 
> 但你必须格外小心：如果你的行李箱里混进了一顶无比珍贵、吹弹可破的黄金王冠（**异常离群特征 Outlier**），而你盲目地把它也当成粗糙木头粗暴压扁，王冠就会被彻底砸碎。
> 像 **SmoothQuant** 和 **AWQ** 这样的聪明量化算法，就像高超的减震气泡膜：它们精准识别出那 1% 绝不可碰伤的珍贵黄金部件，施加特殊缩放保护；而对于剩余 99% 的普通衣服和木块，则果断压缩到极致的 4 位体积。

---

## 步骤 2：承前启后的关键过渡

我们如何把连续平滑的高精度 16 位浮点数（FP16/BF16），转化为计算机芯片寄存器中紧凑的 8 位（INT8/FP8）或 4 位（INT4）整数，又如何在离散整数网格上正确执行矩阵乘法 $\mathbf{Y} = \mathbf{X} \mathbf{W}$？

在第 20 章中我们已严格证明：大模型的自回归解码阶段深陷在 **显存带宽受限区**：
$$I_{\text{decode}} \approx 1.0 \text{ FLOP/Byte} \ll I^*$$
解码期硬件的耗时，绝大部分都在等待权重参数从显存漫长地爬向计算核心。

权重参数在物理显存中的体积直接由其存储精度决定：
- **FP16 / BF16**：每个参数占 16 位 = 2.0 字节。
- **FP8（Hopper / Blackwell 新特性）**：每个参数占 8 位 = 1.0 字节（显存减半）。
- **INT4（AWQ / GPTQ）**：每个参数占 4 位 = 0.5 字节（**显存压缩 4 倍**！）。

若将一个 70B（700 亿）参数的模型从 FP16（约 140 GB）量化压缩到 INT4（约 35 GB）：
1. 整个模型原本需要 2 到 4 张专业计算卡，现在单张消费级显卡（例如 48GB 或 80GB）即可从容吞下！
2. 每步生成时显存总线需要搬运的字节量直降 4 倍，**使得逐字生成的硬件推理速度直接爆发式提升近 4 倍**！

承前启后的核心过渡问题是：
$$\text{如何将连续实数 } x \in \mathbb{R} \text{ 映射到极窄的离散整数网格 } \{-8, \dots, 7\} \text{ 并将舍入失真降至最低，又如何化解大模型中具有毁灭性破坏力的突发离群激活值（Outliers）？}$$

---

## 步骤 3：严谨数学公式与推导

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        均匀对称 INT4 量化网格映射几何示意                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 连续连续的实数范围（FP16）：                                                           │
│   -x_max ────────────────────────────── 0 ────────────────────────────── +x_max        │
│      ▲                                  ▲                                  ▲           │
│      │ 缩放比例因子 s = x_max / 7       │                                  │           │
│      ▼                                  ▼                                  ▼           │
│ 离散离散的整数格子（INT4）：                                                           │
│     -7    -6    -5    -4    -3   -2   -1    0   +1   +2   +3   +4   +5   +6   +7       │
│   [ 4 位二进制: 1001 ]               [ 0000 ]               [ 4 位二进制: 0111 ]       │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 27.1:</strong> 均匀对称量化通过缩放因子 s，将连续动态区间严格投射到包含 15 个离散槽位的有符号 4 位整数网格中。</figcaption>
</figure>

### 1. 均匀对称量化与反量化数学公式

设目标量化位宽为 $b$ 位有符号整数（例如 $b=4$ 对应区间 $[-7, 7]$，$b=8$ 对应区间 $[-127, 127]$）。

对于连续实数权重向量 $\mathbf{w} \in \mathbb{R}^d$，其 <dfn id="def-scale-zh">量化缩放比例因子（Scale）</dfn> $s$ 定义为：

$$
s = \frac{\max_{i} |w_i|}{2^{b-1} - 1} \in \mathbb{R}^+
$$

<dfn id="def-quant-zh">量化映射函数</dfn> $Q(w)$ 将浮点数投射到最近的离散整数网格点：

$$
q = Q(w) = \operatorname{clip}\left(\left\lfloor \frac{w}{s} \right\rceil, \; -(2^{b-1}-1), \; 2^{b-1}-1\right) \in \mathbb{Z}
$$

其中 $\lfloor \cdot \rceil$ 表示四舍五入取整函数，$\operatorname{clip}(x, a, b)$ 将越界数值截断在安全范围 $[a, b]$ 内。

在计算时，通过 <dfn id="def-dequant-zh">反量化函数（Dequantization）</dfn> 还原浮点近似值：

$$
\hat{w} = \tilde{Q}(q) = s \times q \approx w
$$

该过程的单点绝对量化误差被严格界定在半个网格步长内：$|w - \hat{w}| \le \frac{s}{2}$。

---

### 2. 仅权重量化 vs. 权重-激活全量化

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 27.1:</strong> 主流大模型量化流派分类全景。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">量化流派</th>
      <th align="center">权重存储精度</th>
      <th align="center">激活计算精度</th>
      <th align="left">底层硬件执行机制</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>FP16 基线</strong></td>
      <td align="center">16 位浮点</td>
      <td align="center">16 位浮点</td>
      <td>原生 FP16 张量核心</td>
    </tr>
    <tr>
      <td><strong>W4A16（AWQ / GPTQ）</strong></td>
      <td align="center"><strong>4 位整数（INT4）</strong></td>
      <td align="center">16 位浮点</td>
      <td>显存中存储 4 位权重，载入片上 SRAM 时瞬间动态反量化为 FP16 参与计算</td>
    </tr>
    <tr>
      <td><strong>W8A8（SmoothQuant）</strong></td>
      <td align="center"><strong>8 位整数（INT8）</strong></td>
      <td align="center"><strong>8 位整数（INT8）</strong></td>
      <td>原生 INT8 整数张量核心 GEMM（算力翻倍且显存减半）</td>
    </tr>
    <tr>
      <td><strong>FP8（Hopper / Ada）</strong></td>
      <td align="center"><strong>8 位浮点（E4M3）</strong></td>
      <td align="center"><strong>8 位浮点（E4M3）</strong></td>
      <td>硬件原生 FP8 GEMM（兼具动态范围与双倍吞吐）</td>
    </tr>
  </tbody>
</table>

---

### 3. SmoothQuant：等价数学变换降伏异常离群通道

2022 年，Dettmers 等人发现：当模型规模突破 6.7B 临界点时，会出现惊人的 <dfn id="def-outliers-zh">涌现离群通道（Emergent Outliers）</dfn>。在数千个特征通道中，仅仅 0.1% 的极少数通道其激活值会突然暴涨到常规值的数十倍乃至上百倍（常规通道数值 $\approx 2.0$，离群通道高达 $\approx 150.0$）。
若直接对激活值执行 8 位整数量化，过大的动态范围会将剩下 99.9% 的正常通道全部碾碎为 0，导致语言模型输出乱码胡话。

**SmoothQuant** 创造性地运用了矩阵乘法的对角缩放恒等变换：

$$
\mathbf{Y} = \mathbf{X} \mathbf{W} = \left(\mathbf{X} \operatorname{diag}(\mathbf{s})^{-1}\right) \cdot \left(\operatorname{diag}(\mathbf{s}) \mathbf{W}\right) = \hat{\mathbf{X}} \hat{\mathbf{W}}
$$

其中 $\mathbf{s} \in \mathbb{R}^d$ 为逐通道的平滑对角缩放向量。
为了在激活与权重之间实现完美的难度迁移平衡，每个通道的缩放因子 $s_j$ 严格由几何均值迁移方程决定：

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1 - \alpha}}
$$

超参数 $\alpha \in [0, 1]$（通常取 $\alpha = 0.5$）负责将尖锐突起的离群激活峰值平摊压扁，并平滑迁移至原本数值平坦的权重矩阵中。
SmoothQuant 让千亿参数模型首次全量跑通了 **W8A8 INT8 张量核心矩阵乘**，精度几乎毫无折损！

---

### 4. AWQ：激活感知权重整数量化

Lin 等人（2024）深刻观察到：在 4 位权重量化（W4A16）中，并非所有权重都同等重要——**仅仅只有 1% 的稀疏权重直接影响模型最终的推理表现**，而这些关键权重恰好对应着激活幅度较大的核心通道！

AWQ 通过求解如下加权重构误差极小化问题，找到最优的通道保护缩放矩阵 $\mathbf{S} = \operatorname{diag}(\mathbf{s})$：

$$
\mathbf{W}^* = \arg\min_{\mathbf{W}'} \left\| \mathbf{W} \mathbf{X} - Q(\mathbf{W} \mathbf{S}) \mathbf{S}^{-1} \mathbf{X} \right\|_F^2
$$

通过在执行低位截断前将关键特征维度的权重按比例放大，AWQ 在 4 位精度下牢牢锁住了模型的困惑度（Perplexity），无需昂贵反向传播重训练即可直接部署。

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2022 年">2022 年</time> &mdash; <strong>LLM.int8() 发现涌现离群值</strong>（<cite>Tim Dettmers 等，华盛顿大学</cite>）</dt>
  <dd>首次揭示了模型规模在 67 亿参数附近发生的离群特征突变相变，奠定了大模型量化必须重点处理激活离群通道的理论基石。</dd>
  <dt><time datetime="2023 年">2023 年</time> &mdash; <strong>SmoothQuant 奠定 W8A8 理论</strong>（<cite>Guangxuan Xiao 等，麻省理工学院，ICML 2023</cite>）</dt>
  <dd>通过等价缩放迁移技术，成功打破了激活不能低精度量化的神话，使全整型 INT8 推理在千亿模型上工业化落地。</dd>
  <dt><time datetime="2023–2024 年">2023–2024 年</time> &mdash; <strong>GPTQ 与 AWQ 统一 4 位标准</strong>（<cite>Frantar 等，ICLR 2023；Lin 等，MLSys 2024</cite>）</dt>
  <dd>确立了 W4A16 成为全球开源大模型部署的事实标准，使得原本庞大沉重的 70B 级别顶级模型能够在普通单张个人显卡上飞速运转。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

我们通过手算一个微型 4 维权重向量，完整推导对称 INT4 的量化与反量化全过程。

设原始浮点权重为：
$$
\mathbf{w} = [-6.3, \; 1.8, \; 0.4, \; 7.0]
$$
目标：映射到 $[-7, +7]$ 的 4 位有符号整数离散网格。

---

### 第 1 步：计算量化比例因子 $s$
寻找向量绝对值最大项：
$$
\max |w_i| = \max(|-6.3|, |1.8|, |0.4|, |7.0|) = 7.0
$$
4 位有符号整数最大表示上限为 $2^{4-1} - 1 = 7$。
$$
s = \frac{7.0}{7} = 1.0
$$

---

### 第 2 步：逐元素映射量化（$q = \lfloor w / s \rceil$）
1. 对于 $w_1 = -6.3$：
   $$q_1 = \left\lfloor \frac{-6.3}{1.0} \right\rceil = \lfloor -6.3 \rceil = -6$$
2. 对于 $w_2 = 1.8$：
   $$q_2 = \left\lfloor \frac{1.8}{1.0} \right\rceil = \lfloor 1.8 \rceil = +2$$
3. 对于 $w_3 = 0.4$：
   $$q_3 = \left\lfloor \frac{0.4}{1.0} \right\rceil = \lfloor 0.4 \rceil = 0$$
4. 对于 $w_4 = 7.0$：
   $$q_4 = \left\lfloor \frac{7.0}{1.0} \right\rceil = \lfloor 7.0 \rceil = +7$$

最终存入 GPU 显存的 INT4 整数向量为：
$$
\mathbf{q} = [-6, \; +2, \; 0, \; +7] \in \mathbb{Z}^4
$$
实际存储空间从 64 位（$4 \times 16$）暴跌至 16 位（$4 \times 4$），**显存开销直接缩减了 75%**！

---

### 第 3 步：反量化还原与误差分析
计算反量化还原向量 $\hat{\mathbf{w}} = s \times \mathbf{q}$：
$$
\hat{\mathbf{w}} = 1.0 \times [-6, \; +2, \; 0, \; +7] = [-6.0, \; 2.0, \; 0.0, \; 7.0]
$$

检验单点量化误差 $\mathbf{e} = \mathbf{w} - \hat{\mathbf{w}}$：
- $e_1 = -6.3 - (-6.0) = -0.3$
- $e_2 = 1.8 - 2.0 = -0.2$
- $e_3 = 0.4 - 0.0 = +0.4$
- $e_4 = 7.0 - 7.0 = 0.0$

所有分量的误差值 $|e_i|$ 严格小于半步长 $s/2 = 0.5$！数值特征被高度保真地保留了下来，而显存吞吐需求直接降至原来的四分之一。

---

## 步骤 6：核心精髓总结

<fieldset>
<legend><strong>核心教学要点</strong></legend>
<p>由于自回归生成受制于显存带宽墙，<strong>模型量化（Quantization）</strong> 通过将权重从 16 位压缩至 8 位或 4 位，直接在线性物理意义上将解码时的显存数据流搬运耗时缩减了 2 到 4 倍。</p>
<p>现代前沿的 <strong>SmoothQuant</strong> 与 <strong>AWQ</strong> 算法通过对离群激活通道的精准数学等价平滑与显著性保护，从根本上消除了低比特截断带来的精度雪崩，使千亿级巨型模型得以在消费级硬件上飞速推理。</p>
</fieldset>
