# 第 10 章：我在句子的哪里？（位置编码与 RoPE 旋转位置嵌入）


## 第 1 步：3 岁小孩直觉（运动员号码布与旋转表盘） {: #step-1 }

!!! note "3岁小孩的直觉: 被剪碎的照片与飞速旋转的表盘齿轮"
    想象你是一名体育摄影师，正在操场上给幼儿园的跑步比赛拍照。

    十个活力四射的小朋友在草地跑道上奋力向终点冲刺。比赛结束后，你把这张充满欢声笑语的照片洗了出来。

    但是，粗心的裁判在比赛前忘记给小朋友们贴**跑步号码布**了！

    当你端详这张照片时，你看到十个奔跑的小朋友并排在一起。可是因为谁身上都没有号码布，草地上也没有划出第几跑道的白线，一个让人抓狂的麻烦出现了：

    如果你用剪刀把照片里的每一个小朋友分别剪成独立的小纸片，然后像洗扑克牌一样在桌上随便打乱重排，整张画面看起来依然完全合理！你根本无法分辨到底谁跑在第 1 位、谁是第 2 位、谁又是最后一个起跑的。

    在人类的语言里，词语的先后顺序决定了一切：

    - *“小狗咬人”* 是一起让人心疼的普通意外。
    - *“人咬小狗”* 则是令人瞠目结舌的爆炸新闻！

    然而，对于只懂得把所有词元扔进矩阵里算点积的计算机来说，这两句话包含的词汇完全一模一样。如果没有位置标签，计算机就像把所有词倒进了果汁机，根本分不清到底是谁咬了谁！

    为了解决这个难题，聪明的裁判给每一个跑者发了一枚带有**旋转表盘**的魔法徽章：

    1. **不同速度的表盘指针**：徽章上有好几根指针。第一根针转得飞快（你每跑一步，它就跳动一格）；第二根针转得不紧不慢；第三根针则像笨重的大钟表时针，跑很久才挪动一小步。
    2. **每一步都有独一无二的角度**：在第 1 步时，指针指向某个方向；跑到第 2 步时，指针扭转了一个角度；跑到第 5 步时，它转得更远了。整条跑道上没有任何两步的指针角度会完全撞车。
    3. **真正神奇的是“相对角度差”**：跑者 A 在第 5 步，跑者 B 在第 2 步。不管现在是清晨还是傍晚，你只要量一量他们两人表盘之间的夹角，就立刻知道：他们刚好相差了整整 3 步的距离！

    在现代大语言模型中，这种像旋转表盘一样的神奇机制就叫做 **RoPE（Rotary Position Embedding，旋转位置嵌入）**。它不往词语原本的意思里硬塞杂乱的数值，而是通过把向量的方向指针旋转一个与位置精确对应的角度，优雅地刻下了语序印记！

<figure>
<pre>
语序混淆危机（朴素注意力机制完全丧失语序感知）：

句子 A: [小狗] [追逐] [花猫]  ──► 注意力词袋: {小狗, 追逐, 花猫}
句子 B: [花猫] [追逐] [小狗]  ──► 注意力词袋: {花猫, 追逐, 小狗}
结果：点积计算结果完全一模一样！模型沦为“语义色盲”，分不清谁追谁。

RoPE 旋转方案（每个词元根据自身位置 m 旋转表盘）：

第 m=1 步: [小狗] ──► 表盘旋转 1 x θ: ──► [指针指向 28°]
第 m=2 步: [追逐] ──► 表盘旋转 2 x θ: ──► [指针指向 57°]
第 m=3 步: [花猫] ──► 表盘旋转 3 x θ: ──► [指针指向 86°]

相对距离守恒："小狗" (m=1) 与 "花猫" (m=3) 的夹角 = 3θ - 1θ = 2θ（永远代表相隔 2 步！）
</pre>
<figcaption><strong>图 10.1：</strong> RoPE 将词元的绝对位置转化为表盘上的旋转角度，利用两者的夹角差精确锁定相对距离。</figcaption>
</figure>

---

## 第 2 步：承前启后的关键过渡 {: #step-2 }

!!! question "计算连接问题: 为什么朴素注意力具有“置换等价性”？"
    在第 06 章与第 08 章中，我们深入推导了注意力机制通过点积衡量匹配度的过程：



    $$
    S_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
    $$



    现在，假设我们用一个置换矩阵 $\mathbf{P} \in \{0, 1\}^{T \times T}$ 随意打乱输入词元的顺序（例如把原本的第 1 个词和第 3 个词对调）。由于投影矩阵 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ 是对每一行独立施加线性变换的：



    $$
    \mathbf{Q}_{\text{perm}} = \mathbf{P}\mathbf{Q}, \quad \mathbf{K}_{\text{perm}} = \mathbf{P}\mathbf{K}, \quad \mathbf{V}_{\text{perm}} = \mathbf{P}\mathbf{V}
    $$



    当我们计算点积打分矩阵时：



    $$
    \mathbf{Q}_{\text{perm}} \mathbf{K}_{\text{perm}}^\top = (\mathbf{P}\mathbf{Q})(\mathbf{P}\mathbf{K})^\top = \mathbf{P}\mathbf{Q}\mathbf{K}^\top \mathbf{P}^\top
    $$



    经过 Softmax 并乘以 $\mathbf{V}_{\text{perm}}$ 后：



    $$
    \operatorname{Attention}(\mathbf{Q}_{\text{perm}}, \mathbf{K}_{\text{perm}}, \mathbf{V}_{\text{perm}}) = \mathbf{P} \cdot \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
    $$



    这一数学性质在代数上被称为<dfn id="def-permutation-equivariance-zh">置换等价性（Permutation Equivariance）</dfn>。

    它的物理含义是：如果你把输入的词语随意打乱，模型算出来的输出向量也会原封不动地跟着打乱——**内部的任何交互强度都没有发生任何实质性改变**！模型天生把一个句子看作一个毫无先后顺序的“无序词袋（Bag of Words）”。

    最早的 Transformer（Vaswani 等人，2017）试图通过直接在词嵌入向量上叠加静态绝对位置向量来解决这个问题：



    $$
    \mathbf{x}_m \leftarrow \mathbf{x}_m + \mathbf{p}_m
    $$



    但这种加法方案存在两大致命缺陷：
    1. **语义空间被污染**：把巨大的位置数字强行加进语义特征里，就像在纯净的墨水里倒进了泥沙，破坏了词义本身的几何距离。
    2. **死板的绝对位置，缺乏相对感知**：模型学会了“第 500 个位置”的长相，却很难自动推导出“A 和 B 距离相隔 3 步”这一具有平移不变性的相对规律。

    *“我们究竟如何在数学上为词元烙印上位置信息，使得任意两个词元之间的点积打分严格、纯粹地只取决于它们的相对距离 $(m - n)$，同时完全不向语义向量里注入嘈杂的加性噪音？”*

---

## 第 3 步：严谨数学公式与架构推导 {: #step-3 }

### 1. 经典起点：正弦余弦绝对位置编码（Sinusoidal PE）

在开山之作《Attention Is All You Need》（Vaswani 等人，2017）中，作者使用了固定周期的正弦与余弦函数。

对于位于位置 $m \in \{0, 1, \dots, T-1\}$ 的词元，在其特征向量的偶数维度与奇数维度分别定义：



$$
\begin{aligned}
\text{PE}_{(m, 2i)} &= \sin\left(\frac{m}{10000^{2i/d}}\right) \\
\text{PE}_{(m, 2i+1)} &= \cos\left(\frac{m}{10000^{2i/d}}\right)
\end{aligned}
$$



其中 $i \in \{0, 1, \dots, \frac{d}{2}-1\}$ 为维度通道索引。令角频率标量为 $\theta_i = \frac{1}{10000^{2i/d}}$，则位置 $m$ 的编码向量为：



$$
\mathbf{p}_m = \begin{bmatrix}
\sin(m\theta_0) \\
\cos(m\theta_0) \\
\sin(m\theta_1) \\
\cos(m\theta_1) \\
\vdots \\
\sin(m\theta_{d/2 - 1}) \\
\cos(m\theta_{d/2 - 1})
\end{bmatrix} \in \mathbb{R}^d
$$



波长形成了一个从 $2\pi$ 到 $2\pi \cdot 10000$ 的几何级数：
- 低维度通道（$i = 0$）：$\theta_0 = 1$，波长约为 $6.28$ 个词元（极速跳动的秒针）。
- 高维度通道（$i = \frac{d}{2}-1$）：波长长达数万个词元（缓慢运转的年历盘）。

尽管设计十分精妙，但它采用了加法注入：$\widetilde{\mathbf{x}}_m = \mathbf{x}_m + \mathbf{p}_m$。在后续计算点积 $(\mathbf{x}_m + \mathbf{p}_m)^\top (\mathbf{x}_n + \mathbf{p}_n)$ 时，交叉项 $\mathbf{x}_m^\top \mathbf{p}_n$ 与 $\mathbf{p}_m^\top \mathbf{x}_n$ 产生了不可控的内容与位置相互污染。

---

### 2. 现代大模型基石：RoPE 旋转位置嵌入

中国学者苏剑林（2021）提出的 <dfn id="def-rope-zh">RoPE（Rotary Position Embedding）</dfn> 彻底摒弃了向量加法，转而在 2D 坐标子空间中通过**正交旋转变换**将位置注入 Query 和 Key。

#### 2D 旋转基础模块

对于一个二维向量 $\mathbf{v} = [v_1, v_2]^\top$，将其逆时针旋转角度 $m\theta$ 的线性变换对应着标准的 2D 旋转矩阵：



$$
\mathbf{R}_{\theta, m} = \begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$



#### 完整的 $d$ 维高维旋转算子

对于特征维度为 $d$ 的 Query 向量 $\mathbf{q}_m \in \mathbb{R}^d$ 与 Key 向量 $\mathbf{k}_n \in \mathbb{R}^d$（其中 $d$ 为偶数，如 $d = 64$ 或 $d = 128$），RoPE 将这 $d$ 个维度两两成对拆解为 $\frac{d}{2}$ 个独立的二维平面：



$$
(q^{(1)}, q^{(2)}), \quad (q^{(3)}, q^{(4)}), \quad \dots, \quad (q^{(d-1)}, q^{(d)})
$$



每个二维分量分别施加由对应通道频率决定的旋转角 $m\theta_i$：



$$
\mathbf{R}_{\Theta, m}^d = \begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \dots & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \dots & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \dots & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \dots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{bmatrix} \in \mathbb{R}^{d \times d}
$$



其中各通道的基础角频率与经典正弦编码完全一致：



$$
\theta_i = 10000^{-2(i-1)/d}, \quad i \in \left\{1, 2, \dots, \frac{d}{2}\right\}
$$



经过 RoPE 旋转后的 Query 和 Key 向量分别写作：



$$
\widetilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m}^d \mathbf{q}_m, \quad \widetilde{\mathbf{k}}_n = \mathbf{R}_{\Theta, n}^d \mathbf{k}_n
$$



**极其关键的细节**：Value 矩阵 $\mathbf{V}$ **完全不需要施加任何旋转**！Value 向量代表信息载荷本体，而非几何检索坐标。

<details>
<summary><strong>数学符号速查与张量定义清单</strong></summary>
<dl>
  <dt><strong>$m, n \in \mathbb{N}$</strong></dt>
  <dd>词元在序列中的绝对时间步坐标：$m$ 为 Query 发起位置，$n$ 为 Key 被检索位置（$0 \le m, n \lt T$）。</dd>
  <dt><strong>$d$（头特征维度 Head Dimension）</strong></dt>
  <dd>注意力头的特征向量维度（必须为偶数，通常为 $d = 64$ 或 $d = 128$）。</dd>
  <dt><strong>$\theta_i$（第 $i$ 组二维平面的基础角频率）</strong></dt>
  <dd>空间旋转角速度：$\theta_i = b^{-2(i-1)/d}$，其中底数基频 $b = 10000$（在长文本模型中常提升至 $500000$ 以上）。</dd>
  <dt><strong>$\mathbf{R}_{\theta_i, m} \in \mathbb{R}^{2 \times 2}$</strong></dt>
  <dd>二维标准正交旋转矩阵：满足 $\mathbf{R}_{\theta_i, m}^\top \mathbf{R}_{\theta_i, m} = \mathbf{I}_2$，行列式 $\det(\mathbf{R}) = 1$。</dd>
  <dt><strong>$\mathbf{R}_{\Theta, m}^d \in \mathbb{R}^{d \times d}$</strong></dt>
  <dd>由 $\frac{d}{2}$ 个二维旋转块构成的分块对角正交旋转矩阵。</dd>
  <dt><strong>$\widetilde{\mathbf{q}}_m, \widetilde{\mathbf{k}}_n \in \mathbb{R}^d$</strong></dt>
  <dd>完成位置角度旋转后的位置敏感型 Query 与 Key 向量。</dd>
</dl>
</details>

---

### 3. 核心严谨证明：为什么 RoPE 严格保证相对距离不变性？

为什么 RoPE 能一统现代开源大模型？核心原因在于它从纯代数上严格保证了：**Query 与 Key 的内积严格纯粹地只取决于它们的相对距离 $(m - n)$**。

以下是完整的代数推导过程：

#### 第一步：二维旋转矩阵的转置与求逆
二维旋转矩阵具有标准正交性，其转置矩阵就等于其逆矩阵，相当于反向旋转同样的角度：



$$
\mathbf{R}_\alpha^\top = \begin{bmatrix}
\cos(\alpha) & \sin(\alpha) \\
-\sin(\alpha) & \cos(\alpha)
\end{bmatrix} = \begin{bmatrix}
\cos(-\alpha) & -\sin(-\alpha) \\
\sin(-\alpha) & \cos(-\alpha)
\end{bmatrix} = \mathbf{R}_{-\alpha}
$$



#### 第二步：两个旋转矩阵的连乘性质
两个同平面的旋转矩阵相乘，等于旋转角度的直接代数相加：



$$
\mathbf{R}_\alpha \mathbf{R}_\beta = \mathbf{R}_{\alpha + \beta}
$$



将第一步的转置性质与连乘性质结合：



$$
\mathbf{R}_{\theta, m}^\top \mathbf{R}_{\theta, n} = \mathbf{R}_{\theta, -m} \mathbf{R}_{\theta, n} = \mathbf{R}_{\theta, (n - m)}
$$



#### 第三步：注意力打分内积的代数闭包
现在计算旋转后的向量内积：



$$
\begin{aligned}
\langle \widetilde{\mathbf{q}}_m, \widetilde{\mathbf{k}}_n \rangle &= \widetilde{\mathbf{q}}_m^\top \widetilde{\mathbf{k}}_n \\
&= (\mathbf{R}_{\Theta, m}^d \mathbf{q}_m)^\top (\mathbf{R}_{\Theta, n}^d \mathbf{k}_n) \\
&= \mathbf{q}_m^\top \left( (\mathbf{R}_{\Theta, m}^d)^\top \mathbf{R}_{\Theta, n}^d \right) \mathbf{k}_n
\end{aligned}
$$



由于高维旋转矩阵 $\mathbf{R}_{\Theta, m}^d$ 是分块对角的，每一个二维子平面之间互不干扰，直接应用第二步的性质：



$$
(\mathbf{R}_{\Theta, m}^d)^\top \mathbf{R}_{\Theta, n}^d = \mathbf{R}_{\Theta, (n - m)}^d
$$



回代整理，便得到了令整个 AI 界赞叹的黄金等式：



$$
\langle \widetilde{\mathbf{q}}_m, \widetilde{\mathbf{k}}_n \rangle = \mathbf{q}_m^\top \mathbf{R}_{\Theta, (n - m)}^d \mathbf{k}_n
$$



请凝视等式右侧：
- 绝对位置 $m$ 与 $n$ 已经彻底消失！
- 表达式中仅存两者的**相对距离差 $(n - m)$**！
- 如果词元 A 在位置 100，词元 B 在位置 103，距离为 $103 - 100 = 3$；如果整句话平移到位置 5000 和 5003，距离仍然是 $5003 - 5000 = 3$。它们之间的点积打分**绝对丝毫不差**！

---

### 4. 复数欧拉公式视角（极度优雅的一行证明）

在复变函数中，在二维平面上将一个点 $(x, y)$ 旋转 $\phi$ 角，完全等同于给复数 $z = x + i y$ 乘上单位复数 $e^{i\phi}$：



$$
z \cdot e^{i\phi} = (x + iy)(\cos\phi + i\sin\phi) = (x\cos\phi - y\sin\phi) + i(x\sin\phi + y\cos\phi)
$$



如果将第 $k$ 个二维平面视为复数 $q_{(k)} \in \mathbb{C}$ 和 $k_{(k)} \in \mathbb{C}$：



$$
\widetilde{q}_{(k), m} = q_{(k)} e^{i m \theta_k}, \quad \widetilde{k}_{(k), n} = k_{(k)} e^{i n \theta_k}
$$



两个二维实数向量的点积，完全等于两个对应复数在共轭相乘后的实部 $\operatorname{Re}(u v^*)$：



$$
\begin{aligned}
\operatorname{Re}\left( \widetilde{q}_{(k), m} \cdot \widetilde{k}_{(k), n}^* \right) &= \operatorname{Re}\left( (q_{(k)} e^{i m \theta_k}) \cdot (k_{(k)} e^{i n \theta_k})^* \right) \\
&= \operatorname{Re}\left( q_{(k)} k_{(k)}^* \cdot e^{i m \theta_k} e^{-i n \theta_k} \right) \\
&= \operatorname{Re}\left( q_{(k)} k_{(k)}^* \cdot e^{i (m - n) \theta_k} \right)
\end{aligned}
$$



仅仅通过高中的复数乘法指数运算法则，相对位置不变性就被瞬间证毕！

---

### 5. 显卡工程落地：零显存分配的高速逐元素实现

在 PyTorch 或 CUDA 底层代码中，我们**绝对不会**在显存里真实开辟那个巨大的 $d \times d$ 稀疏矩阵 $\mathbf{R}_{\Theta, m}^d$，否则会白白浪费海量显存与带宽。

事实上，观察二维旋转公式的展开式：



$$
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
= \begin{bmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_1 \sin(m\theta) + x_2 \cos(m\theta) \end{bmatrix}
= \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \cos(m\theta) + \begin{bmatrix} -x_2 \\ x_1 \end{bmatrix} \sin(m\theta)
$$



对于一整根向量 $\mathbf{x} = [x_1, x_2, x_3, x_4, \dots, x_{d-1}, x_d]^\top$，我们定义偶奇交叉翻转向量 $\mathbf{x}_{\text{rot}}$：



$$
\mathbf{x}_{\text{rot}} = [-x_2, x_1, -x_4, x_3, \dots, -x_d, x_{d-1}]^\top
$$



则整个 RoPE 变换可以直接通过两行极速的逐元素（Element-wise）乘加搞定：



$$
\mathbf{R}_{\Theta, m}^d \mathbf{x} = \mathbf{x} \odot \cos(\mathbf{m}\Theta) + \mathbf{x}_{\text{rot}} \odot \sin(\mathbf{m}\Theta)
$$



这不仅彻底消除了矩阵乘法开销，还达到了 GPU 显存吞吐带宽的理论极限！

---

## 第 4 步：历史渊源与技术演进 {: #step-4 }

<figure>
<pre>
自然语言处理位置表征技术的演进简史：

1950s-2010s: 词袋模型 / 朴素注意力 ──► 无位置信息。天然具备置换等价性。
                                         "小狗追猫" == "猫追小狗"
      │
      ▼
2017年: Vaswani 等人（Transformer） ──► 正弦绝对位置编码：x + PE(pos)
                                         固定多频谐波；向语义向量里注入加性噪音。
      │
      ▼
2018-2019年: 可学习绝对位置嵌入（BERT, GPT-2） ──► 嵌入查找表：W_pos ∈ R^{T_max x d}
                                                   存在硬编码上限！无法外推超越 T_max 的长文本。
      │
      ▼
2018-2020年: 相对注意力偏置（Shaw, T5） ────────► 注意力打分偏置：S_{ij} + b_{i-j}
                                                   需要维护 O(T^2) 的巨大偏置矩阵，拖慢 FlashAttention。
      │
      ▼
2021年: 苏剑林（RoFormer） ──────────────────────► 旋转位置嵌入（RoPE）
                                                   乘性 2D 正交旋转作用于 Q、K；
                                                   相对距离严格不变，零显存开销，完美融合 FlashAttention；
                                                   成为 LLaMA、Mistral、Qwen、DeepSeek 的工业绝对标配。
</pre>
<figcaption><strong>图 10.2：</strong> 从无序词袋到乘性旋转几何的代际技术演化。</figcaption>
</figure>

### 1. 探索相对位置的曲折长路

为了理解 RoPE 为何能统治当代大模型，我们需要剖析前几代方案各自踩过的巨坑：

<fieldset>
<legend><strong>四代位置表征方案的优势与致命软肋</strong></legend>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 10.1：</strong> 语言模型主流位置编码机制横向对比。</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">编码方案</th>
      <th align="center">数学公式</th>
      <th align="left">代表模型</th>
      <th align="left">核心缺陷与技术瓶颈</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>正弦绝对编码</strong></td>
      <td align="center">$\mathbf{x}_m + \mathbf{p}_m$</td>
      <td>Transformer (2017)</td>
      <td>
        <del>加性噪音污染！</del> 强行将坐标加进语义嵌入。展开点积时产生非预期的内容-位置交叉耦合项。
      </td>
    </tr>
    <tr>
      <td><strong>可学习绝对嵌入</strong></td>
      <td align="center">$\mathbf{x}_m + \mathbf{W}_{\text{pos}}[m]$</td>
      <td>BERT (2018), GPT-2 (2019)</td>
      <td>
        <del>硬编码长度天花板！</del> 若查找表最大为 $T_{\max} = 2048$，则第 $2049$ 步无参数可用，模型完全无法外推更长文本。
      </td>
    </tr>
    <tr>
      <td><strong>相对偏置（T5 / Shaw）</strong></td>
      <td align="center">$S_{ij} + b_{i-j}$</td>
      <td>Shaw (2018), T5 (2020)</td>
      <td>
        <del>显存带宽黑洞！</del> 必须在注意力打分上叠加一个 $T \times T$ 的全局偏置张量，与片上高速融合算子（FlashAttention）水火不容。
      </td>
    </tr>
    <tr bgcolor="#fdfdf0">
      <td><strong>旋转位置嵌入（RoPE）</strong></td>
      <td align="center">$\mathbf{R}_m \mathbf{q}_m, \, \mathbf{R}_n \mathbf{k}_n$</td>
      <td>LLaMA, Mistral, Qwen, DeepSeek</td>
      <td>
        <ins><strong>现代 AI 的黄金标准！</strong></ins> 相对距离严格等价于旋转差 $\mathbf{R}_{n-m}$，零显存查找表，不干扰 Value，天然支持 128k 超长文本外推！
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

### 2. 苏剑林的代数洞见（2021）

2021 年，中国独立 AI 研究者**苏剑林**在他的学术博客《追问旋转位置编码》中提出了一个直击灵魂的数学猜想：

> *“我们能否找到一对变换函数 $f_Q(\mathbf{x}_m, m)$ 和 $f_K(\mathbf{x}_n, n)$，使得它们的内积 $\langle f_Q(\mathbf{x}_m, m), f_K(\mathbf{x}_n, n) \rangle$ 恒等于一个只包含相对差值 $(m - n)$ 的函数 $g(\mathbf{x}_m, \mathbf{x}_n, m - n)$？”*

苏剑林通过微分几何与泛函方程推导证明：在实数坐标系与基本连续性假设下，这个方程的**唯一非平凡解**就是二维分块正交旋转矩阵 $\mathbf{R}_{\Theta, m}^d$。

该成果发表于学术论文 *RoFormer: Enhanced Transformer with Rotary Position Embedding*（Su et al., 2021）。2023 年初，Meta 团队在训练初代 LLaMA 时全面采用 RoPE，随后它如野火燎原般成为了几乎所有现代主流大语言模型不可动摇的标准配置。

---

### 3. 长文本外推革命：RoPE 插值与 YaRN

由于 RoPE 是建立在连续旋转角度 $m\theta$ 基础之上的，AI 研究人员发现它可以像橡皮筋一样平滑拉伸，从而无需从零重训就能将模型的上下文从 4k 暴涨到 32k、128k 甚至 100 万词元：

1. **线性位置插值（Position Interpolation）**：
   若想将上下文窗口扩大 $s$ 倍（如扩大 4 倍：$4\text{k} \to 16\text{k}$），只需在推理时将位置步数整体缩放：


   $$
   m' = \frac{m}{s}
   $$


   这使得拉长后的所有旋转角度依然落在模型预训练时见识过的熟悉区间 $[0, 4000\theta]$ 内！

2. **NTK-Aware 缩放与 YaRN**：
   高频指针（跳得快的秒针）和低频指针（转得慢的年历盘）在长文本下的退化规律不同。通过针对性地动态调整基频底数（如将基数从 $10000$ 提升至 $500000$ 或上千万），现代模型（如 LLaMA-3、Qwen-2.5）可以在长达 12.8 万词元的海量文档检索中做到百步穿杨、“大海捞针”全绿通过！

---

## 第 5 步：手算极简数值示例（2D 旋转平面全流程） {: #step-5 }

为了把抽象的几何旋转化为每一个小学生都能亲手验算的算术，我们选取一个二维子空间（$d = 2$），展开全程纸笔手算。

### 1. 基础参数与向量设定

设当前二维特征平面的角频率为 $\theta = 0.5 \text{ 弧度}$（约等于 $28.65^\circ$）。

句子里有两个词元：

<p align="center">
  <kbd>词元 A: Query 发起者，位于位置 m = 1</kbd> &emsp;
  <kbd>词元 B: Key 响应者，位于位置 n = 3</kbd>
</p>

两者的相对距离为：



$$
n - m = 3 - 1 = 2 \quad \text{（Key 位于 Query 右侧 2 步距离）}
$$



假设未施加位置旋转前，它们的原始 Query 和 Key 特征向量分别为：



$$
\mathbf{q}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{k}_3 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}
$$



注意：如果不加位置信息，两者的原始点积为：



$$
\mathbf{q}_1^\top \mathbf{k}_3 = (1.0)(0.0) + (0.0)(1.0) = 0.0
$$



---

### 2. 旋转位置 $m = 1$ 处的词元 A

位置 $m = 1$ 的旋转弧度为：



$$
\phi_1 = m \theta = 1 \times 0.5 = 0.5 \text{ 弧度}
$$



查三角函数表（保留 4 位小数）：
- $\cos(0.5) \approx 0.8776$
- $\sin(0.5) \approx 0.4794$

词元 A 对应的二维旋转矩阵 $\mathbf{R}_{0.5, 1}$ 为：



$$
\mathbf{R}_{0.5, 1} = \begin{bmatrix}
\cos(0.5) & -\sin(0.5) \\
\sin(0.5) & \cos(0.5)
\end{bmatrix} = \begin{bmatrix}
0.8776 & -0.4794 \\
0.4794 & 0.8776
\end{bmatrix}
$$



对 Query 向量 $\mathbf{q}_1$ 实施旋转：



$$
\widetilde{\mathbf{q}}_1 = \mathbf{R}_{0.5, 1} \mathbf{q}_1 = \begin{bmatrix}
0.8776 & -0.4794 \\
0.4794 & 0.8776
\end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.8776 \\ 0.4794 \end{bmatrix}
$$



---

### 3. 旋转位置 $n = 3$ 处的词元 B

位置 $n = 3$ 的旋转弧度为：



$$
\phi_3 = n \theta = 3 \times 0.5 = 1.5 \text{ 弧度}
$$



查三角函数表：
- $\cos(1.5) \approx 0.0707$
- $\sin(1.5) \approx 0.9975$

词元 B 对应的二维旋转矩阵 $\mathbf{R}_{0.5, 3}$ 为：



$$
\mathbf{R}_{0.5, 3} = \begin{bmatrix}
\cos(1.5) & -\sin(1.5) \\
\sin(1.5) & \cos(1.5)
\end{bmatrix} = \begin{bmatrix}
0.0707 & -0.9975 \\
0.9975 & 0.0707
\end{bmatrix}
$$



对 Key 向量 $\mathbf{k}_3$ 实施旋转：



$$
\widetilde{\mathbf{k}}_3 = \mathbf{R}_{0.5, 3} \mathbf{k}_3 = \begin{bmatrix}
0.0707 & -0.9975 \\
0.9975 & 0.0707
\end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} -0.9975 \\ 0.0707 \end{bmatrix}
$$



---

### 4. 计算旋转后的 RoPE 点积打分

现在计算带有位置感知属性的两个新向量之间的点积：



$$
\begin{aligned}
\langle \widetilde{\mathbf{q}}_1, \widetilde{\mathbf{k}}_3 \rangle &= \widetilde{\mathbf{q}}_1^\top \widetilde{\mathbf{k}}_3 \\
&= (0.8776)(-0.9975) + (0.4794)(0.0707) \\
&= -0.875406 + 0.033894 \\
&= \mathbf{-0.8415}
\end{aligned}
$$



原本毫无关系的两个正交向量，因为在空间中扭转了不同的角度，精确算出了 $-0.8415$ 的关联打分！

---

### 5. 验证奇迹：用相对差公式直接一步验算（$n - m = 2$）

现在，我们直接调用苏剑林证明的相对距离黄金法则，看看不计算各自的旋转、直接用两者的相对差值矩阵是否能算出相同答案！

两者的相对距离为 $n - m = 3 - 1 = 2$。
相对旋转角为：



$$
\Delta\phi = (n - m)\theta = 2 \times 0.5 = 1.0 \text{ 弧度}
$$



查三角函数表：
- $\cos(1.0) \approx 0.5403$
- $\sin(1.0) \approx 0.8415$

相对旋转矩阵 $\mathbf{R}_{0.5, 2}$ 为：



$$
\mathbf{R}_{0.5, 2} = \begin{bmatrix}
\cos(1.0) & -\sin(1.0) \\
\sin(1.0) & \cos(1.0)
\end{bmatrix} = \begin{bmatrix}
0.5403 & -0.8415 \\
0.8415 & 0.5403
\end{bmatrix}
$$



直接计算相对公式 $\mathbf{q}_1^\top \mathbf{R}_{0.5, 2} \mathbf{k}_3$：



$$
\begin{aligned}
\mathbf{q}_1^\top \mathbf{R}_{0.5, 2} \mathbf{k}_3 &= \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 0.5403 & -0.8415 \\ 0.8415 & 0.5403 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} \\
&= \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} -0.8415 \\ 0.5403 \end{bmatrix} \\
&= \mathbf{-0.8415}
\end{aligned}
$$



<mark>两路手算结果在小数点后第 4 位丝毫不差、完全吻合！</mark>

---

### 6. 平移不变性测试：全句后移 10 步后的神奇一致性

如果这句话被放到了长篇小说的后面，位置变成了 $m' = 11$ 与 $n' = 13$ 呢？

它们的相对距离仍然是：



$$
n' - m' = 13 - 11 = 2
$$



快速手算验证：
1. $\phi_{11} = 11 \times 0.5 = 5.5 \text{ 弧度}$。$\cos(5.5) \approx 0.7087, \sin(5.5) \approx -0.7055$。
   $\widetilde{\mathbf{q}}_{11} = [0.7087, -0.7055]^\top$。
2. $\phi_{13} = 13 \times 0.5 = 6.5 \text{ 弧度}$。$\cos(6.5) \approx 0.9766, \sin(6.5) \approx 0.2151$。
   $\widetilde{\mathbf{k}}_{13} = [-0.2151, 0.9766]^\top$。
3. 计算内积：


   $$
   \widetilde{\mathbf{q}}_{11}^\top \widetilde{\mathbf{k}}_{13} = (0.7087)(-0.2151) + (-0.7055)(0.9766) = -0.1524 - 0.6889 = \mathbf{-0.8413} \approx \mathbf{-0.8415}
   $$



打分结果在平移了整整 10 个身位后依然完美锁死！这证明 RoPE 拥有无懈可击的**空间平移不变性**。

---

## 第 6 步：核心精髓总结 {: #step-6 }

!!! tip "核心要点: RoPE 旋转位置嵌入的核心精髓"
    **RoPE 通过精妙的二维坐标正交旋转，将抽象的相对词距映射为几何表盘上的旋转夹角差。**

    它用纯粹的乘性正交变换取代了粗暴的加性向量噪声，在彻底治愈注意力“置换等价性盲盒”的同时，达成了“绝对位置输入、相对距离输出”的代数奇迹。这不仅赋予了大语言模型敏锐的语序洞察力，更为当今百万级超长上下文外推奠定了坚不可摧的数学底座。
