# 第 07 章：公平的投票箱（Softmax 函数与概率归一化）


## 第 1 步：3 岁小孩直觉（抢披萨的呼喊比赛与分披萨游戏） {: #step-1 }

!!! note "3岁小孩的直觉: 抢披萨的呼喊比赛与分披萨游戏"
    想象三个小朋友围坐在餐桌旁，大声喊出今晚披萨想要加的配料：

    - 小朋友 1 轻声耳语：<kbd>“蘑菇！”</kbd>（音量得分：4）
    - 小朋友 2 扯开嗓门尖叫：<kbd>“意大利辣香肠！”</kbd>（音量得分：7）
    - 小朋友 3 用平常常态说话：<kbd>“芝士！”</kbd>（音量得分：6）

    如果你手里只有这一组音量原始得分 $[4, 7, 6]$，你怎么把一张完整的披萨公平地分给他们？

    你显然不能把披萨切成“音量单位”。一张披萨是一个严丝合缝的整体（100%）。切出来的每一块披萨加起来必须正好等于整张披萨。更重要的是，没有任何一块披萨可以是负数（你总不能把别人肚子里已经吃掉的披萨挖出来吧！），也不能有任何一块披萨大过整张披萨。

    如果此时来了第 4 个心情糟糕的小朋友，在旁边小声嘀咕了一个负数得分：$-3$，你更不可能分给他“负数块披萨”！

    聪明的披萨大厨发明了一个**“公平投票箱”**游戏：

    1. **把所有抱怨变成正数堆**：大厨把每个人的音量得分放进一台“神奇指数放大机”（$e^z$）。哪怕是负数的嘀咕，也会被变成一小撮闪闪发光的正数面粉（$e^{-3} \approx 0.05$）；而最高分的大喊则变成一座巍峨的配料小山（$e^7 \approx 1097$）。没有任何东西是负数！
    2. **量一量整张桌子的总配料**：大厨把所有配料倒进同一个大盆里，称出总重量（$1555$ 克）。
    3. **按比例切分披萨**：每个小朋友分到的披萨份额，严格等于他自己的配料重量除以大盆的总重量。

    眨眼之间，每个人分到的披萨都严格落在 $0\%$ 到 $100\%$ 之间，而且所有人分到的份额加起来**严丝合缝正好等于 100%**！

<figure>
<pre>
[原始呼喊: Logits z]     [神奇指数放大机: e^z]     [分到的披萨比例: Softmax]
小朋友 1 ("蘑菇"):   4  ─────────►   54.6 克   ────────►   3.5% 的披萨
小朋友 2 ("香肠"):   7  ─────────► 1096.6 克   ────────►  70.5% 的披萨（大赢家！）
小朋友 3 ("芝士"):   6  ─────────►  403.4 克   ────────►  26.0% 的披萨
─────────────────────────────────────────────────────────────────────────────
全桌配料总和:                      1554.6 克   ────────► 100.0% 整张披萨
</pre>
<figcaption><strong>图 7.1：</strong> 将无拘无束、可能为负的原始音量得分转化为严格相加等于 100% 的非负概率分布。</figcaption>
</figure>

---

## 第 2 步：计算跨越的桥梁问题 {: #step-2 }

!!! question "计算连接问题: 从无边界的点积得分迈向合法的概率分布"
    在第 06 章中，歧义词 <kbd>"bank"</kbd>（河岸/银行）射出了自己的查询探针 $\mathbf{q}_3$，与句子中的三个词 <kbd>"The"</kbd>、<kbd>"river"</kbd>、<kbd>"bank"</kbd> 的招牌向量 $\mathbf{k}_j$ 分别进行点积碰撞，得到了原始匹配得分：



    $$
    \mathbf{z} = [4.0, 7.0, 6.0]
    $$



    但在线性代数的世界中，这些未加约束的原始点积（在深度学习中统称为 **Logits**）存在三个数学隐患：

    1. 取值范围无拘无束，可以是实数轴上的任意数值 $(-\infty, +\infty)$。
    2. 它们相加绝不可能自然等于 1（$4 + 7 + 6 = 17 \neq 1$）。
    3. 如果两个向量夹角大于 90 度，点积会出现负数。

    为了在注意力机制中将各个词的意义按比例加权融合（以及在最终模型输出层从 10 万个词表中挑出概率最大的预测词），大模型必须解决这个核心数学问题：

    *“我们如何把任意一组可能为正、为负、为零的无约束实数，在保留候选者相对高低顺序的前提下，平滑地转化为所有元素严格大于 0、且加和严丝合缝等于 100% 的合法概率分布？”*

---

## 第 3 步：严谨数学公式与架构推导 {: #step-3 }

### 1. Softmax 函数的标准数学定义

设 $\mathbf{z} = [z_1, z_2, \dots, z_N]^\top \in \mathbb{R}^N$ 为一个包含 $N$ 个候选打分的未归一化实数向量（即 <dfn id="def-logit">Logits</dfn>）。

<dfn id="def-softmax">Softmax 函数</dfn> 将该向量映射为一个概率分布向量 $\mathbf{s} = \operatorname{softmax}(\mathbf{z}) \in \mathbb{R}^N$：



$$
\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$



<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 7.1：</strong> Softmax 函数的核心数学符号与物理内涵清单</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="22%">数学符号</th>
      <th scope="col" align="left" width="22%">类型与定义域</th>
      <th scope="col" align="left" width="56%">严格数学含义与大模型中的物理作用</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left">$\mathbf{z}$</th>
      <td align="left">向量 $\in \mathbb{R}^N$</td>
      <td>输入的未归一化得分向量（Logits）。在注意力机制中，$z_j = \mathbf{q}_i^\top \mathbf{k}_j$。</td>
    </tr>
    <tr>
      <th scope="row" align="left">$z_i$</th>
      <td align="left">标量 $\in (-\infty, +\infty)$</td>
      <td>第 $i$ 个候选对象的原始得分。可正、可负、可为零。</td>
    </tr>
    <tr>
      <th scope="row" align="left">$e$</th>
      <td align="left">数学常数 $\approx 2.71828$</td>
      <td>自然底数（欧拉常数），自然指数函数的底。</td>
    </tr>
    <tr>
      <th scope="row" align="left">$e^{z_i}$</th>
      <td align="left">标量 $\in (0, +\infty)$</td>
      <td>指数化后的得分。由于指数函数的特性，对任意实数 $z_i$，恒有 $e^{z_i} > 0$。</td>
    </tr>
    <tr>
      <th scope="row" align="left">$\sum_{j=1}^N e^{z_j}$</th>
      <td align="left">标量 $\in (0, +\infty)$</td>
      <td><strong>归一化分母</strong>。在统计物理学中被称为<em>配分函数（Partition Function $Z$）</em>。</td>
    </tr>
    <tr>
      <th scope="row" align="left">$\operatorname{softmax}(\mathbf{z})_i$</th>
      <td align="left">标量 $\in (0, 1)$</td>
      <td>分配给第 $i$ 个选项的归一化概率，严格满足 $\sum_{i=1}^N \operatorname{softmax}(\mathbf{z})_i = 1.0$。</td>
    </tr>
  </tbody>
</table>

---

### 2. 概念梯子：为什么要采用自然指数 $e^z$？

为什么先驱数学家们不采用更简单的归一化技巧？一个合法的概率分布必须严格满足柯尔莫哥洛夫概率公理的两大基石：
1. **非负性**：每一个概率值必须大于等于 0，即 $p_i \ge 0$。
2. **归一性**：所有候选对象的概率之和必须严格为 1，即 $\sum_{i=1}^N p_i = 1.0$。

让我们沿着直觉备选方案的推导阶梯逐级向上排查，看看更简单的方案到底在哪些致命场景下全线崩溃。

#### 尝试 1：直接线性归一化（Linear Normalization）

最朴素的直觉是将每个得分除以所有得分之和：



$$
p_i = \frac{z_i}{\sum_{j=1}^N z_j}
$$



实践中的致命缺陷：
- **致命缺陷 1（除零崩溃）**：若原始得分加和恰好为零，分母直接消失。例如当得分向量为 $\mathbf{z} = [2.0, -3.0, 1.0]^\top$ 时：


  $$
  \sum_{j=1}^3 z_j = 2.0 + (-3.0) + 1.0 = 0
  $$


  计算 $p_i = \frac{z_i}{0}$ 会直接触发浮点除以零错误（`ZeroDivisionError` / `inf`），导致训练程序瞬间崩盘。
- **致命缺陷 2（负数概率）**：当存在负数得分且分母为正时，会产生负概率。例如当 $\mathbf{z} = [2.0, -1.0, 3.0]^\top$ 时，总和为 $\sum z_j = 4.0$，计算得：


  $$
  p_2 = \frac{-1.0}{4.0} = -0.25 = -25\%
  $$


  负概率在现实世界中没有任何物理意义，彻底违背概率公理。

#### 尝试 2：绝对值归一化（Absolute Value Normalization）

为了消除负数，很自然的尝试是先取绝对值 $|z_i|$ 再进行加和：



$$
p_i = \frac{|z_i|}{\sum_{j=1}^N |z_j|}
$$



实践中的致命缺陷：
- **致命缺陷（破坏排序与对称性混淆）**：绝对值彻底摧毁了得分的方向性意义。假设候选词 1 的得分为 $z_1 = -10.0$（极度不匹配），而候选词 2 的得分为 $z_2 = +10.0$（天作之合）。绝对值操作将两者都映射为 $10.0$：


  $$
  |-10.0| = |+10.0| = 10.0 \implies p_1 = p_2
  $$


  模型将给表现最差的词与表现最好的词分配完全相同的概率，完全颠倒非黑即白！

#### 尝试 3：线性整流截断归一化（ReLU Normalization）

为了消除负数且不翻转正负，尝试使用 ReLU 函数 $\max(0, z_i)$ 将负数统一截断为 0：



$$
p_i = \frac{\max(0, z_i)}{\sum_{j=1}^N \max(0, z_j)}
$$



实践中的致命缺陷：
- **致命缺陷 1（全负数崩溃）**：如果某一层的输出全为负数（例如 $\mathbf{z} = [-2.0, -5.0, -1.0]^\top$），则对所有 $j$ 均有 $\max(0, z_j) = 0$，分母为 0 导致 $\frac{0}{0} = \text{NaN}$。
- **致命缺陷 2（梯度死绝 / 丧失学习信号）**：对于所有被截断为 0 的负得分候选者，其局部导数严格为 0：


  $$
  \frac{\partial \max(0, z_i)}{\partial z_i} = 0
  $$


  在反向传播时，没有任何梯度能回传给这些被拒绝的词元。神经网络无法得知自己“错得有多离谱”，彻底丧失改进参数的学习通道。

#### 终极解法：自然指数函数 $e^z$

自然指数函数 $f(z) = e^z$ 完美解决了上述所有难题：



$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$



- **严格全域非负**：对实数轴上的任意数值 $z \in (-\infty, +\infty)$，恒有 $e^z > 0$。分母 $\sum e^{z_j} > 0$ 恒大于零，彻底杜绝除零崩溃，且保证每个概率 $p_i \in (0, 1)$。
- **严格单调递增（保持排序）**：由于 $\frac{d}{dz} e^z = e^z > 0$，函数严格单调上升（$z_a > z_b \iff e^{z_a} > e^{z_b} \iff p_a > p_b$）。高分候选者永远获得更高的概率，排序秩序秋毫无犯。
- **处处平滑可微**：指数函数为无穷阶光滑函数（$C^\infty$），导数处处不为零，为梯度反向传播提供无阻碍的绿色通道。
- **赢家通吃效应（Softmax 放大）**：指数曲线的非线性陡峭特性，能平滑地放大微小得分差距，使最有信心的词元脱颖而出，同时温和保留微小的长尾探索概率。

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 7.2：</strong> 各类归一化备选方案数学特性横向对比</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="22%">方案名称</th>
      <th scope="col" align="left" width="24%">归一化公式</th>
      <th scope="col" align="center" width="18%">全域非负？</th>
      <th scope="col" align="center" width="18%">保持排序？</th>
      <th scope="col" align="left" width="18%">反向传播梯度特性</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left">线性归一化</th>
      <td align="left">$p_i = \frac{z_i}{\sum z_j}$</td>
      <td align="center"><del>否（可能产生负数）</del></td>
      <td align="center">是</td>
      <td align="left">若 $\sum z_j = 0$ 则未定义</td>
    </tr>
    <tr>
      <th scope="row" align="left">绝对值归一化</th>
      <td align="left">$p_i = \frac{|z_i|}{\sum |z_j|}$</td>
      <td align="center">是</td>
      <td align="center"><del>否（负数对称翻转）</del></td>
      <td align="left">在 $z=0$ 处不可微</td>
    </tr>
    <tr>
      <th scope="row" align="left">ReLU 截断归一化</th>
      <td align="left">$p_i = \frac{\max(0, z_i)}{\sum \max(0, z_j)}$</td>
      <td align="center">是</td>
      <td align="center">部分保持</td>
      <td align="left">负数区域梯度全死（为 0）</td>
    </tr>
    <tr bgcolor="#f0f7f0">
      <th scope="row" align="left"><strong>Softmax（$e^z$）</strong></th>
      <td align="left"><strong>$p_i = \frac{e^{z_i}}{\sum e^{z_j}}$</strong></td>
      <td align="center"><strong>是（严格 &gt; 0）</strong></td>
      <td align="center"><strong>是（严格单调递增）</strong></td>
      <td align="left"><strong>处处光滑连续且非零</strong></td>
    </tr>
  </tbody>
</table>

---

### 3. 工程防溢出防线：平移不变性（Shift Invariance）

在现代计算机体系中，32 位单精度浮点数（<abbr title="IEEE 754 32-bit Single-Precision Float">FP32</abbr>）能表示的最大数值约为 $e^{88.7} \approx 3.4 \times 10^{38}$。如果大模型的未训练权重导致某一层输出了较大的点积（例如 $z_i = 1000$），计算机在计算 $e^{1000}$ 时会瞬间触发浮点上溢，输出无穷大 `inf`。而计算 `inf / inf` 会直接产生 `NaN`（Not a Number），导致整场训练海啸式崩溃！

幸运的是，Softmax 函数拥有一个近乎神圣的数学性质：**平移不变性**。

#### 平移不变性的严格数学证明

设 $c \in \mathbb{R}$ 为任意实数常数。我们将输入的每一个分量 $z_i$ 同时减去 $c$：



$$
\frac{e^{z_i - c}}{\sum_{j=1}^N e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_{j=1}^N \left(e^{z_j} \cdot e^{-c}\right)} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \cdot \sum_{j=1}^N e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}
$$



分子和分母中提取出的常数因子 $e^{-c}$ 在约分中**完全对消**！

#### 工业界的数值稳定防上溢法则

在 PyTorch、TensorFlow 和 FlashAttention 的底层实现中，系统会强制取常数 $c = \max_{j}(z_j)$：



$$
z'_i = z_i - \max_{j}(z_j)
$$



此时，整个向量中的最大值变成了 $0$（因为 $\max(z) - \max(z) = 0$），而其余所有元素均被平移为负数或零。因此：



$$
e^{z'_i} \in (0, 1] \quad (\forall i)
$$



指数运算的最大可能输出被死死锁死在 $e^0 = 1.0$，浮点数上溢在数学上被永久免疫！

---

### 4. 掌控创造力与理性的旋钮：温度系数（Temperature $T$）

在大语言模型的推理与解码阶段，开发者常调节一个大于 0 的超参数——**温度（Temperature $T > 0$）**：



$$
\operatorname{softmax}\left(\frac{\mathbf{z}}{T}\right)_i = \frac{e^{z_i / T}}{\sum_{j=1}^N e^{z_j / T}}
$$



<figure>
<pre>
   T -&gt; 0（冰冻绝对零度）           T = 1.0（标准适温）           T -&gt; inf（沸腾高温）
   [极度确信：近似 Argmax]         [模型真实原生分布]            [完全均匀分布：彻底混沌]
           |                               │                            │
      1.00 ┤     █                    0.70 ┤     █                 0.33 ┤  █   █   █
           │     │                         │     │                      │  │   │   │
           └─────┴──────                   └─────┴──────                └──┴───┴───┴──
            z1   z2   z3                    z1   z2   z3                 z1  z2  z3
</pre>
<figcaption><strong>图 7.3：</strong> 温度系数 $T$ 对候选词概率地形图的重构效应。</figcaption>
</figure>

- **低温模式（$T \to 0$，例如 $T = 0.2$）**：
  分母上的微小数值放大了得分之间的差距。原本第一名的优势被呈指数级放大，最终以接近 $100\%$ 的概率压倒性当选。模型表现为严谨、保守、确定性强（适合写代码、解数学题）。
- **标准模式（$T = 1.0$）**：
  保持神经网络参数训练得出的原始语义概率分布。
- **高温模式（$T \to \infty$，例如 $T = 2.0$）**：
  分母上的大数值压平了所有的得分差异，所有项的 $z_i / T \to 0$。由于 $e^0 = 1$，所有候选词的分数逐渐趋向均等（$1/N$）。模型表现出天马行空的想象力，但过高会导致前言不搭后语的胡言乱语。

---

### 5. 反向传播的微积分：局部导数（雅可比矩阵）

神经网络的训练依赖于梯度下降。为了让误差信号能够穿透 Softmax 层回传给前端的 Query 和 Key 投影矩阵，我们必须推导出 Softmax 的局部偏导数。

令输出概率为 $s_i = \frac{e^{z_i}}{\sum_{k=1}^N e^{z_k}}$。记分子 $u = e^{z_i}$，分母 $v = \sum_{k=1}^N e^{z_k}$。根据微积分的除法求导法则（商法则）：



$$
\frac{\partial s_i}{\partial z_j} = \frac{\frac{\partial u}{\partial z_j} v - u \frac{\partial v}{\partial z_j}}{v^2}
$$



注意到分母对任意 $z_j$ 的导数为 $\frac{\partial v}{\partial z_j} = \frac{\partial}{\partial z_j}\left(e^{z_1} + \dots + e^{z_j} + \dots\right) = e^{z_j}$。

现在我们必须分两种情形展开分析：

#### 情形 1：对角线元素（当 $i = j$ 时）

此时分子 $u = e^{z_i}$ 对自身变量 $z_i$ 求导，$\frac{\partial u}{\partial z_i} = e^{z_i}$：



$$
\begin{aligned}
\frac{\partial s_i}{\partial z_i} &= \frac{e^{z_i} v - e^{z_i} e^{z_i}}{v^2} \\
&= \frac{e^{z_i}}{v} - \left(\frac{e^{z_i}}{v}\right)^2 \\
&= s_i - s_i^2 \\
&= s_i (1 - s_i)
\end{aligned}
$$



#### 情形 2：非对角线元素（当 $i \neq j$ 时）

此时分子 $u = e^{z_i}$ 与自变量 $z_j$ 完全无关，$\frac{\partial u}{\partial z_j} = 0$：



$$
\begin{aligned}
\frac{\partial s_i}{\partial z_j} &= \frac{0 \cdot v - e^{z_i} e^{z_j}}{v^2} \\
&= -\frac{e^{z_i}}{v} \cdot \frac{e^{z_j}}{v} \\
&= -s_i s_j
\end{aligned}
$$



#### 统一的雅可比矩阵表达式

引入克罗内克函数（Kronecker Delta）$\delta_{ij}$（当 $i=j$ 时为 1，其余情况为 0）：



$$
\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j)
$$



这个导数公式惊人地优美：**Softmax 的梯度仅取决于它自身的输出概率！**在 GPU 上反向传播时，无需保留昂贵的中间激活，直接用前向输出相乘即可完成求导。

---

## 第 4 步：历史渊源与技术演进 {: #step-4 }

<dl>
  <dt><time datetime="1868">1868</time> &mdash; <strong>路德维希·玻尔兹曼与统计热力学（Ludwig Boltzmann）</strong></dt>
  <dd>
    奥地利物理学家玻尔兹曼在研究封闭容器内气体分子碰撞时发现：在温度为 $T$ 的热平衡系统中，物理系统处于能量为 $E_i$ 的微观状态的概率服从正规系综分布（麦克斯韦-玻尔兹曼分布）：



$$
P(i) = \frac{e^{-E_i / (k_B T)}}{\sum_j e^{-E_j / (k_B T)}}
$$



    物理系统更倾向于落入能量更低的微观态。将负能量 $-E_i$ 替换为算法效用得分 $z_i$，便诞生了现代概率归一化函数。
  </dd>

  <dt><time datetime="1959">1959</time> &mdash; <strong>R. Duncan Luce 与选择公理（Luce's Choice Axiom）</strong></dt>
  <dd>
    在数理心理学领域，Luce 提出了人类在多个候选项目中做决策的选择公理：选择某一选项的概率正比于该选项的主观效用比。采用指数效用函数 $u(z) = e^z$ 时，即推导出了心理测量学中的 Softmax 选择模型。
  </dd>

  <dt><time datetime="1989">1989</time> &mdash; <strong>John S. Bridle 与神经网络 Softmax</strong></dt>
  <dd>
    英国学者 John S. Bridle 在里程碑式论文 <em>"Probabilistic Interpretation of Feedforward Classification Network Outputs"</em> 中正式将该公式引入前馈神经网络。他将其命名为 <samp>"Softmax"</samp>，意在强调它是不可导的“硬最大值”（Hardmax / $\operatorname{argmax}$）的连续平滑可导替代品。
  </dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Vaswani 等人与注意力机制路由</strong></dt>
  <dd>
    在现代 Transformer 开山论文 <em>"Attention Is All You Need"</em> 中，Softmax 被置于缩放点积注意力的数学心脏：



$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$



    Softmax 赋予了模型动态调节上下文聚焦权重的非线性中枢能力。
  </dd>
</dl>

---

## 第 5 步：手算极简数值示例 {: #step-5 }

现在，让我们完整接续 **第 06 章第 5 步** 的数值篇章。

歧义词 <kbd>"bank"</kbd> 与三个词 <kbd>"The"</kbd>、<kbd>"river"</kbd>、<kbd>"bank"</kbd> 计算出了未归一化的原始点积匹配得分：



$$
\mathbf{z} = [z_1, z_2, z_3] = [4.0, 7.0, 6.0]
$$



让我们一步一步，通过纯手工运算将其转化为注意力权重，并最终加权合成出新的上下文向量！

---

### 计算执行核对清单

<fieldset>
<legend><strong>计算执行核对清单</strong></legend>
<p><input type="checkbox" checked disabled> <strong>步骤 5.1：</strong> 纯手工计算标准 Softmax 指数与概率分配</p>
<p><input type="checkbox" checked disabled> <strong>步骤 5.2：</strong> 验证工业级数值防溢出平移法（$c = \max$）</p>
<p><input type="checkbox" checked disabled> <strong>步骤 5.3：</strong> 对比不同温度旋钮（$T = 0.5$ 与 $T = 2.0$）的概率形变</p>
<p><input type="checkbox" checked disabled> <strong>步骤 5.4：</strong> 结合 Value 向量完成注意力上下文融合加权</p>
</fieldset>

---

### 步骤 5.1：标准 Softmax 纯手工运算

- **子步骤 A：计算各分量的指数（$e^{z_i}$）**：
  - $e^{z_1} = e^4 \approx 54.5982$
  - $e^{z_2} = e^7 \approx 1096.6332$
  - $e^{z_3} = e^6 \approx 403.4288$

- **子步骤 B：累加归一化分母（$\sum_{j=1}^3 e^{z_j}$）**：


  $$
  \sum = 54.5982 + 1096.6332 + 403.4288 = 1554.6602
  $$



- **子步骤 C：计算归一化注意力权重（$s_i = e^{z_i} / \sum$）**：
  - $s_1 (\text{"The"}) = \frac{54.5982}{1554.6602} \approx \mathbf{0.0351} \quad (3.51\%)$
  - $s_2 (\text{"river"}) = \frac{1096.6332}{1554.6602} \approx \mathbf{0.7054} \quad (70.54\%)$
  - $s_3 (\text{"bank"}) = \frac{403.4288}{1554.6602} \approx \mathbf{0.2595} \quad (25.95\%)$

- **子步骤 D：校验加和完整性**：


  $$
  0.0351 + 0.7054 + 0.2595 = \mathbf{1.0000} \quad (100.00\%)
  $$



---

### 步骤 5.2：数值平移稳定版检验（$c = \max$）

现在按工程防溢出标准重新计算一遍：

- **子步骤 A：提取最大得分**：
  $c = \max([4, 7, 6]) = 7.0$。

- **子步骤 B：中心化平移**：


  $$
  \mathbf{z}' = [4 - 7, 7 - 7, 6 - 7] = [-3.0, 0.0, -1.0]
  $$



- **子步骤 C：计算平移后的指数**：
  - $e^{-3.0} \approx 0.04979$
  - $e^{0.0} = 1.00000$
  - $e^{-1.0} \approx 0.36788$

- **子步骤 D：累加新分母**：


  $$
  \sum = 0.04979 + 1.00000 + 0.36788 = 1.41767
  $$



- **子步骤 E：计算最终概率**：
  - $s'_1 = \frac{0.04979}{1.41767} \approx \mathbf{0.0351}$
  - $s'_2 = \frac{1.00000}{1.41767} \approx \mathbf{0.7054}$
  - $s'_3 = \frac{0.36788}{1.41767} \approx \mathbf{0.2595}$

计算结果在保留 4 位有效小数下**完全一致**！但参与运算的最大数值从 $1096.6$ 被安全削减至 $1.0$。

---

### 步骤 5.3：温度系数对目光聚焦的调节

调动温度旋钮时，注意力权重的直观变化如下：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 7.2：</strong> 不同温度设定下词元注意力权重的分布对比</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">候选词元</th>
      <th scope="col" align="center" width="25%">低温专注 ($T = 0.5$)</th>
      <th scope="col" align="center" width="25%">标准原生 ($T = 1.0$)</th>
      <th scope="col" align="center" width="30%">高温扩散 ($T = 2.0$)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><kbd>"The"</kbd></th>
      <td align="center">0.15% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0015">0.15%</meter></td>
      <td align="center">3.51% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.0351">3.51%</meter></td>
      <td align="center">12.86% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1286">12.86%</meter></td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"river"</kbd> (核心线索)</th>
      <td align="center"><mark><strong>87.99%</strong></mark> <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.8799">87.99%</meter></td>
      <td align="center"><strong>70.54%</strong> <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.7054">70.54%</meter></td>
      <td align="center"><strong>52.79%</strong> <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.5279">52.79%</meter></td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"bank"</kbd> (自身参照)</th>
      <td align="center">11.86% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.1186">11.86%</meter></td>
      <td align="center">25.95% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.2595">25.95%</meter></td>
      <td align="center">34.35% <br><meter min="0" max="1" low="0.1" high="0.5" optimum="0.8" value="0.3435">34.35%</meter></td>
    </tr>
  </tbody>
</table>

- 当 $T = 0.5$ 时，线索词 <kbd>"river"</kbd> 的权重暴增至 **$88\%$**，注意力光束极度收窄！
- 当 $T = 2.0$ 时，各个词被平摊，注意力光束向全场均匀散射。

---

### 步骤 5.4：结合 Value 向量加权合成语义上下文

终于来到了注意力机制最终的丰收时刻！

在第 06 章第 3 节中，三个词元分别携带着自己的内容载荷向量（Value 向量 $\mathbf{v}_j \in \mathbb{R}^2$）：



$$
\mathbf{v}_1 (\text{"The"}) = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad
\mathbf{v}_2 (\text{"river"}) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
\mathbf{v}_3 (\text{"bank"}) = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$



歧义词 <kbd>"bank"</kbd> 经过注意力洗礼后获得的新上下文表征向量 $\mathbf{c}_{\text{bank}}$，就是这三组载荷在 Softmax 概率下的**加权线性组合**：



$$
\begin{aligned}
\mathbf{c}_{\text{bank}} &= \sum_{j=1}^3 s_j \mathbf{v}_j \\
&= 0.0351 \begin{bmatrix} 1 \\ 2 \end{bmatrix} + 0.7054 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.2595 \begin{bmatrix} 1 \\ 2 \end{bmatrix} \\
&= \begin{bmatrix} 0.0351 \times 1 + 0.7054 \times 0 + 0.2595 \times 1 \\ 0.0351 \times 2 + 0.7054 \times 1 + 0.2595 \times 2 \end{bmatrix} \\
&= \begin{bmatrix} 0.0351 + 0.0 + 0.2595 \\ 0.0702 + 0.7054 + 0.5190 \end{bmatrix} \\
&= \mathbf{\begin{bmatrix} 0.2946 \\ 1.2946 \end{bmatrix}}
\end{aligned}
$$



让我们凝视这个新向量：
- 在进入注意力机制前，<kbd>"bank"</kbd> 的静态向量是孤立的，分不清究竟是金融金库还是河流岸边。
- 经过 Softmax 权重的洗礼，新向量中足足有 **$70.54\%$** 的信息直接源自 <kbd>"river"</kbd>！
- <kbd>"bank"</kbd> 彻底褪去了孤立模糊的面纱，吸收了澎湃的“水系”语义！

---

## 第 6 步：核心精髓总结 {: #step-6 }

<fieldset>
<legend><strong>本章核心记忆卡片</strong></legend>
<p>
<strong>Softmax 是将无边界的线性代数冲量转化为严谨概率信仰的万能转换器：</strong><br>
通过自然指数函数 $e^z$，Softmax 确保了任何候选者都能获得非零的正数生机，同时放大了最高分候选者的压倒性优势；通过配分函数 $\sum e^{z_j}$ 归一化，它构建出了严格相加等于 $100\%$ 的合法概率分布。在 Transformer 的注意力机制中，Softmax 担任着动态信息分配总调度师的角色，精准决定每个词从全句其他词汇中汲取多少养分。
</p>
</fieldset>
