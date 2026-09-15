# 第02章：衡量关联（点积与余弦相似度）

<nav aria-label="Table of Contents">
  <p>
    <strong>目录导航：</strong> 
    <a href="#step-1">1. 3岁小孩的直觉</a> &bull; 
    <a href="#step-2">2. 承上启下的问题</a> &bull; 
    <a href="#step-3">3. 精确数学公式</a> &bull; 
    <a href="#step-4">4. 公式从何而来？</a> &bull; 
    <a href="#step-5">5. 具象微型算例</a> &bull; 
    <a href="#step-6">6. 核心精髓</a>
  </p>
</nav>

---

<h2 id="step-1">Step 1: 3岁小孩的直觉</h2>

想象一下，你和好朋友坐在玩具房的地毯上玩一个寻宝游戏。你们每人手里拿着一把玩具手电筒。

房间的不同角落摆放着不同的藏宝箱：一个装着软绵绵的毛绒小猫，一个装着活泼好动的小狗，还有一个装着红彤彤的红苹果。

<figure>
<pre>
                  【小猫藏宝箱】
                        ▲
                        / \
                       /   \
           光束 A     /     \     光束 B
                     /       \
                    /         \
              【手电筒 1】   【手电筒 2】
               (正向朝北)     (正向朝北)
</pre>
<figcaption><strong>图 2.1:</strong> 当两束手电筒的光芒照向完全相同的方向时，它们的光晕在同一个藏宝箱上完美交汇叠加。</figcaption>
</figure>

这个游戏的规则十分直观：

1. **同向而行**：如果你和朋友的手电筒指向**完全相同的方向**，两道光束就会融合成一团极其明亮的超级光斑！你们两人对宝藏的位置达成了 100% 的一致共识。
2. **垂直无关**：如果你正对着前方的小猫宝箱，而朋友转头把光束打向右侧的房门，两束光线在空中形成了垂直直角。两道光互不相干、没有交集，你们之间的共识度为**零**。
3. **背道而驰**：如果你向前方照去，而朋友却 180 度大转身朝背后的走廊照射，你们的光束朝向**完全相反的方向**。你们产生了彻底的分歧。

现在，注意这个游戏里最神奇的秘密：

假设朋友拿的是一个小巧的钥匙扣微型手电筒，而你扛的是一具巨大的户外探照灯。你的光束更长、更刺眼，但只要你们两人都笔直对准了小猫宝箱，你们所指向的依然是**完全相同的绝对方向**！

在人类语言中，含义的本质在于**方向**，而不在于箭头的物理长短。大模型需要一把精密的数学标尺，无论两个概念的箭头长度如何，都能瞬间测出它们是否指向同一座思想宝藏。

---

<h2 id="step-2">Step 2: 承上启下的问题</h2>

在第01章中，我们得知所有词汇都被表示为多维空间中的坐标箭头（向量）：

$$
\mathbf{u}, \mathbf{v} \in \mathbb{R}^d
$$

当大语言模型在生成文本或对比两个词语时，计算机究竟该如何度量这两个箭头是否朝向同一方向？

我们如何将空间中两个箭头之间的物理几何夹角，化为计算机硅晶圆能在纳秒内完成的极速加法与乘法？

---

<h2 id="step-3">Step 3: 精确数学公式</h2>

为了度量向量之间的几何对齐程度，大语言模型依赖两大核心数学基石：**点积**（Dot Product，亦称内积）与**余弦相似度**（Cosine Similarity）。

---

### 1. 代数点积（内积）

对于两个维度为 $d$ 的列向量 $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{d \times 1}$：

$$
\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_d \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_d \end{bmatrix}
$$

**点积**（记作 $\mathbf{u} \cdot \mathbf{v}$ 或 $\langle \mathbf{u}, \mathbf{v} \rangle$）将对应维度上的数值相乘，并将乘积求和：

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i = u_1 v_1 + u_2 v_2 + \dots + u_d v_d
$$

在矩阵乘法表示法中，点积等价于转置行向量 $\mathbf{u}^\top \in \mathbb{R}^{1 \times d}$ 与列向量 $\mathbf{v} \in \mathbb{R}^{d \times 1}$ 的乘积：

$$
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} \in \mathbb{R}
$$

---

### 2. 向量模长（$L_2$ 欧氏范数）

箭头 $\mathbf{u}$ 本身有多长？它的物理长度在数学上称为**欧氏范数**或 **$L_2$ 范数**（记作 $\|\mathbf{u}\|$ 或 $\|\mathbf{u}\|_2$），定义为向量与自身做点积后的算术平方根：

$$
\|\mathbf{u}\| = \sqrt{\mathbf{u} \cdot \mathbf{u}} = \sqrt{\sum_{i=1}^d u_i^2} = \sqrt{u_1^2 + u_2^2 + \dots + u_d^2}
$$

---

### 3. 几何点积定义

在欧几里得几何中，点积拥有一个完全等价的几何定义，它将两向量的长度与它们夹角 $\theta$ 的余弦值紧密联系在一起：

$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

请仔细观察这个公式的本质：
点积同时融合了**两件完全不同的物理量**：
1. 向量各自的长度大小（$\|\mathbf{u}\| \|\mathbf{v}\|$）。
2. 它们方向上的协同对齐程度（$\cos(\theta)$）。

---

### 4. 余弦相似度公式

为了彻底剥离向量长度对计算的干扰，专注于**纯粹的方向对齐**，研究人员将点积除以两个向量模长的乘积。这就得到了著名的**余弦相似度**（Cosine Similarity）：

$$
\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

将其分子与分母完全展开为各维度的坐标标量：

$$
\cos(\theta) = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}
$$

---

### 5. 归一化单位向量

如果我们预先将每个向量除以自身的模长，将其长度缩放为 $1.0$ 的**单位向量**（Unit Vector）：

$$
\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|}, \quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}
$$

那么余弦相似度就瞬间简化为这两个单位向量之间的普通点积：

$$
\cos(\theta) = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = \hat{\mathbf{u}}^\top \hat{\mathbf{v}}
$$

---

### 6. 三大几何区间与语义含义

根据圆周三角函数的天然性质，余弦值的取值范围被严格限定在闭区间 $[-1.0, +1.0]$ 之内：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 2.1:</strong> 余弦相似度的三大几何区间与语义映射对照表</caption>
  <thead>
    <tr bgcolor="#f0f0f0">
      <th align="center">余弦数值 $\cos(\theta)$</th>
      <th align="center">夹角 $\theta$</th>
      <th align="left">空间几何姿态</th>
      <th align="left">LLM 中的语义对应关系</th>
      <th align="center">图形化标尺</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>$+1.0$</strong></td>
      <td align="center">$0^\circ$</td>
      <td>朝向完全相同（共线平行）</td>
      <td>完全相同的语义概念 / 绝对同义词</td>
      <td align="center"><meter min="-1" max="1" value="1.0">1.0</meter></td>
    </tr>
    <tr>
      <td align="center"><strong>$0.0$</strong></td>
      <td align="center">$90^\circ$</td>
      <td>互相垂直（正交直角）</td>
      <td>毫不相关的独立主题 / 毫无语义交集</td>
      <td align="center"><meter min="-1" max="1" value="0.0">0.0</meter></td>
    </tr>
    <tr>
      <td align="center"><strong>$-1.0$</strong></td>
      <td align="center">$180^\circ$</td>
      <td>朝向截然相反（反向共线）</td>
      <td>完全相反的概念极性 / 语义反义词</td>
      <td align="center"><meter min="-1" max="1" value="-1.0">-1.0</meter></td>
    </tr>
  </tbody>
</table>

---

### 数学符号拆解速查表

<details>
<summary><strong>点击展开：形式化数学符号速查表</strong></summary>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 2.2:</strong> 第02章数学符号、维度与物理含义</caption>
  <thead>
    <tr bgcolor="#f0f0f0">
      <th align="center">符号</th>
      <th align="left">数学名称</th>
      <th align="center">维度形状</th>
      <th align="left">在 LLM 中的物理含义</th>
      <th align="left">实例</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$\mathbf{u}, \mathbf{v}$</td>
      <td>特征向量</td>
      <td>$\mathbb{R}^{d \times 1}$</td>
      <td>单词或词元的稠密多维坐标</td>
      <td>$\mathbf{u} = [3, 4]^\top$</td>
    </tr>
    <tr>
      <td>$\mathbf{u} \cdot \mathbf{v}$</td>
      <td>点积（内积）</td>
      <td>$\mathbb{R}$（标量）</td>
      <td>综合了方向一致性与长度强度的复合评分</td>
      <td>$3(6) + 4(8) = 50$</td>
    </tr>
    <tr>
      <td>$\|\mathbf{u}\|$</td>
      <td>$L_2$ 范数（模长）</td>
      <td>$\mathbb{R}_{\ge 0}$（非负标量）</td>
      <td>原点到向量箭头的直线物理欧氏距离</td>
      <td>$\sqrt{3^2 + 4^2} = 5$</td>
    </tr>
    <tr>
      <td>$\theta$</td>
      <td>空间夹角</td>
      <td>$[0^\circ, 180^\circ]$</td>
      <td>两概念之间的角度发散程度</td>
      <td>$\theta = 0^\circ$（方向一致）</td>
    </tr>
    <tr>
      <td>$\cos(\theta)$</td>
      <td>余弦相似度</td>
      <td>$[-1.0, 1.0]$</td>
      <td>纯粹的方向协同度，完全不受尺度影响</td>
      <td>$\cos(0^\circ) = 1.0$</td>
    </tr>
    <tr>
      <td>$\hat{\mathbf{u}}$</td>
      <td>单位向量</td>
      <td>$\mathbb{R}^{d \times 1}$ 且 $\|\hat{\mathbf{u}}\| = 1$</td>
      <td>长度固定为 1 的纯方向几何指示标</td>
      <td>$[0.6, 0.8]^\top$</td>
    </tr>
  </tbody>
</table>

<dl>
  <dt><strong>内积空间（Inner Product Space）</strong></dt>
  <dd>配备了内积运算的线性向量空间，该运算满足对称性 $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$、第一分量线性性质以及正定性 $\mathbf{u} \cdot \mathbf{u} \ge 0$。</dd>
  
  <dt><strong>正交性（Orthogonality）</strong></dt>
  <dd>两非零向量 $\mathbf{u}$ 与 $\mathbf{v}$ 正交当且仅当 $\mathbf{u} \cdot \mathbf{v} = 0$，在几何上意味着夹角严格为 $90^\circ$（$\frac{\pi}{2}$ 弧度）。</dd>

  <dt><strong>柯西-施瓦茨不等式（Cauchy-Schwarz Inequality）</strong></dt>
  <dd>数学定理保证 $|\mathbf{u} \cdot \mathbf{v}| \le \|\mathbf{u}\| \|\mathbf{v}\|$ 恒成立，从而在严谨数学上证明了余弦相似度 $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ 永远被牢牢锁定在 $[-1, +1]$ 之间。</dd>
</dl>
</details>

---

<h2 id="step-4">Step 4: 公式从何而来？</h2>

### 1. 历史溯源
点积概念最初由德国博学家**赫尔曼·格拉斯曼（Hermann Grassmann）**于 1844 年在其开创性著作《线性扩张论》（*Die Lineale Ausdehnungslehre*）中提出。19世纪80年代，美国物理学家**约西亚·威拉德·吉布斯（Josiah Willard Gibbs）**与英国电气工程师**奥利弗·亥维赛（Oliver Heaviside）**将点积和叉积从四元数中解耦，奠定了现代向量代数体系。

在计算语言学领域，余弦相似度由 **Gerard Salton** 在 20 世纪 70 年代为其著名的 SMART 信息检索系统创立，成为了**向量空间模型（Vector Space Model, VSM）**的基石。

---

### 2. 为什么不用欧氏距离？模长陷阱与频率偏差

初学者常常发问：
*既然要衡量词语之间的距离，为什么 LLM 不直接用欧氏距离 $\|\mathbf{u} - \mathbf{v}\| = \sqrt{\sum (u_i - v_i)^2}$，反而要大费周章去求夹角余弦？*

因为欧氏距离会掉入致命的**模长陷阱（Magnitude Trap）**（亦称文本长度/频次偏差）。

来看一个绝佳的直观对比：
- 文档 A：一条只有 10 个词的推特短讯，宣布“一只可爱的小猫赢得了比赛”。
- 文档 B：一篇长达 5000 字的维基百科百科全书词条，详尽剖析猫科动物的解剖生理构造。

两篇文档探讨的主题完全相同：**猫（Cat）**！
既然核心主题完全一致，它们的语义特征向量在空间中就指向**完全相同的方向**。

然而，因为文档 B 拥有 5000 个词，它的词频统计和特征激活值极其庞大。在坐标系中：
- 向量 A 是一支很短的轻量级箭头：$\mathbf{u}_A = [1, 2]$
- 向量 B 是一支长达百倍的重型大箭头：$\mathbf{u}_B = [100, 200]$

计算它们的**欧几里得距离**：

$$
\text{Distance}(\mathbf{u}_A, \mathbf{u}_B) = \sqrt{(1 - 100)^2 + (2 - 200)^2} = \sqrt{99^2 + 198^2} \approx \mathbf{221.36}
$$

欧氏距离给出了一个极其庞大的数字（221.36），断定两篇文档毫不相干！它把**篇幅长短（频次）**与**语义意图（主题）**彻底搞混了。

再来看它们的**余弦相似度**：

$$
\cos(\theta) = \frac{(1 \times 100) + (2 \times 200)}{\sqrt{1^2 + 2^2} \sqrt{100^2 + 200^2}} = \frac{100 + 400}{\sqrt{5} \sqrt{50000}} = \frac{500}{\sqrt{250000}} = \frac{500}{500} = \mathbf{1.0}
$$

余弦相似度瞬间看穿了假象：**它们指向完全相同的绝对方向（$\cos(\theta) = 1.0$）**！

<fieldset>
<legend><strong>为什么大语言模型看重方向而非长度？</strong></legend>
在现代 Transformer 嵌入中，高频虚词（如 <kbd>"the"</kbd>、<kbd>"is"</kbd>）以及附带强烈情绪色彩的词汇，在预训练优化过程中往往会自然生长出很大的向量模长。

通过计算余弦值或对向量进行单位化归一，模型得以确保语义比较始终基于纯粹的概念朝向，而不受词元出现频次的喧宾夺主。
</fieldset>

---

### 3. 硅基硬件的秘密：为什么 GPU 极度热爱点积？

为什么点积能够成为当代深度学习无可撼动的核心脉搏？

因为从微观物理层面来看，两个向量的点积就是一连串极为工整的**乘累加（Multiply-Accumulate, MAC）**运算：

$$
\text{累加器输出} \leftarrow \text{累加器输出} + (u_i \times v_i)
$$

现代 AI 芯片（无论是 NVIDIA H100 GPU 还是 Google TPU）在硬件硅片上集成了数以万计的专用加速单元——**张量核心（Tensor Cores）**。Tensor Cores 在物理晶体管级别被设计为可以在单个时钟周期内，并行并发执行数以万计的乘累加操作！

当 LLM 需要同时对比成千上万个词时，它将词向量堆叠为矩阵 $\mathbf{Q}$（查询 Query）和 $\mathbf{K}$（键 Key）。所有词对之间的点积运算，被一次性转化为单场极速的通用矩阵乘法：

$$
\mathbf{S} = \mathbf{Q}\mathbf{K}^\top
$$

矩阵里的每一个元素 $S_{i, j}$，就是词元 $i$ 与词元 $j$ 之间的点积。现代 GPU 运行该运算的速度高达每秒几百万亿次浮点运算（TFLOPS）。

---

<h2 id="step-5">Step 5: 具象微型算例（笔算验证）</h2>

让我们在一个 2 维语义概念空间中，用简单明了的微型数字一步一步完成笔算：
- 维度 1：**猫科动物特征**
- 维度 2：**活泼好动程度**

词汇表中有 3 个代表性概念：
1. <kbd>"cat"</kbd>（成年猫）：$\mathbf{u} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$
2. <kbd>"kitten"</kbd>（小猫咪）：$\mathbf{v} = \begin{bmatrix} 6 \\ 8 \end{bmatrix}$
3. <kbd>"apple"</kbd>（红苹果）：$\mathbf{w} = \begin{bmatrix} -4 \\ 3 \end{bmatrix}$

<figure>
<pre>
   活泼好动度 (第2维)
       ▲
   8.00│                                    ["kitten"] (6, 8)
       │                                     /
   6.00│                                    /
       │                                   /
   4.00│          ["cat"] (3, 4)          /
       │           /                     /  (完全相同的方向！θ = 0°)
   3.00│ ["apple"]                       /
       │  (-4, 3) \                     /
   2.00│           \                   /
       │            \                 /
   0.00└─────────────┴─────────────────┴────────────────────────► 猫科特征
     -4.00          0.00             3.00              6.00       (第1维)
</pre>
<figcaption><strong>图 2.2:</strong> 2D 坐标图。直观可见，“cat” 与 “kitten” 沿着从原点射出的同一条射线完全重叠排列（共线，θ = 0°），而 “apple” 则朝向左上方 90° 直角方向（正交，θ = 90°）。</figcaption>
</figure>

---

### 第 5.1 步：手算向量模长（$\|\cdot\|$）

应用欧氏范数公式 $\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2}$：

1. **"cat" 的模长（$\mathbf{u}$）**：
   $$
   \|\mathbf{u}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5}
   $$

2. **"kitten" 的模长（$\mathbf{v}$）**：
   $$
   \|\mathbf{v}\| = \sqrt{6^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = \mathbf{10}
   $$

3. **"apple" 的模长（$\mathbf{w}$）**：
   $$
   \|\mathbf{w}\| = \sqrt{(-4)^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = \mathbf{5}
   $$

---

### 第 5.2 步：手算代数点积（$\mathbf{u} \cdot \mathbf{v}$）

将对应分量相乘并累加：

1. **"cat" $\cdot$ "kitten"**：
   $$
   \begin{aligned}
   \mathbf{u} \cdot \mathbf{v} &= (u_1 \times v_1) + (u_2 \times v_2) \\
   &= (3 \times 6) + (4 \times 8) \\
   &= 18 + 32 \\
   &= \mathbf{50}
   \end{aligned}
   $$

2. **"cat" $\cdot$ "apple"**：
   $$
   \begin{aligned}
   \mathbf{u} \cdot \mathbf{w} &= (u_1 \times w_1) + (u_2 \times w_2) \\
   &= (3 \times -4) + (4 \times 3) \\
   &= -12 + 12 \\
   &= \mathbf{0}
   \end{aligned}
   $$

<kbd>"cat"</kbd> 与 <kbd>"apple"</kbd> 之间的点积恰好严格为 **$0$**！这说明它们在语义空间中彼此垂直正交。

---

### 第 5.3 步：手算余弦相似度（$\cos(\theta)$）

将点积除以两向量长度的乘积：

1. **"cat" 与 "kitten" 之间的余弦相似度**：
   $$
   \cos(\theta_{\text{cat, kitten}}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{50}{5 \times 10} = \frac{50}{50} = \mathbf{1.0}
   $$

   <p>
     <strong>对齐分数：</strong> <meter min="-1" max="1" value="1.0">1.0</meter>
     <mark><strong>1.0（完美 100% 绝对同向！）</strong></mark>
   </p>

   尽管 <kbd>"kitten"</kbd> 箭头的物理长度是 <kbd>"cat"</kbd> 的整整两倍（$10$ 对比 $5$），它们的余弦相似度依然是完美的 **$1.0$**！

2. **"cat" 与 "apple" 之间的余弦相似度**：
   $$
   \cos(\theta_{\text{cat, apple}}) = \frac{\mathbf{u} \cdot \mathbf{w}}{\|\mathbf{u}\| \|\mathbf{w}\|} = \frac{0}{5 \times 5} = \frac{0}{25} = \mathbf{0.0}
   $$

   <p>
     <strong>对齐分数：</strong> <meter min="-1" max="1" value="0.0">0.0</meter>
     <mark><strong>0.0（完全垂直正交，毫无关联）</strong></mark>
   </p>

---

### 第 5.4 步：单位向量法验证

将每个向量缩放为模长等于 1 的单位向量 $\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}$：

$$
\hat{\mathbf{u}}_{\text{cat}} = \frac{1}{5} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}
$$

$$
\hat{\mathbf{v}}_{\text{kitten}} = \frac{1}{10} \begin{bmatrix} 6 \\ 8 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}
$$

$$
\hat{\mathbf{w}}_{\text{apple}} = \frac{1}{5} \begin{bmatrix} -4 \\ 3 \end{bmatrix} = \begin{bmatrix} -0.8 \\ 0.6 \end{bmatrix}
$$

显而易见，$\hat{\mathbf{u}}_{\text{cat}}$ 与 $\hat{\mathbf{v}}_{\text{kitten}}$ **完全就是同一个单位向量**！

直接计算单位向量点积：

$$
\hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = (0.6 \times 0.6) + (0.8 \times 0.8) = 0.36 + 0.64 = \mathbf{1.0}
$$

$$
\hat{\mathbf{u}} \cdot \hat{\mathbf{w}} = (0.6 \times -0.8) + (0.8 \times 0.6) = -0.48 + 0.48 = \mathbf{0.0}
$$

两种推导路径殊途同归，数学逻辑严丝合缝。

---

<h2 id="step-6">Step 6: 核心精髓</h2>

> [!TIP] 核心精髓
> **点积**是现代人工智能的基础亲和力传感器：它将几何空间中的方向协同，巧妙转化为 GPU 张量核心每秒可执行上万亿次的极速乘累加运算。
> 
> 而通过除以模长，**余弦相似度**剥离了词频与序列长度的干扰，赋予大模型直接审视纯粹概念内涵的能力。
> 
> 然而，现实语言中的单词绝非一成不变：在真实句子里，上下文会不断互动并改造词义。**大语言模型究竟是如何在层与层之间旋转、拉伸并投射这些向量的？** 这正是第2模块的核心——**矩阵乘法（Matrix Multiplication）**。

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../01-vectors-and-spaces/index.html">&larr; 第01章：词汇地图（向量与词嵌入）</a> &nbsp;|&nbsp; 
    <a href="../index.html">课程总览 / 目录</a> &nbsp;|&nbsp; 
    <strong>下一章：</strong> <a href="../03-matrix-multiplication/index.html">第03章：神奇伸缩盒（矩阵乘法） &rarr;</a>
  </p>
</nav>
