# 第03章：空间拉伸魔盒（矩阵乘法）

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

想象一下，你的玩具房地板上平铺着一张巨大的**弹性橡胶布**。

橡胶布上用墨水画满了整整齐齐的正方形方格线，就像数学算术本上的方格纸一样。在橡胶布的最正中心 $(0, 0)$ 点上，放着一辆小巧的红色玩具小汽车。

<figure>
<pre>
       原始方正网格布                           拉伸后的弹性橡胶布
       ┌───┬───┬───┐                                 /───/───/───/
       │   │   │   │                                /   /   /   /
       ├───┼───┼───┤           用力拉扯            /───/───/───/
       │   │ ★ │   │           与倾斜             /   / ★ /   /
       ├───┼───┼───┤       ────────────►         /───/───/───/
       │   │   │   │                            /   /   /   /
       └───┴───┴───┘                           /───/───/───/
</pre>
<figcaption><strong>图 3.1:</strong> 抓紧边缘拉伸橡胶布，会把原本规整的正方形方格拉伸为倾斜的平行四边形，但直线依然笔直，平行线依然平行。</figcaption>
</figure>

现在，想象你双手抓住橡胶布的边缘用力拉扯：
- 你用力往右拉，把整张布拉宽了一倍（**拉伸**）。
- 你双手一上一下倾斜扭动，让方格向一侧斜斜倒去（**剪切与旋转**）。
- 你从上面往下压，把整个布压得更扁（**压缩**）。

在这个拉扯游戏里，请牢牢观察并记住三条黄金法则：

1. **原点被死死钉在地面**：红色小汽车坐落的中心点 $(0, 0)$ 绝对不能移位，它如同被图钉牢牢钉死在地板上。
2. **直线永远保持笔直**：无论你把橡胶布拉得多紧、扭得多斜，原来笔直的网格线绝不会弯曲，也绝不会变成曲里拐弯的波浪线。
3. **平行线永远保持平行**：拉扯前并肩延伸的两条线，拉扯后依然齐头并进，永不相交。

在严谨的数学世界中，这个拉伸橡胶布的游戏拥有一个响当当的名字：**线性变换（Linear Transformation）**。

而所谓**矩阵（Matrix）**，不过是一张精巧的指令卡片，它用一排整齐的数字精准告诉计算机：这张橡胶布究竟被向哪个方向拉了多长、扭了多偏、压了多扁！

---

<h2 id="step-2">Step 2: 承上启下的问题</h2>

在第01章与第02章中，我们认识到单词在大模型的语义地图上栖息为静止的坐标箭头（向量）。

例如，英文单词 <kbd>"bank"</kbd> 进入模型时具有一个固定的初始词嵌入。但是在真实的人类语言中，同一个词在不同句子里的含义千差万别：
- 在 *"river bank"*（河岸）中，这个词需要朝“流水、大自然、地理地貌”的方向移动。
- 在 *"bank deposit"*（银行存款）中，这个词必须朝“金钱、金融、金库”的方向移动。

光有一本静态的词嵌入字典，是远远无法构成人工智能大脑的——单词的含义不能像冰雕一样凝固不动！

这就引出了两个最根本的疑惑：
1. **什么是“权重”（$\mathbf{W}$）？它与我们在第01章学到的“嵌入”（$\mathbf{x}$）有何本质区别？**
2. **人工神经网络究竟是如何通过简单的算术运算，同时旋转、拉伸并重塑成千上万个词嵌入的？**

---

<h2 id="step-3">Step 3: 精确数学公式</h2>

在现代神经网络的每一个层级中，重塑向量空间都依托于深度学习最根本的基础方程式：**仿射线性变换（Affine Linear Transformation）**。

---

### 1. 从词嵌入到模型权重：权重究竟是什么？

在写下数学公式之前，我们必须先理清深度学习中两类最根本数字的天然分野：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 3.1:</strong> 根本分水岭：词嵌入（Embeddings）与模型权重（Weights）</caption>
  <thead>
    <tr bgcolor="#f0f0f0">
      <th align="left" width="18%">核心维度</th>
      <th align="left" width="32%">词嵌入向量（$\mathbf{x}$）</th>
      <th align="left" width="32%">权重矩阵（$\mathbf{W}$）</th>
      <th align="left" width="18%">直观生活比喻</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>它究竟是什么？</strong></td>
      <td><strong>输入数据</strong>：某个具体单词或词元的静态数字坐标。</td>
      <td><strong>处理引擎</strong>：网络学到的如何扭转、筛选、加工数据的数学规则。</td>
      <td>面团 vs. 压面机</td>
    </tr>
    <tr>
      <td><strong>它从何而来？</strong></td>
      <td>用户每次输入句子时，根据单词编号即时从词汇表中查表取出。</td>
      <td>在海量语料上训练数月得到，推理时<strong>永久固化在显存中</strong>。</td>
      <td>被观察的物体 vs. 彩色滤镜</td>
    </tr>
    <tr>
      <td><strong>它会随时改变吗？</strong></td>
      <td>会！句子里的每一个新词，都会带来全新的嵌入向量。</td>
      <td>不会！无论输入什么句子，权重矩阵里的数值都保持不变。</td>
      <td>车厢里的乘客 vs. 铁轨轨道</td>
    </tr>
    <tr>
      <td><strong>语法角色</strong></td>
      <td><strong>名词</strong>：被加工、被观察的对象本体。</td>
      <td><strong>动词</strong>：作用于名词之上的透镜、动作与运算。</td>
      <td>舞台上的演员 vs. 导演与聚光灯</td>
    </tr>
  </tbody>
</table>

<fieldset>
<legend><strong>回溯第01章：你其实已经结识过第一个权重矩阵了！</strong></legend>
在第01章中，我们引入了<strong>嵌入查找矩阵</strong> $\mathbf{E} \in \mathbb{R}^{|V| \times d}$。
请留意这个奇妙的统一性：$\mathbf{E}$ 本质上就是我们遇到的第一个权重矩阵！它容纳了 $|V|$ 个词向量，负责将离散的单词编号（一热编码向量）翻译成连续的语义空间坐标：

$$
\mathbf{x}_i^\top = \mathbf{e}_i^\top \mathbf{E}
$$

而到了第03章，层级中的<strong>权重矩阵</strong> $\mathbf{W}$ 则接过了接力棒：它接收现有的词嵌入 $\mathbf{x}$，并将它进一步投影、旋转到更深层、更高级的特征空间中。

如果说嵌入矩阵 $\mathbf{E}$ 是一本给每个单词赋予初始定义的<strong>词典</strong>，那么权重矩阵 $\mathbf{W}$ 就是负责理解这些词语如何相互作用的<strong>思维透镜</strong>。
</fieldset>

---

### 2. 核心变换方程式

对于一个维度为 $k$ 的输入词嵌入向量 $\mathbf{x} \in \mathbb{R}^{k \times 1}$、一个权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times k}$，以及一个偏置向量 $\mathbf{b} \in \mathbb{R}^{m \times 1}$，变换后的输出向量 $\mathbf{y} \in \mathbb{R}^{m \times 1}$ 为：

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

让我们把这个公式彻底展开为矩阵和向量的内部元素：

$$
\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix} = \begin{bmatrix} W_{1,1} & W_{1,2} & \cdots & W_{1,k} \\ W_{2,1} & W_{2,2} & \cdots & W_{2,k} \\ \vdots & \vdots & \ddots & \vdots \\ W_{m,1} & W_{m,2} & \cdots & W_{m,k} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_k \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
$$

在这个公式中：
- $\mathbf{W}\mathbf{x}$ 负责执行**橡胶布拉伸**：拉伸、旋转、倾斜剪切，或者改变词嵌入空间的维度大小。
- $+\, \mathbf{b}$ 负责执行**平移移动**：把拉伸好的整个坐标网格平移滑动，将坐标原点挪到更合适的新位置。

---

### 3. 内维匹配兼容法则

两个矩阵能够相乘的**充分必要条件**是：第一个矩阵的列数，必须严格等于第二个矩阵的行数。

<figure>
<pre>
            矩阵 A                     矩阵 B                    输出矩阵 C
      ┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
    m │                    │   k │                    │   m │                    │
      │                    │  ───│                    │  ───│                    │
    ▼ │                    │   ▼ │                    │   ▼ │                    │
      └────────────────────┘     └────────────────────┘     └────────────────────┘
         ◄─────── k ───────►        ◄─────── n ───────►        ◄─────── n ───────►
                   ▲                          ▲
                   └──── 必须分毫不差严格相等 ──┘
                             (k == k)
</pre>
<figcaption><strong>图 3.2:</strong> 矩阵相乘的内维匹配法则。中间的内维度 $k$ 在相乘累加后坍缩消失，留下尺寸为 $(m \times n)$ 的输出矩阵。</figcaption>
</figure>

写成维度公式：

$$
(m \times k) \times (k \times n) \longrightarrow (m \times n)
$$

如果矩阵 $\mathbf{A} \in \mathbb{R}^{m \times k}$，矩阵 $\mathbf{B} \in \mathbb{R}^{k \times n}$，则乘积矩阵 $\mathbf{C} \in \mathbb{R}^{m \times n}$ 中的每一个元素 $C_{i, j}$，都是通过将矩阵 $\mathbf{A}$ 的**第 $i$ 行**与矩阵 $\mathbf{B}$ 的**第 $j$ 列**做**点积**计算出来的：

$$
C_{i, j} = \sum_{r=1}^k A_{i, r} B_{r, j} = A_{i, 1} B_{1, j} + A_{i, 2} B_{2, j} + \dots + A_{i, k} B_{k, j}
$$

<fieldset>
<legend><strong>矩阵乘法本质上就是一张并行的点积阵列！</strong></legend>
请留意它与第02章的深刻契合：
乘积矩阵 $\mathbf{C}$ 里的每一个格子，纯粹就是左矩阵的一条水平横行向量与右矩阵的一条垂直坚列向量之间的<strong>点积</strong>。

矩阵乘法是一个高度组织化、结构化的并行计算流水线，用一场行云流水的运算同时并行求解成千上万个点积。
</fieldset>

---

### 4. 几何秘密：基向量最终落在了何处？（基于词嵌入的直观透视）

为什么研究神经网络模型的研究人员要如此执着于“基向量落在了何处”？

当大多数人初学矩阵乘法时，脑海中往往只有逐行点乘的死板算术规则：

$$
y_i = (\mathbf{W} \text{ 的第 } i \text{ 行}) \cdot \mathbf{x}
$$

虽然这解释了芯片在底层如何进行微观计算，但它把整个神经网络层变成了一张毫无生机的冰冷数字网格，无法帮我们建立对特征处理的直观物理认知。

从**矩阵的列（基向量的归宿）**来审视矩阵乘法，才能揭示神经网络重塑语义特征的真正工程蓝图。

#### 从词嵌入的角度看，“基向量”究竟是什么？

回想第01章，我们知道每个词嵌入 $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$ 都是由具体的语义坐标组成的（例如 $x_1 = 2$ 代表猫科属性，$x_2 = 1$ 代表活泼程度）。

而所谓标准基向量，其实就是该语义空间中**最纯粹、最极端的纯特征嵌入**：
- $\hat{\mathbf{i}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 代表一个拥有 **100% 纯粹第1特征**（如纯粹的“猫科属性”）、其余属性完全为 0 的纯概念嵌入。
- $\hat{\mathbf{j}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ 代表一个拥有 **100% 纯粹第2特征**（如纯粹的“活泼程度”）、其余属性完全为 0 的纯概念嵌入。

现实中任何一个具体单词的嵌入向量，都只是这两样纯粹特征原料的调配产物：$\mathbf{x} = x_1 \hat{\mathbf{i}} + x_2 \hat{\mathbf{j}}$。

#### 矩阵各列：神经网络的“特征转译词典”

现在，请观察当权重矩阵透镜 $\mathbf{W} = \begin{bmatrix} W_{1,1} & W_{1,2} \\ W_{2,1} & W_{2,2} \end{bmatrix}$ 投射在这些纯特征基向量嵌入上时，会发生什么：

$$
\mathbf{W} \hat{\mathbf{i}} = \begin{bmatrix} W_{1,1} & W_{1,2} \\ W_{2,1} & W_{2,2} \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} W_{1,1} \\ W_{2,1} \end{bmatrix} = \mathbf{W} \text{ 的第 1 列}
$$

$$
\mathbf{W} \hat{\mathbf{j}} = \begin{bmatrix} W_{1,1} & W_{1,2} \\ W_{2,1} & W_{2,2} \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} W_{1,2} \\ W_{2,2} \end{bmatrix} = \mathbf{W} \text{ 的第 2 列}
$$

这就揭示了神经网络空间变换的核心真谛：

> **权重矩阵的每一列，正是该层将纯特征基向量变换后，它们在输出空间中着陆的新坐标！**

矩阵 $\mathbf{W}$ 的每一列，实际上是该网络层编写的一条**特征转译词典词条**：
- **第 1 列（$\mathbf{w}_{:, 1}$）**回答了：*“如果输入的词汇携带 1 个单位的第1特征（猫科属性），本层应该为它催生出怎样的新特征？”*
- **第 2 列（$\mathbf{w}_{:, 2}$）**回答了：*“如果输入的词汇携带 1 个单位的第2特征（活泼度），本层应该为它催生出怎样的新特征？”*

#### 矩阵乘法本质是“概念配方搅拌机”

当一个真实的词嵌入向量 $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$（如 $\text{"cat"} = [2, 1]^\top$）进入神经网络层时，矩阵乘法在几何上执行的是：

$$
\mathbf{W}\mathbf{x} = x_1 (\mathbf{W} \text{ 的第 1 列}) + x_2 (\mathbf{W} \text{ 的第 2 列})
$$

输入嵌入向量 $\mathbf{x}$ 绝非冰冷的数学谜题，它是一份**概念搅拌配方**：
> *“请从第 1 列的概念中取 $x_1$ 份，从第 2 列的概念中取 $x_2$ 份，将它们充分混合，作为输出表征！”*

<fieldset>
<legend><strong>为什么“基向量-列视角”能彻底贯通后续模块？</strong></legend>
掌握矩阵的列视角，是理解后续深层神经网络机制的关键钥匙：
<ol>
  <li><strong>注意力投影矩阵（$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$）（第3模块）</strong>：当词嵌入被转译为“查询（Query）”时，$\mathbf{W}_Q$ 的各列定义了搜索空间的全新坐标轴——将原始词汇特征转译为<em>“当前词元正在向上下文探寻什么问题？”</em>。</li>
  <li><strong>前馈记忆网络（FFN）（第5模块）</strong>：在 LLaMA 和 GPT 中，中间层将向量升维至上万维再降维投影。这些矩阵的列构成了大模型的键值记忆槽，存储着庞大的事实性知识联想。</li>
  <li><strong>机制可解释性（Mechanistic Interpretability）</strong>：当 AI 安全研究人员尝试在模型内部寻找“拒绝回答方向”或“诚实向量”时，他们分析的正是这些矩阵各列所张成的线性组合方向。</li>
</ol>
</fieldset>

---

### 5. 深度学习约定：行向量与批处理矩阵

在纯数学教科书中，向量传统上被写为竖立的列向量：$\mathbf{y} = \mathbf{W}\mathbf{x}$。

然而在工业界深度学习框架（PyTorch、JAX、Hugging Face）中，文本是按**词元批次（Batches of Tokens）**进行并行处理的，其中每一个词元被表示为一条**水平行向量**。

对于一个包含 $T$ 个词元的序列，每个词元维度为 $d_{\text{in}}$：
- 输入矩阵为 $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{in}}}$（每一行是一个词元的嵌入向量）。
- 权重矩阵为 $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$。
- 偏置向量为 $\mathbf{b} \in \mathbb{R}^{1 \times d_{\text{out}}}$（自动广播加到全部 $T$ 行上）。

前向传播计算将输入矩阵置于左侧：

$$
\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

核对维度尺寸：

$$
(T \times d_{\text{in}}) \times (d_{\text{in}} \times d_{\text{out}}) \longrightarrow (T \times d_{\text{out}})
$$

依据转置恒等式 $(\mathbf{W}\mathbf{x})^\top = \mathbf{x}^\top \mathbf{W}^\top$，两套表述在数学实质上完全对等。

---

### 数学符号拆解速查表

<details>
<summary><strong>点击展开：形式化数学符号速查表</strong></summary>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 3.2:</strong> 第03章矩阵运算的形式化数学符号、维度与物理含义</caption>
  <thead>
    <tr bgcolor="#f0f0f0">
      <th align="center">符号</th>
      <th align="left">数学名称</th>
      <th align="center">标准形状</th>
      <th align="left">在 LLM 中的物理含义</th>
      <th align="left">实例</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$\mathbf{W}$</td>
      <td>权重矩阵</td>
      <td>$\mathbb{R}^{m \times k}$ 或 $\mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$</td>
      <td>神经网络中可学习的空间线性变换参数</td>
      <td>前馈网络中的投影权重矩阵</td>
    </tr>
    <tr>
      <td>$\mathbf{x}$</td>
      <td>输入向量</td>
      <td>$\mathbb{R}^{k \times 1}$</td>
      <td>变换前单个词元的嵌入坐标</td>
      <td>$\mathbf{x} \in \mathbb{R}^{4096 \times 1}$（Llama 3 维度）</td>
    </tr>
    <tr>
      <td>$\mathbf{X}$</td>
      <td>批输入矩阵</td>
      <td>$\mathbb{R}^{T \times d_{\text{in}}}$</td>
      <td>将 $T$ 个词元向量堆叠为水平行构成的序列矩阵</td>
      <td>$2048 \text{ 词元} \times 4096 \text{ 维度}$</td>
    </tr>
    <tr>
      <td>$\mathbf{b}$</td>
      <td>偏置向量</td>
      <td>$\mathbb{R}^{m \times 1}$</td>
      <td>平移滑动整个坐标系的恒定偏移量</td>
      <td>加在每个变换后点的坐标上</td>
    </tr>
    <tr>
      <td>$\mathbf{y}, \mathbf{Y}$</td>
      <td>输出空间表征</td>
      <td>$\mathbb{R}^{m \times 1}$ 或 $\mathbb{R}^{T \times d_{\text{out}}}$</td>
      <td>经过重塑后准备输送给下一层的隐藏状态</td>
      <td>融合了上下文的新特征坐标</td>
    </tr>
    <tr>
      <td>$\hat{\mathbf{i}}, \hat{\mathbf{j}}$</td>
      <td>基向量</td>
      <td>$\mathbb{R}^{d \times 1}$</td>
      <td>定义标准空间方向的单位坐标轴</td>
      <td>$[1, 0]^\top, [0, 1]^\top$</td>
    </tr>
  </tbody>
</table>

<dl>
  <dt><strong>线性变换（Linear Transformation）</strong></dt>
  <dd>向量空间之间的数学映射 $T(\mathbf{x})$，必须严格满足可加性 $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ 与标量齐次性 $T(c\mathbf{u}) = cT(\mathbf{u})$。</dd>
  
  <dt><strong>仿射变换（Affine Transformation）</strong></dt>
  <dd>在线性变换的基础上叠加一个平移向量：$f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$。严格的线性变换必须将原点固定在零点，而仿射变换可以把原点自由滑动到空间任意位置。</dd>

  <dt><strong>GEMM（通用矩阵乘法，General Matrix Multiply）</strong></dt>
  <dd>高性能计算标准算子：$\mathbf{C} \leftarrow \alpha \mathbf{A}\mathbf{B} + \beta \mathbf{C}$。它是整个现代机器学习中被工程优化得最极致的微观计算核心。</dd>
</dl>
</details>

---

<h2 id="step-4">Step 4: 公式从何而来？</h2>

### 1. 历史溯源
矩阵代数由英国数学家**阿瑟·凯莱（Arthur Cayley）**于 1858 年在其划时代的论文《矩阵理论备忘录》（*A Memoir on the Theory of Matrices*）中系统创立。凯莱发明矩阵并非为了简单存储数据表格，而是为了用简洁的代数形式表达线性方程组中的复合线性代换。

20世纪50至60年代，早期神经网络先驱**弗兰克·罗森布拉特（Frank Rosenblatt）**（感知机发明人）将矩阵-向量乘积引入人工智能，用以模拟生物大脑中多条树突输入信号汇聚为单个神经元突触膜电位的过程。

---

### 2. 为什么是线性变换？两大无可替代的超能力

为什么从最古老的多层感知机（MLP）到当今最先进的 Transformer，每一种深度学习架构都以矩阵乘法为绝对核心？

#### 超能力 1：精准、光滑的可微求导特性
当训练拥有数千亿参数的大模型时，我们必须精确计算改变每一个微小的权重 $W_{i, j}$ 会对最终的预测损失 $\mathcal{L}$ 产生怎样的影响。

因为矩阵乘法完全由线性的加法和乘法构成，它的导数惊人地纯净简练：

$$
y_i = \sum_{r=1}^k W_{i, r} x_r + b_i \implies \frac{\partial y_i}{\partial W_{i, j}} = x_j
$$

对权重 $W_{i, j}$ 的变化率，仅仅就是输入的激活值 $x_j$ 本身！这使得**反向传播算法（Backpropagation）**能够无需解复杂方程，同时并发更新数以千亿计的参数。

#### 超能力 2：保真几何结构
线性变换保留共线性与平行线。如果在输入空间中三个词向量构成了一条生动的语义类比：

$$
\mathbf{x}_{\text{king}} - \mathbf{x}_{\text{man}} + \mathbf{x}_{\text{woman}} \approx \mathbf{x}_{\text{queen}}
$$

应用线性变换 $\mathbf{W}$ 后，这个关系分毫不差地继续成立：

$$
\mathbf{W}(\mathbf{x}_{\text{king}} - \mathbf{x}_{\text{man}} + \mathbf{x}_{\text{woman}}) = \mathbf{W}\mathbf{x}_{\text{king}} - \mathbf{W}\mathbf{x}_{\text{man}} + \mathbf{W}\mathbf{x}_{\text{woman}} \approx \mathbf{W}\mathbf{x}_{\text{queen}}
$$

大模型可以将整套概念体系旋转、投影并拉伸到新的子空间中，而完全不会撕裂或破坏概念之间的内在联系。

---

### 3. 硅基硬件的秘密：为什么 GPU 是矩阵乘法吞吐巨兽？

请对比计算机在计算两个 $(N \times N)$ 矩阵相乘时所消耗的两种资源：
- **需要从内存读取的数据量**：$2 \times N^2$ 个浮点数（矩阵 $\mathbf{A}$ 和 $\mathbf{B}$）。
- **需要执行的算术运算次数**：$2 \times N^3$ 次微观运算（乘法与加法）。

注意这两者的比例：

$$
\frac{\text{运算次数}}{\text{内存传输量}} = \frac{O(N^3)}{O(N^2)} = O(N)
$$

这就是计算机体系结构中梦寐以求的圣杯——**高算术强度（High Arithmetic Intensity）**！  
对于大规模矩阵，GPU 只需要从昂贵缓慢的显存中加载一次数字，就能在不同的点积计算中将其复用数百次。

现代 AI 芯片（如 NVIDIA H100 GPU 或 Google TPU）利用张量核心构建了**脉动阵列（Systolic Arrays）**。数据在硅晶圆的物理二维乘法网格中富有节奏地流动，如同心脏泵血一般，无需等待缓慢的显存通信，每秒即可吞吐数百万亿次矩阵运算。

---

<h2 id="step-5">Step 5: 具象微型算例（笔算验证）</h2>

让我们用简单的微型数字，在纸上一笔一划推演完整的矩阵变换过程。

我们有一个输入词向量 <kbd>"cat"</kbd>，位于 2 维空间中：
- 维度 1：**毛茸茸度** $= 2$
- 维度 2：**活泼度** $= 1$

$$
\mathbf{x}_{\text{cat}} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$

我们希望通过一个权重矩阵 $\mathbf{W}$ 将其投影，该矩阵拉伸了毛茸茸度并倾斜了活泼度，随后加上一个偏置偏移 $\mathbf{b}$：

$$
\mathbf{W} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

让我们一步一步手算出最终的输出向量 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$。

---

### 第 5.1 步：核对维度兼容性

在动手算数前，始终先检查形状：
- 权重矩阵 $\mathbf{W}$：形状为 $(2 \times 2)$，所以 $m = 2, k = 2$。
- 输入向量 $\mathbf{x}$：形状为 $(2 \times 1)$，所以 $k = 2, n = 1$。
- 内维度吻合：$k = 2 = 2$。
- 输出形状将为：$(m \times n) = (2 \times 1)$。

维度 100% 完美匹配。

---

### 第 5.2 步：矩阵 $\mathbf{W}$ 乘以向量 $\mathbf{x}$

利用点积法则计算每一行的数值：

$$
\begin{aligned}
y_1^{\text{raw}} &= (\mathbf{W} \text{ 的第 1 行}) \cdot \mathbf{x} \\
&= (W_{1,1} \times x_1) + (W_{1,2} \times x_2) \\
&= (2 \times 2) + (1 \times 1) \\
&= 4 + 1 \\
&= \mathbf{5}
\end{aligned}
$$

$$
\begin{aligned}
y_2^{\text{raw}} &= (\mathbf{W} \text{ 的第 2 行}) \cdot \mathbf{x} \\
&= (W_{2,1} \times x_1) + (W_{2,2} \times x_2) \\
&= (0 \times 2) + (3 \times 1) \\
&= 0 + 3 \\
&= \mathbf{3}
\end{aligned}
$$

得到初步的原始线性投影：

$$
\mathbf{W}\mathbf{x} = \begin{bmatrix} 5 \\ 3 \end{bmatrix}
$$

---

### 第 5.3 步：加上偏置向量 $\mathbf{b}$

加上恒定的平移向量 $\mathbf{b} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$：

$$
\mathbf{y} = \begin{bmatrix} 5 \\ 3 \end{bmatrix} + \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 5 + 1 \\ 3 + (-1) \end{bmatrix} = \begin{bmatrix} \mathbf{6} \\ \mathbf{2} \end{bmatrix}
$$

最终求出的变换后向量为：

$$
\mathbf{y}_{\text{cat}} = \begin{bmatrix} \mathbf{6} \\ \mathbf{2} \end{bmatrix}
$$

<p>
  <strong>变换后的新坐标：</strong>
  第 1 维：<mark><strong>6.0</strong></mark> &nbsp;|&nbsp; 
  第 2 维：<mark><strong>2.0</strong></mark>
</p>

---

### 第 5.4 步：从几何基底视角验算

让我们查验在这场变换中，两个标准坐标轴分别落在了哪里：

1. **水平基向量 $\hat{\mathbf{i}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$**：
   $$
   \mathbf{W}\hat{\mathbf{i}} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}
   $$
   水平单位轴在水平方向上被拉伸放大了 $2$ 倍！

2. **垂直基向量 $\hat{\mathbf{j}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$**：
   $$
   \mathbf{W}\hat{\mathbf{j}} = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}
   $$
   垂直单位轴向右倾斜了 $+1$ 个单位，并在垂直方向上被拉长了 $3$ 倍！

<figure>
<pre>
       原始基向量方格                           变换后的基向量平行四边形
            ▲                                        ▲
            │                                    3.00│       • W(j) = (1, 3)
        1.00│   • j = (0, 1)                         │      /
            │   │                                    │     /
            │   └───► i = (1, 0)                     │    /
        0.00└───────┴────────►                   0.00└───┴───► W(i) = (2, 0)
          0.00    1.00                             0.00 1.00 2.00
</pre>
<figcaption><strong>图 3.3:</strong> 变换将原本的正方形重塑为倾斜拉伸的平行四边形，其边界边缘完全由矩阵 W 的两列所定义。</figcaption>
</figure>

直接用基底坐标重构 $\mathbf{W}\mathbf{x}$：

$$
\mathbf{W}\mathbf{x} = 2 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 1 \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 4 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 5 \\ 3 \end{bmatrix}
$$

基底视角的推导与逐行点积计算的结果毫厘不差！

---

<h2 id="step-6">Step 6: 核心精髓</h2>

> [!TIP] 核心精髓
> **权重矩阵**是一张可编程的弹性橡胶布，它在多维空间中拉伸、旋转并投射语义向量；而**偏置向量**则将坐标框架滑动到最佳的原点位置。
> 
> 矩阵乘法是大语言模型的动力核心引擎，因为它将平滑优雅的可微求导特性，与 GPU 张量核心无与伦比的硬件并行吞吐量完美结合。
> 
> 但这里隐藏着一个深刻的致命困境：**如果你将两层、三层甚至一百层拉伸橡胶布叠在一起，会发生什么？** 正如我们在第04章即将发现的那样，纯粹的连续矩阵相乘会在数学上瞬间坍缩为一张扁平的单层拉伸！为了孕育深邃的智能，我们必须引入**激活函数：单向单向阀门（Activation Functions）**。

---

<nav aria-label="Chapter Navigation">
  <p>
    <a href="../02-dot-product-and-similarity/index.html">&larr; 第02章：衡量关联（点积与余弦相似度）</a> &nbsp;|&nbsp; 
    <a href="../index.html">课程总览 / 目录</a> &nbsp;|&nbsp; 
    <strong>下一章：</strong> <a href="../04-activation-functions/index.html">第04章：单向阀门（激活函数：ReLU, GELU, SwiGLU） &rarr;</a>
  </p>
</nav>
