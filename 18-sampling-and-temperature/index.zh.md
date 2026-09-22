# 第 18 章：给模型加点温度（采样机制、温度系数、Top-k 与 Top-p）

---

## 步骤 1：3 岁孩子也能懂的直觉（故事创作的三重引擎）

> [!INTUITION] 从一眼看完全谱到掷骰子讲故事
> 想象你手里有一个极其聪明的讲故事八音盒。为了给孩子们娓娓道来一个精彩的故事，八音盒在机械内部会经历三个截然不同的运转阶段：
>
> 1. **阅读阶段（<dfn id="def-prefill-zh">预填充阶段，Pre-fill Phase</dfn>）&mdash; 一眼看全整张食谱**：
>    - 当你给八音盒一个开头或提问时，你递给它一张写满文字的卡片：<samp>“从前在神秘的森林里有一只……”</samp>
>    - 八音盒绝不会像识字初学者那样一个字一个字慢吞吞地读。相反，它的全景透镜在**同一瞬间、百分之一秒内一口气扫视整张卡片上的每一个字**！
>    - 它瞬间梳理出字与字之间的所有前因后果，在工作台上整齐铺开所有拼图碎片，并备好记忆档案以备后续写作。这就是**预填充（Pre-fill）**：以极高速度全并行消化你的整段提示词。
>
> 2. **写作阶段（<dfn id="def-decoding-zh">解码生成阶段，Decoding Phase</dfn>）&mdash; 逐张倒下的多米诺骨牌**：
>    - 读完之后，八音盒开始落笔写新故事。但它绝不可能跨越时间：在“今天”的词还没写出来之前，它根本不知道“明天”该写什么！
>    - 它先吐出第 1 个字：<kbd>"小"</kbd>。
>    - 然后它抬起笔，回头重新审视脑海里的全部记忆与这个“小”字，再吐出第 2 个字：<kbd>"狐"</kbd>。
>    - 接着再审视一遍，吐出第 3 个字：<kbd>"狸"</kbd>。
>    - 每一个新词的诞生都像一张倒下的多米诺骨牌：严格单向串行，一个接一个。这就是**解码（Decoding）**：自回归的逐词元生成。
>
> 3. **修补绝技（<dfn id="def-midfill-zh">中间填空，Mid-fill / Fill-in-the-Middle</dfn>）&mdash; 架设悬崖间的断桥**：
>    - 如果你已经写好了故事的精彩开头和圆满结尾，唯独中间空了一大段需要补充，该怎么办？
>    - 普通的八音盒只会顺着尾巴往前走，根本看不到后面的结局。
>    - **中间填空（Mid-fill）** 是一套巧妙的拼接魔术：它把“结局”直接剪贴到“开头”的后方，然后让八音盒开始生成缺失的“中间”。八音盒不仅没有破坏单向行进的规则，还能在落笔时同时看到前后两座大山！
>
> 4. **温度旋钮（<dfn id="def-temperature-zh">采样温度, Temperature, 记作 $T$</dfn>）与贵宾室保安**：
>    - 八音盒在准备吐出每一个词时，究竟如何做抉择？
>    - **冰封冷冻（$T \to 0$）**：完全丧失随机性，永远机械挑选最稳妥的词，沦为死记硬背的计算器。
>    - **宜人室温（$T = 0.7$）**：兼具严谨逻辑与生动文采，故事跌宕起伏。
>    - **沸腾烈火（$T = 5.0$）**：思维极度狂暴混乱，吐出荒诞不经的胡言乱语。
>    - **门口的保安（Top-$k$ 与 Top-$p$）**：守在候选室门前，彻底驱逐离谱荒谬的长尾词汇（比如在“从前有一只……”后面出现 <kbd>"微波炉"</kbd>）！

<figure>
<pre>
大语言模型端到端推理执行全景链路：

输入提示词（Prompt）："法国的首都是"
              │
              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 阶段 1：预填充阶段 PRE-FILL（全文本上下文并行摄入）                     │
│ • 输入：全部 S 个提示词元全量并发注入 [S × d]                           │
│ • 硬件特征：算力受限型 GEMM（张量核心 100% 满负荷轰鸣）                 │
│ • 核心产物：在 GPU 显存中初始化完整的键值缓存（KV Cache）               │
│ • 延迟度量指标：首词元生成时间（TTFT, Time To First Token）             │
└────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 阶段 2：解码生成阶段 DECODING（自回归单步串行推进）                     │
│ • 输入：严格只有上一步产生的单个新词元向量 [1 × d]                      │
│ • 硬件特征：访存带宽受限型 GEMV（从显存搬运全部历史 KV 缓存）           │
│ • 核心动作：将新键值写入 KV Cache；单查询向量与历史键做注意力计算       │
│ • 延迟度量指标：每个词元生成耗时（TPOT, Time Per Output Token）         │
└────────────────────────────────────────────────────────────────────────┘
              │
              ▼ 产生原始对数几率向量 z ∈ R^|V|
┌────────────────────────────────────────────────────────────────────────┐
│ 阶段 3：采样与截断过滤 SAMPLING（创造力调控与安全护栏）                 │
│ • 温度缩放：z_i / T                                                    │
│ • 概率映射：p_i = exp(z_i/T) / sum exp(z_j/T)                          │
│ • 危险长尾截断：Top-k（固定候选名额）或 Top-p（动态能量核）              │
│ • 离散抽样：吐出最终词元 "巴黎" ──► 拼入输入末尾 ──► 循环执行阶段 2！    │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 18.1：</strong> 大模型推理全生命周期：并行预填充构建初始 KV Cache，串行解码逐词元前推，采样策略精细塑造语言的创造力与严谨度。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

> [!BRIDGING] 从并行摄入到串行吐字：硬件架构与概率生成的鸿沟
> 当用户将一段长达 2,000 字的提示词发送给大模型时，计算引擎会在先后顺序上面临截然不同的硬件与数学挑战：
>
> 1. **预填充与解码的算力鸿沟**：为什么顶级 GPU 能在短短 15 毫秒内一口气吃透这 2,000 字的提示词，但在接下来生成 100 个字时，却要苦苦耗费 2,000 毫秒（足足 2 秒）？
> 2. **致命的内存墙（Memory Wall）**：为什么在逐字生成阶段，价值数十万的高性能显卡，其强大的张量计算核心竟然有超过 85% 的时间处于“空转摸鱼”状态，只能苦苦等待显存数据搬运？
> 3. **中间填空的工程智慧**：标准的单向因果注意力只能从左往右生成，那么当我们需要补全代码文件**中间**缺失的代码时，如何在不修改底层因果注意力的前提下让模型同时看到前后文？
> 4. **贪婪解码的死循环陷阱**：在生成每一步吐出的对数几率向量 $\mathbf{z} \in \mathbb{R}^{|V|}$ 之后，为什么我们绝不能直接每次都挑选概率第 1 名的词元（$\arg\max$）？
>
> 2019 年，阿里·霍尔茨曼（Ari Holtzman）等人通过数学证明（<cite>《The Curious Case of Neural Text Degeneration》</cite>）：单纯的**贪婪解码（Greedy Decoding）** 会导致模型必然跌入确定性的死循环复读陷阱（$A \to B \to A \to B$）。自然的人类语言既非刻板单调，也非纯粹混乱。
>
> “我们如何用数学形式严格刻画预填充与解码阶段？因果模型如何优雅实现中间填空？又如何利用统计热力学中的玻尔兹曼分布与动态核采样让机器讲出灵动的自然人话？”

---

## 步骤 3：严谨数学推导与公式

### 1. 预填充阶段（Pre-fill Phase：全上下文并行消化与 KV 缓存初始化）

在预填充阶段，长度为 $S$ 的完整用户提示词以嵌入矩阵的形式一次性输入模型：$\mathbf{X} \in \mathbb{R}^{S \times d}$。

Transformer 借助单次密集的通用矩阵乘法（GEMM）同时算出所有 $S$ 个词元的查询、键和值：

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{S \times d_k}, \quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{S \times d_k}, \quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{S \times d_v}
$$

整个提示词序列的全量自注意力分数矩阵在单次硬件时钟中全并行并发计算：

$$
\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \in \mathbb{R}^{S \times S}
$$

$$
\mathbf{O} = \mathbf{A}\mathbf{V} \in \mathbb{R}^{S \times d_v}
$$

其中 $\mathbf{M} \in \mathbb{R}^{S \times S}$ 为下三角因果掩码矩阵。

#### 核心产物：初始化键值缓存（KV Cache）
由于这 $S$ 个提示词元在后续的每一步生成中都将被反复作为历史背景回看，我们绝不丢弃它们的键和值，而是将它们以连续内存形式写入 GPU 的高带宽显存（HBM），作为初始 **KV Cache**：

$$
\mathbf{K}_{\text{cache}}^{(0)} = \mathbf{K} \in \mathbb{R}^{S \times d_k}, \quad \mathbf{V}_{\text{cache}}^{(0)} = \mathbf{V} \in \mathbb{R}^{S \times d_v}
$$

#### 预填充阶段的硬件执行特征：
- **核心算子**：通用矩阵乘法（<abbr title="General Matrix Multiply">GEMM</abbr>）。
- **计算密度（Arithmetic Intensity）**：每从显存加载 1 字节数据所执行的浮点运算次数高达 $\mathcal{O}(S \cdot d)$。由于提示词长度 $S$ 通常多达数百或数千，计算密度极高（远超 $100\text{ FLOPs/Byte}$）。
- **硬件瓶颈**：**算力受限（Compute-Bound）**。GPU 的张量计算核心（Tensor Cores）全速满载轰鸣。
- **核心工程性能指标**：**首词元生成时间（<abbr title="Time To First Token">TTFT</abbr>, Time To First Token）**。

---

### 2. 解码生成阶段（Decoding Phase：自回归增量单步迭代与内存墙）

当提示词完成预填充后，模型切换进入串行解码阶段，一步一步生成后续词元 $t = S+1, S+2, \dots$。

在生成第 $t$ 步时，模型的输入严格**只有上一步刚刚吐出的单个词元向量**：

$$
\mathbf{x}_t \in \mathbb{R}^{1 \times d}
$$

我们仅对这单个词元进行线性投影，生成单行查询、键和值：

$$
\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k}, \quad
\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{1 \times d_k}, \quad
\mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{1 \times d_v}
$$

#### 增量追加写入 KV 缓存：
无需重复计算历史词元，直接将这组新的 $(\mathbf{k}_t, \mathbf{v}_t)$ 拼接追加进显存中的缓存池：

$$
\mathbf{K}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{K}_{\text{cache}} \\ \mathbf{k}_t \end{bmatrix} \in \mathbb{R}^{t \times d_k}, \quad
\mathbf{V}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{V}_{\text{cache}} \\ \mathbf{v}_t \end{bmatrix} \in \mathbb{R}^{t \times d_v}
$$

#### 单查询增量注意力计算：
单行查询向量 $\mathbf{q}_t$ 回溯扫描全部 $t$ 个历史键：

$$
\mathbf{a}_t = \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{\text{cache}}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times t}
$$

$$
\mathbf{o}_t = \mathbf{a}_t \mathbf{V}_{\text{cache}} \in \mathbb{R}^{1 \times d_v}
$$

#### 解码阶段的硬件执行特征（致命的内存墙）：
- **核心算子**：矩阵-向量乘法（<abbr title="General Matrix Vector Multiply">GEMV</abbr>）。
- **计算密度**：极度低下（仅约 $1\text{--}2\text{ FLOPs/Byte}$）。为了仅仅计算这 1 个新词元的输出，GPU 不得不把模型全部数十亿/数百亿权重参数（例如 70B 模型约 $140\text{ GB}$ 权重）以及所有历史 KV 缓存，从显存颗粒完整搬运一遍到核心寄存器中！
- **硬件瓶颈**：**访存带宽受限（Memory-Bandwidth-Bound）**。计算核心绝大多数时间都在饥渴地等待显存总线的数据搬运。
- **核心工程性能指标**：**词元生成延迟（<abbr title="Time Per Output Token">TPOT</abbr>, Time Per Output Token）**，即用户所感知的吐字流速。

---

### 3. 中间填空（Fill-in-the-Middle / FIM）推理执行流程

当开发者让大模型在现有代码中间补全缺失函数时：
- **前缀（$P$）**：光标前面的代码片段。
- **后缀（$S$）**：光标后面的代码片段。
- **待填中间（$M$）**：需要模型智能补全的目标内容。

在传统的从左到右模型中，在 $P$ 后面直接生成 $M$ 无法参考 $S$，因为因果掩码遮蔽了未来信息。

**中间填空的推理执行协议：**
1. **输入序列重排包装**：推理引擎使用特殊定界符将输入重构为前缀-后缀-中间（PSM）格式：
   $$
   \mathbf{X}_{\text{infill}} = \langle\text{PRE}\rangle \circ P \circ \langle\text{SUF}\rangle \circ S \circ \langle\text{MID}\rangle
   $$
2. **预填充执行**：整条序列 $\mathbf{X}_{\text{infill}}$ 一次性并行输入模型执行预填充，显存中同时建立好 $P$ 和 $S$ 的全部 KV Cache。
3. **解码执行**：从 $\langle\text{MID}\rangle$ 之后的第一个位置启动自回归串行解码，逐字吐出中间词元 $m_1, m_2, \dots$。由于在物理序列上 $P$ 和 $S$ 都排在前面，生成的每个中间词元都能毫无阻碍地同时回看前缀与后缀！
4. **终止条件**：当模型吐出特殊终止符号 $\langle\text{EOT}\rangle$ 时解码结束，系统将中间生成的字符串原样插入到开发者的光标位置。

---

### 4. 温度系数缩放（玻尔兹曼分布，Boltzmann Distribution）

设模型最顶层输出的未归一化对数几率为 $\mathbf{z} = [z_1, z_2, \dots, z_{|V|}]^\top \in \mathbb{R}^{|V|}$。
**温度缩放 Softmax** 将词元 $i$ 的生成概率严格定义为：

$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}
$$

其中 $T > 0$ 为**温度超参数**。

#### 不同温度区间的严谨数学极限性质：
1. **当 $T \to 0^+$ 时（极寒贪婪极限，Argmax）**：
   最大对数几率 $z_{\max}$ 与其余所有对数几率的差值被放大到无穷大：
   $$
   \lim_{T \to 0^+} p_i(T) = \begin{cases} 1 & \text{若 } z_i = \max_j z_j \\ 0 & \text{其它情况} \end{cases}
   $$
   整个概率分布塌缩为确定性的 **One-Hot 狄拉克 $\delta$ 分布**。

2. **标准温度（$T = 1.0$）**：
   完全还原训练阶段交叉熵损失所使用的纯净原始 Softmax 分布：
   $$
   p_i(1.0) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
   $$

3. **当 $T \to \infty$ 时（狂暴均匀极限，白噪声）**：
   所有缩放后的对数几率都趋近于零（$z_i / T \to 0$），导致分子 $\exp(z_i / T) \to 1$：
   $$
   \lim_{T \to \infty} p_i(T) = \frac{1}{|V|}
   $$
   整个概率分布完全拉平，词表中所有词的被选概率完全均等（最大信息熵状态）。

---

### 5. Top-$k$ 截断过滤算法（Fan et al., 2018）

Top-$k$ 过滤强制只保留对数几率最高的前 $k$ 个词元，将其余所有词元的对数几率强行抹平为负无穷大：

$$
z'_i = \begin{cases} z_i & \text{若 } z_i \ge z_{(k)} \\ -\infty & \text{其它情况} \end{cases}
$$

其中 $z_{(k)}$ 是词表中第 $k$ 大的对数几率。
掩码屏蔽之后，在剩下的 $k$ 个候选词上重新计算 Softmax：

$$
p'_i = \frac{\exp(z'_i / T)}{\sum_{j=1}^{|V|} \exp(z'_j / T)}
$$

---

### 6. Top-$p$ 截断过滤算法（核采样，Nucleus Sampling；Holtzman et al., 2019）

Top-$k$ 存在死板的硬伤：固定的 $k=50$ 在模型极有把握时显得太臃肿（塞进了 49 个无关杂音词），而在语境极其开放复杂时又显得太狭隘（粗暴抹杀了合理的多样性）。

**Top-$p$（核采样）算法**通过累积概率动态自适应调整候选池窗口：
1. 将词表中所有词元按照概率大小降序排列：
   $$
   p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(|V|)}
   $$
2. 寻找使累积分布函数（<abbr title="Cumulative Distribution Function">CDF</abbr>）首次突破阈值 $p \in (0, 1]$ 的最小临界索引 $k^*$：
   $$
   k^* = \min \left\{ k : \sum_{i=1}^k p_{(i)} \ge p \right\}
   $$
3. 将前 $k^*$ 个核心词元集合定义为“核心核”（Nucleus $V^{(p)}$）：
   $$
   V^{(p)} = \{ (1), (2), \dots, (k^*) \}
   $$
4. 严格在核心核 $V^{(p)}$ 内部重新归一化概率分布：
   $$
   p'_i = \begin{cases} \frac{p_i}{\sum_{j \in V^{(p)}} p_j} & \text{若 } i \in V^{(p)} \\ 0 & \text{其它情况} \end{cases}
   $$

<figure>
<pre>
Top-p 核采样在不同语境下的动态弹性缩放表现：

情境 A：极度自信确定的语境（"法国的首都是……"）
  词元：        [ 巴黎 (0.97) │ 里昂 (0.01) │ 马赛 (0.01) │ ... ]
  累积概率：      0.97 >= 0.90 ──► 瞬间截断！候选池仅包含 1 个词！

情境 B：自由广阔的开放语境（"她打开窗户，看到外面有一只……"）
  词元：        [ 小鸟 (0.18) │ 树叶 (0.15) │ 猫咪 (0.12) │ 蝴蝶 (0.10) ... ]
  累积概率：      0.18 + 0.15 + 0.12 + 0.10 + ... >= 0.90 ──► 候选池自动膨胀为 18 个词！
</pre>
<figcaption><strong>图 18.2：</strong> 核采样在模型笃定时收缩至唯一解，在语境开放时动态扩充候选池。</figcaption>
</figure>

---

## 步骤 4：历史源流与思考演进（硬件瓶颈与采样理论）

<dl>
  <dt><time datetime="1877">1877</time> &mdash; <strong>路德维希·玻尔兹曼（Ludwig Boltzmann）</strong></dt>
  <dd>创立了统计热力学，证明了处于热力学温度 $T$ 下的物理系统，其微观状态 $i$（能量为 $E_i$）出现的概率严格服从麦克斯韦-玻尔兹曼分布 $p_i \propto \exp(-E_i / k_B T)$。在现代大模型中，负对数几率 $-z_i$ 恰好扮演了粒子的微观能量状态。</dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>阿希什·瓦斯瓦尼（Ashish Vaswani）等人</strong>（<cite>《Attention Is All You Need》</cite>）</dt>
  <dd>废除了循环神经网络的时序递归，以空间自注意力取而代之，彻底确立了大模型推理的双阶段分野：能够瞬间全并发完成的<strong>预填充阶段（Pre-fill）</strong>（$O(1)$ 串行步长），与受制于自回归因果约束而必须串行推进的<strong>解码阶段（Decoding）</strong>（$O(T)$ 串行步长）。</dd>

  <dt><time datetime="2018">2018</time> &mdash; <strong>安吉拉·范、迈克·刘易斯 与 雅恩·多芬</strong>（<cite>《Hierarchical Neural Story Generation》</cite>）</dt>
  <dd>在 Meta AI 提出了将 Top-$k$ 截断随机采样引入神经文本生成，证实斩断概率尾部可以大幅压制模型胡言乱语与机械复读的现象。</dd>

  <dt><time datetime="2019">2019</time> &mdash; <strong>阿里·霍尔茨曼 等人</strong>（<cite>《The Curious Case of Neural Text Degeneration》</cite>）</dt>
  <dd>深入剖析了人类语言的统计学分布真相，揭示了贪婪解码与固定 Top-$k$ 的根本缺陷，开创了 Top-$p$ 核采样机制，该算法至今仍是 ChatGPT、Claude、Gemini 等主流大模型的默认解码利器。</dd>

  <dt><time datetime="2022">2022</time> &mdash; <strong>穆罕默德·巴伐利亚（Mohammad Bavarian）等人</strong>（<cite>《Efficient Training of Language Models to Fill in the Middle》</cite>）</dt>
  <dd>在 OpenAI 发明了中间填空（FIM）技术，证明通过在训练阶段对前缀、中间和后缀做随机位置重排，就能让标准单向因果模型在不修改任何底层架构的情况下直接具备代码中间补全能力。</dd>

  <dt><time datetime="2023">2023</time> &mdash; <strong>权宇锡（Woosuk Kwon）等人</strong>（<cite>《Efficient Memory Management for Large Language Model Serving with PagedAttention》</cite>）</dt>
  <dd>开创了 vLLM 与 PagedAttention，借鉴操作系统虚拟内存分页机制来管理显存中的 KV Cache，彻底解决了自回归解码阶段显存碎片化与浪费的顽疾，让解码吞吐量实现几何级跃升。</dd>
</dl>

---

## 步骤 5：手把手超简单数字积木（全流程推理演算与概率手算）

### 1. 预填充与解码算力追踪（微型张量维度与硬件访存演算）

假设有一个极简的大语言模型，嵌入维度 $d = 4$，注意力投影维度 $d_k = 4$。
我们向其输入一个包含 3 个词元的提示词：<kbd>"谁 是 你"</kbd>（即序列长度 $S = 3$）。

```
提示词输入:   x_1="谁", x_2="是", x_3="你"  (S = 3)
模型维度:     d = 4
```

#### 阶段 1：预填充执行（消化整段提示词）
1. **输入嵌入矩阵**：$\mathbf{X} \in \mathbb{R}^{3 \times 4}$。
2. **高并发密集投影**：
   $$
   \mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{3 \times 4}, \quad
   \mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{3 \times 4}, \quad
   \mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{3 \times 4}
   $$
3. **全注意力矩阵计算**（GEMM）：
   $$
   \mathbf{S}_{\text{attn}} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{4}} + \mathbf{M} \in \mathbb{R}^{3 \times 3}
   $$
   所有 9 组注意力交互在单次硬件时钟中**并发齐射完成**。
4. **初始化 KV 缓存**：
   将 $\mathbf{K} \in \mathbb{R}^{3 \times 4}$ 与 $\mathbf{V} \in \mathbb{R}^{3 \times 4}$ 写入显存：
   $$\text{缓存占用数据量} = 2 \times (3 \times 4) = 24 \text{ 个浮点数}。$$
5. **吐出第 1 个生成词**：模型根据最后一行输出向量 $\mathbf{O}[3]$ 计算对数几率，采样吐出 <kbd>"我"</kbd>。

#### 阶段 2：解码生成执行（生成第 2 个新词元）
在获得 <kbd>"我"</kbd> 之后，进入第 $t = 4$ 步：
1. **输入单向量**：严格只有词元 <kbd>"我"</kbd> 的单行向量 $\mathbf{x}_4 \in \mathbb{R}^{1 \times 4}$。
2. **轻量投影**：
   $$
   \mathbf{q}_4 = \mathbf{x}_4 \mathbf{W}_Q \in \mathbb{R}^{1 \times 4}, \quad
   \mathbf{k}_4 = \mathbf{x}_4 \mathbf{W}_K \in \mathbb{R}^{1 \times 4}, \quad
   \mathbf{v}_4 = \mathbf{x}_4 \mathbf{W}_V \in \mathbb{R}^{1 \times 4}
   $$
3. **增量追加 KV 缓存**：
   $$
   \mathbf{K}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{K}_{\text{cache}} \\ \mathbf{k}_4 \end{bmatrix} \in \mathbb{R}^{4 \times 4}, \quad
   \mathbf{V}_{\text{cache}} \leftarrow \begin{bmatrix} \mathbf{V}_{\text{cache}} \\ \mathbf{v}_4 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
   $$
4. **增量注意力计算（GEMV）**：
   $$
   \mathbf{a}_4 = \operatorname{softmax}\left(\frac{\mathbf{q}_4 \mathbf{K}_{\text{cache}}^\top}{\sqrt{4}}\right) \in \mathbb{R}^{1 \times 4}
   $$
   $\mathbf{q}_4$ 是单行向量（$1 \times 4$），乘上 $4 \times 4$ 的历史键矩阵。
   为了完成这微小的几次乘加点积，显卡不得不把**整套庞大的模型权重与全部缓存数据从显存重新搬运一遍**！

---

### 2. 中间填空（FIM）执行追踪

假设我们希望模型补全如下函数：
- **前缀（$P$）**：`def square(x):`
- **后缀（$S$）**：`return y`
- **期望填入的中间（$M$）**：`y = x * x`

1. 推理引擎将其包装为 PSM 格式输入：
   ```
   [PRE] def square(x):\n [SUF] return y [MID]
   ```
2. **预填充阶段**：一次性并行摄入全部 7 个提示词元，显存中同时建立好函数签名 `def square(x):` 和返回值 `return y` 的全部键值缓存。
3. **解码阶段**：从 `[MID]` 之后开始，模型逐字生成 `y`、`=`、`x`、`*`、`x`，每一步都能同时关照上方的函数名与下方的返回值。
4. **结束收尾**：模型吐出 `[EOT]` 终止符，推理引擎将 `y = x * x` 剪切提取，准确嵌入开发者光标所在位置。

---

### 3. 采样概率计算演练（温度系数、Top-$k$ 与 Top-$p$ 手算）

考虑一个包含 4 个词元的极小词表 $V = \{\text{猫}, \text{狗}, \text{鱼}, \text{微波炉}\}$，模型顶层吐出的原始对数几率为：

$$
\mathbf{z} = [z_{\text{猫}} = 4.0, \; z_{\text{狗}} = 2.0, \; z_{\text{鱼}} = 1.0, \; z_{\text{微波炉}} = -1.0]
$$

<fieldset>
<legend><strong>执行清单</strong></legend>
<p><input type="checkbox" checked disabled> <strong>步骤 A：</strong> 根据指定温度 $T$ 缩放对数几率 $\tilde{z}_i = z_i / T$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 B：</strong> 计算各分子的自然指数 $e^{\tilde{z}_i}$ 并求和。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 C：</strong> 归一化计算当前各词元的真实概率 $p_i$。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 D：</strong> 在 $T=1.0$ 下执行 Top-$k=2$ 过滤与重归一化。</p>
<p><input type="checkbox" checked disabled> <strong>步骤 E：</strong> 在 $T=1.0$ 下执行 Top-$p=0.90$ 核采样过滤与重归一化。</p>
</fieldset>

#### 状态 A：基准标准温度（$T = 1.0$）
- 缩放对数几率：$\mathbf{z} / 1.0 = [4.0, 2.0, 1.0, -1.0]$
- 计算指数：
  - $e^{4.0} \approx 54.5982$
  - $e^{2.0} \approx 7.3891$
  - $e^{1.0} \approx 2.7183$
  - $e^{-1.0} \approx 0.3679$
  - 指数总和 $= 54.5982 + 7.3891 + 2.7183 + 0.3679 = \mathbf{65.0735}$
- 计算概率：
  - $p(\text{猫}) = 54.5982 / 65.0735 = \mathbf{0.8390} \; (83.9\%)$
  - $p(\text{狗}) = 7.3891 / 65.0735 = \mathbf{0.1136} \; (11.4\%)$
  - $p(\text{鱼}) = 2.7183 / 65.0735 = \mathbf{0.0418} \; (4.2\%)$
  - $p(\text{微波炉}) = 0.3679 / 65.0735 = \mathbf{0.0057} \; (0.6\%)$

#### 状态 B：冷静低温状态（$T = 0.5$ &mdash; 显著锐化，收敛确定性）
- 缩放对数几率：$\mathbf{z} / 0.5 = [8.0, 4.0, 2.0, -2.0]$
- 计算指数：
  - $e^{8.0} \approx 2980.9580$
  - $e^{4.0} \approx 54.5982$
  - $e^{2.0} \approx 7.3891$
  - $e^{-2.0} \approx 0.1353$
  - 指数总和 $= 2980.9580 + 54.5982 + 7.3891 + 0.1353 = \mathbf{3043.0806}$
- 计算概率：
  - $p(\text{猫}) = 2980.9580 / 3043.0806 = \mathbf{0.9796} \; (98.0\%)$
  - $p(\text{狗}) = 54.5982 / 3043.0806 = \mathbf{0.0179} \; (1.8\%)$
  - $p(\text{鱼}) = 7.3891 / 3043.0806 = \mathbf{0.0024} \; (0.2\%)$
  - $p(\text{微波炉}) = 0.1353 / 3043.0806 = \mathbf{0.00004} \; (0.004\%)$

<mark>看：当温度降至 $T = 0.5$ 时，<kbd>"猫"</kbd> 的当选概率从 $83.9\%$ 飙升至 $98.0\%$！分布极度锐化，几乎变成了无可争议的确定性预测。</mark>

#### 状态 C：狂躁高温状态（$T = 2.0$ &mdash; 显著拉平，增加混乱度）
- 缩放对数几率：$\mathbf{z} / 2.0 = [2.0, 1.0, 0.5, -0.5]$
- 计算指数：
  - $e^{2.0} \approx 7.3891$
  - $e^{1.0} \approx 2.7183$
  - $e^{0.5} \approx 1.6487$
  - $e^{-0.5} \approx 0.6065$
  - 指数总和 $= 7.3891 + 2.7183 + 1.6487 + 0.6065 = \mathbf{12.3626}$
- 计算概率：
  - $p(\text{猫}) = 7.3891 / 12.3626 = \mathbf{0.5977} \; (59.8\%)$
  - $p(\text{狗}) = 2.7183 / 12.3626 = \mathbf{0.2199} \; (22.0\%)$
  - $p(\text{鱼}) = 1.6487 / 12.3626 = \mathbf{0.1334} \; (13.3\%)$
  - $p(\text{微波炉}) = 0.6065 / 12.3626 = \mathbf{0.0491} \; (4.9\%)$

<mark>当温度升至 $T = 2.0$ 时，毫无关联的离谱词 <kbd>"微波炉"</kbd> 竟然捞到了整整 $4.9\%$ 的抽中概率！</mark>

---

#### 截断算法实战演练（Top-$k$ 与 Top-$p$ 在 $T = 1.0$ 下的实操）

以基准分布为例：
- $\text{猫}: 0.8390$
- $\text{狗}: 0.1136$
- $\text{鱼}: 0.0418$
- $\text{微波炉}: 0.0057$

##### 应用 Top-$k = 2$ 截断：
1. 强制仅保留支持率前 2 名：$\{\text{猫}, \text{狗}\}$。剔除 $\{\text{鱼}, \text{微波炉}\}$。
2. 保留候选词的概率总和：$0.8390 + 0.1136 = \mathbf{0.9526}$。
3. 重新归一化分配概率：
   - $p'(\text{猫}) = 0.8390 / 0.9526 = \mathbf{0.8808} \; (88.1\%)$
   - $p'(\text{狗}) = 0.1136 / 0.9526 = \mathbf{0.1192} \; (11.9\%)$
   - $p'(\text{鱼}) = \mathbf{0.0000}$
   - $p'(\text{微波炉}) = \mathbf{0.0000}$

##### 应用 Top-$p = 0.90$ 核采样截断：
1. 从高到低依次累加概率：
   - 第 1 名：$\text{猫} \to \text{当前累加值} = 0.8390 < 0.90$（尚未达标，继续吸纳）
   - 第 2 名：$\text{狗} \to \text{当前累加值} = 0.8390 + 0.1136 = 0.9526 \ge 0.90$（**成功越过 90% 门槛！在此处截断封门！**）
2. 核心核候选池同样刚好收容 $\{\text{猫}, \text{狗}\}$。
3. 归一化后概率：
   - $p'(\text{猫}) = \mathbf{88.1\%}$
   - $p'(\text{狗}) = \mathbf{11.9\%}$
   - 离谱词元 <kbd>"微波炉"</kbd> 被干脆利落地完全扼杀在摇篮中！

---

## 步骤 6：核心精要（一句话记住核心奥秘）

> [!TIP] 大模型推理与采样的核心心法
> **预填充以算力吞吐全景，构建记忆档案；解码以时序倒推骨牌，逐字前行；中间填空搭起悬崖跨度，沟通两端；采样策略则用热力学旋钮与概率护栏，将数学矩阵精巧转化为鲜活的人类智慧。**
>
> 深入理解计算密集（GEMM）与访存密集（GEMV）的物理界限，配合温度系数与核采样的动态雕琢，是驾驭现代大语言模型高效运行与精彩生成的根本基石。
