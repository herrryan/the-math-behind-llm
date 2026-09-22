# 第 06 章：图书馆寻宝记——查询、键与值（Q、K、V 投影矩阵）


## 第 1 步：3 岁小孩的直觉（3-Year-Old Intuition） {: #step-1 }

想象你牵着爸爸妈妈的手，走进了一座巨大无比的**魔法图书馆**。

这座图书馆里藏着数百万本故事书。今天，你的老师给你布置了一个寻宝任务：**“找出哪本书讲了‘喷火恐龙’的故事！”**

在没有任何魔法之前，如果你想找到答案，你只能把图书馆里的每一本书都从书架上抽出来，从第 1 页一字一句读到第 500 页。这会把你活活累趴下，几天几夜也读不完。

于是，图书管理员爷爷发明了一套绝顶聪明的**“三张魔法卡片”**寻宝游戏：

<figure>
<pre>
   [你手里的心愿卡: Query] ──► "我想看会喷火的飞天大恐龙！"
                                       │
                                       ▼ (拿着心愿卡在书架前逐一比对)
                                ┌──────┴──────┐
                                ▼             ▼
     [书架外部的标签卡: Key 1] "烘焙蛋糕"    [书架外部的标签卡: Key 2] "远古飞龙"
            比对匹配度: 0% ✘                        比对匹配度: 95% ✔
                                                      │
                                                      ▼ (匹配度极高！抽出来打开书)
                                            [书本内部的宝藏故事: Value 2]
                                            "从前有一只浑身赤红的喷火飞龙..."
</pre>
<figcaption><strong>图 5.1：</strong> 图书馆寻宝记的三卡协同。你手握心愿卡（Query），与书脊标签（Key）进行快速匹配，最后只翻开高分匹配书本内部的真实故事（Value）。</figcaption>
</figure>

### 1. 第一张卡片：心愿卡（Query，简称 Q &mdash; “我在寻找什么？”）
- 这是你走进图书馆时，手里攥着的那张便签纸。
- 它代表了**你当前最饥渴的提问**。比如：“谁能告诉我关于‘喷火’的事情？”

### 2. 第二张卡片：门牌标签卡（Key，简称 K &mdash; “我是关于什么的？”）
- 图书馆里的每本书，书脊外面都贴着一张简短的**分类检索标签**。
- 标签并不记载整个长篇故事，它只负责**向路过的人宣传自己的主题**：比如“园艺植物”、“法式甜点”、“史前飞龙”、“浩瀚星空”。
- 它的唯一使命，就是与你手里的心愿卡（Query）碰一碰，看看**两个人合不合拍、匹配度有多高**。

### 3. 第三张卡片：宝藏故事本体（Value，简称 V &mdash; “我肚子里的真实墨水”）
- 当你发现某本书的标签卡（Key）和你的心愿卡（Query）百分之百吻合时，你才会激动地把这本书从书架上抽出来，翻开书页。
- 书页里密密麻麻记载的**真正知识、情节与文字内容**，就是宝藏故事本体（Value）。

---

### 为什么不能把“标签卡（Key）”和“宝藏故事（Value）”混为一谈？

设想一下：如果图书管理员偷懒，把书本正文的 500 页故事直接印在书脊外面，书架立刻会变得臃肿不堪！
更关键的是：**“检索线索”与“内容本身”在逻辑上是截然不同的两件事**。
- 当你想查阅“治疗感冒的药方”时，你检索的关键词是标签（Key = “医学退烧”）；
- 但你最终希望带走的，是药方里记载的真实成分配比（Value = “对乙酰氨基酚 500 毫克”）。

在现代大语言模型（Transformer）的大脑里：
- 每一个词（Token）都同时扮演**寻宝者**（发出 Query）和**藏宝书**（提供 Key 供人检索，奉献 Value 给人吸纳）。
- **这就是著名的注意力机制三剑客：Query、Key、Value！**

---

## 第 2 步：计算跨越的桥梁问题（The Bridging Question） {: #step-2 }

在第 01 章至第 04 章中，我们见证了单词如何化作静态的词向量 $\mathbf{x} \in \mathbb{R}^d$，并通过前馈神经网络进行空间变换与激活筛选。在刚刚结束的微型大脑工坊（Lab 01）中，我们更是亲手用 80 行纯 Python 训练了一个基于 Bengio 2003 的前馈网络。

但工坊的实验残酷地暴露了一个**根本性缺陷**：
在传统前馈网络中，每个词的向量表示在输入阶段是**死板固定、孤立僵化**的！

请看这句经典的人类语言歧义句：
> *“他在**河岸**（bank）边散步，随后走进了附近的**银行**（bank）办理汇款。”*

<figure>
<pre>
   静态嵌入字典 (Lookup Table E)
   ┌─────────┬──────────────────────┐
   │ Token   │ 静态嵌入向量 x       │
   ├─────────┼──────────────────────┤
   │ "bank"  │ [ 0.82, -0.41, 0.15 ] │ ◄── 无论是河岸还是银行，都是同一个死向量！
   └─────────┴──────────────────────┘

   句子中的第 1 个 bank ──► 紧挨着 "河岸"、"泥泞"、"流水" ──► 它的真实语义应当是 [水利地理]！
   句子中的第 2 个 bank ──► 紧挨着 "汇款"、"利息"、"账户" ──► 它的真实语义应当是 [金融机构]！
</pre>
<figcaption><strong>图 5.2：</strong> 静态嵌入的“一词多义失忆症”。没有上下文信息的主动检索，同一个词无论放在什么句子中，其初始表示都完全相同。</figcaption>
</figure>

计算机面临着迫在眉睫的桥梁挑战：
1. **动态语境渴求**：位于第 1 个位置的 `"bank"`，必须有能力主动**环顾四周**，向周围的词打听：“喂！你们谁和‘河流水体’有关？请把你们的信息借给我融合一下！”
2. **计算转化问题**：计算机如何将“拿着心愿卡比对书脊标签，再抽取宝藏内容”的实体寻宝游戏，转化为可以通过 GPU 高速并行的矩阵乘法？
3. **投影必要性疑问**：为什么一个词不能直接拿自己的原始输入向量 $\mathbf{x}$ 去和别人点积？为什么必须通过三个完全不同的矩阵（$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$）把原始向量投影到三个不同的子空间？

---

## 第 3 步：严谨数学公式与推导（The Exact Math & Formula） {: #step-3 }

---

### 1. 核心数学公式定义

设一个包含 $T$ 个 Token 的输入文本序列，经过词嵌入矩阵查找后，形成输入特征矩阵：

$$
\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

其中：
- $T$ 为序列的时间步长度（Token 数量，如上下文窗口内的 2048 个词）；
- $d_{\text{model}}$ 为大模型的隐层主维度（如 LLaMA-3 8B 中 $d_{\text{model}} = 4096$，GPT-3 中 $d_{\text{model}} = 12288$）。

为了赋予模型在不同角色维度上观察信息的能力，Transformer 引入了**三个完全独立、可学习的线性投影参数矩阵**：

$$
\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

通过三次并行的矩阵乘法，原始输入矩阵 $\mathbf{X}$ 被瞬间投射为三个全新的几何空间：

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{T \times d_k} \quad (\text{查询矩阵，Query Matrix})
$$

$$
\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{T \times d_k} \quad (\text{键矩阵，Key Matrix})
$$

$$
\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{T \times d_v} \quad (\text{值矩阵，Value Matrix})
$$

在单头注意力（Single-Head Attention）的经典设定中，通常取 $d_k = d_v = d_{\text{model}}$；而在现代多头注意力机制（Multi-Head Attention）中，设头数为 $h$，则每个注意力子空间的维度被切分为：

$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

例如在 LLaMA-3 8B 中，$d_{\text{model}} = 4096, h = 32$，因此每个头对应的注意力子空间维度为 $d_k = \frac{4096}{32} = 128$。

---

### 2. 单个 Token 视角的行向量分解

宏观的矩阵乘法本质上是每个 Token 独立向量运算的并行打包。让我们剥离矩阵外壳，透视位于第 $i$ 个位置的单字行向量 $\mathbf{x}_i^\top \in \mathbb{R}^{1 \times d_{\text{model}}}$：

$$
\mathbf{q}_i^\top = \mathbf{x}_i^\top \mathbf{W}_Q \in \mathbb{R}^{1 \times d_k} \quad (\text{Token } i \text{ 的查询向量})
$$

$$
\mathbf{k}_i^\top = \mathbf{x}_i^\top \mathbf{W}_K \in \mathbb{R}^{1 \times d_k} \quad (\text{Token } i \text{ 的键向量})
$$

$$
\mathbf{v}_i^\top = \mathbf{x}_i^\top \mathbf{W}_V \in \mathbb{R}^{1 \times d_v} \quad (\text{Token } i \text{ 的值向量})
$$

<figure>
<pre>
                      原始输入 Token 向量 x_i  [1 × d_model]
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   W_Q [d_model × d_k]       W_K [d_model × d_k]       W_V [d_model × d_v]
         │                         │                         │
         ▼                         ▼                         ▼
   查询向量 q_i [1 × d_k]     键向量 k_i [1 × d_k]      值向量 v_i [1 × d_v]
   "我想寻找什么？"           "我有什么特征可供检索？"    "如果我被选中，我提供什么？"
</pre>
<figcaption><strong>图 5.3：</strong> 单个 Token 输入向量通过三个独立的投影矩阵，同时分化为三种不同职责的角色向量。</figcaption>
</figure>

每个分化出的向量承担着极其精确的分工：
- <dfn id="def-query-vector"><strong>查询向量 $\mathbf{q}_i$</strong></dfn>：当前 Token 向全宇宙发射的**“检索探针”**。它在问：“为了理解我现在的语境，我需要哪些特定的线索？”
- <dfn id="def-key-vector"><strong>键向量 $\mathbf{k}_j$</strong></dfn>：第 $j$ 个 Token 面向外部世界的**“身份名片”**。它在公示：“我拥有这些语法属性、语义类型和实体标签，欢迎比对！”
- <dfn id="def-value-vector"><strong>值向量 $\mathbf{v}_j$</strong></dfn>：第 $j$ 个 Token 蕴含的**“有效载荷（Payload）”**。它代表了：“一旦你确定我很重要，这就是我贡献给你融入新表示的真实知识编码。”

---

### 3. 第一性原理深潜：为什么必须设立三套独立的矩阵？（直接用原始向量 $\mathbf{X}$ 会发生什么灾难？）

许多初学 Transformer 的人都会产生一个本能的疑问：
> *“既然每个词原本就已经有一个维度为 $d_{\text{model}}$ 的特征向量 $\mathbf{x}$ 了，为什么不能直接用原始向量彼此点乘（即令 $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$）？为什么一定要花费三倍的参数量去训练 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$？”*

让我们从几何与代数的第一性原理出发，推导如果我们强行令 $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$ 会爆发的**三大致命数学灾难**：

#### 灾难 1：点积的对称性陷阱（The Symmetry Trap）
向量的点积运算在代数上具有天然的交换律：

$$
\mathbf{x}_i \cdot \mathbf{x}_j = \mathbf{x}_j \cdot \mathbf{x}_i
$$

如果直接使用原始向量计算相关性，那么 **Token $i$ 对 Token $j$ 的注意力强度，必然百分之百严格等于 Token $j$ 对 Token $i$ 的注意力强度**！

但这完全违背了人类自然语言的逻辑本质：
- 在句子 *“小猫（Token $A$）追赶（Token $B$）大老鼠（Token $C$）”* 中：
  - 动词 *“追赶”* 极度渴求找到它的主语 *“小猫”* 和宾语 *“老鼠”*（需要分配高达 90% 的注意力）；
  - 但名词 *“小猫”* 在表达自身实体时，对动词 *“追赶”* 的依赖程度往往要弱得多；
- 在英语修饰结构 *“extremely（极其）beautiful（美丽）”* 中：
  - 形容词 *“beautiful”* 必须深刻依赖副词 *“extremely”* 来确定修饰程度；
  - 但副词本身只是一个轻量修饰符，并不需要吸收整个形容词的复杂实体特征。

**语言的关系是高度有向、高度不对称的（Asymmetric & Directed）**。
通过引入独立的 $\mathbf{W}_Q$ 和 $\mathbf{W}_K$，相关度计算变成了：

$$
\text{Score}(i \to j) = \mathbf{q}_i^\top \mathbf{k}_j = (\mathbf{x}_i^\top \mathbf{W}_Q)(\mathbf{x}_j^\top \mathbf{W}_K)^\top = \mathbf{x}_i^\top (\mathbf{W}_Q \mathbf{W}_K^\top) \mathbf{x}_j
$$

由于矩阵相乘通常不满足对称性（$\mathbf{W}_Q \mathbf{W}_K^\top \ne \mathbf{W}_K \mathbf{W}_Q^\top$），这便彻底**打破了对称性诅咒**，使模型能够精确建模 $i$ 关注 $j$ 与 $j$ 关注 $i$ 截然不同的有向依赖！

#### 灾难 2：自恋狂自注意力陷阱（The Self-Absorption Bias）
在线性代数中，任何非零向量与自身的点积，等于其模长的平方（即欧几里得范数的平方）：

$$
\mathbf{x}_i \cdot \mathbf{x}_i = \|\mathbf{x}_i\|^2 = \sum_{k=1}^d x_{ik}^2 > 0
$$

根据柯西-施瓦茨不等式（Cauchy-Schwarz Inequality）：

$$
|\mathbf{x}_i \cdot \mathbf{x}_j| \le \|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|
$$

如果两个词向量的模长相仿，那么**一个词与自己的点积，几乎永远大于它与其他任何不同词的点积**！
- 如果没有 $\mathbf{W}_Q$ 和 $\mathbf{W}_K$ 的转置投影，每个词计算出的自相关分数 $\mathbf{x}_i \cdot \mathbf{x}_i$ 会以压倒性优势碾压周围的所有词；
- 经过 Softmax 归一化后，每个词将分配 99% 的注意力给自己，变成目中无人的“超级自恋狂”，对上下文的信息视而不见！
- 引入 $\mathbf{W}_Q$ 和 $\mathbf{W}_K$ 后，$\mathbf{q}_i$ 与 $\mathbf{k}_i$ 被映射到了不同的空间：$\mathbf{q}_i^\top \mathbf{k}_i = \mathbf{x}_i^\top (\mathbf{W}_Q \mathbf{W}_K^\top) \mathbf{x}_i$。只要 $\mathbf{W}_Q \mathbf{W}_K^\top$ 不是正定单位阵，词语对自己计算出的相关度就完全可以低于对上下文关键线索词的相关度，迫使模型将目光投向浩瀚的上下文！

#### 灾难 3：寻址模式与内容载荷的冲突（Address vs. Payload Conflict）
在计算机体系结构与数据库设计中，这是早已被奉为圭臬的准则：**内存地址（Address/Key）绝不等于内存数据（Data/Value）**。
- **寻址空间（Query & Key）**：关注的是“模式、关系与匹配标签”。例如：“我是一个需要动词的助动词”、“我是一个阴性单数名词”。这些是用来建立几何几何关联的控制流信号；
- **载荷空间（Value）**：关注的是“实体、概念与知识语义”。例如：“这是一只体重 4 公斤、花斑色、正在打瞌睡的折耳猫”。
- 如果强令 $\mathbf{V} = \mathbf{X}$，那么每次注意力更新都会把“控制检索特征”和“知识语义特征”硬性掺杂在一起，导致词向量在网络层层递进时产生严重的信息污染与表征退化。
- 独立的 $\mathbf{W}_V$ 赋予了模型终极的自由度：**哪怕 Key 匹配得分完全相同，模型也可以决定向后传递完全不同维度的有效载荷！**

---

### 4. 严谨数学变量与维度对照表

<details name="ch05-specs" open>
<summary><strong>点击展开：Q、K、V 完整数学符号与维度速查手册</strong></summary>

<dl>
  <dt><strong>$\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$</strong></dt>
  <dd>输入序列经过词嵌入和位置编码后的特征矩阵，包含 $T$ 个词，每个词维度为 $d_{\text{model}}$。</dd>

  <dt><strong>$T \in \mathbb{N}^+$</strong></dt>
  <dd>当前输入序列的时间步长度（Sequence Length / Context Window），例如 2048、4096 或 8192。</dd>

  <dt><strong>$d_{\text{model}} \in \mathbb{N}^+$</strong></dt>
  <dd>Transformer 主干隐藏层通道维度（如 4096）。</dd>

  <dt><strong>$d_k \in \mathbb{N}^+$</strong></dt>
  <dd>查询（Query）与键（Key）的子空间维度。为了能够执行点积 $\mathbf{q}_i^\top \mathbf{k}_j$，$\mathbf{Q}$ 和 $\mathbf{K}$ 的列维度必须严格相等。</dd>

  <dt><strong>$d_v \in \mathbb{N}^+$</strong></dt>
  <dd>值（Value）的子空间维度。数学上 $d_v$ 可以不等于 $d_k$，但在现代大模型工程实现中通常设 $d_v = d_k$。</dd>

  <dt><strong>$\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  <dd>查询投影权重矩阵（Query Projection Weights）。</dd>

  <dt><strong>$\mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  <dd>键投影权重矩阵（Key Projection Weights）。</dd>

  <dt><strong>$\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$</strong></dt>
  <dd>值投影权重矩阵（Value Projection Weights）。</dd>

  <dt><strong>$\mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{T \times T}$</strong></dt>
  <dd>未归一化的原始注意力关联矩阵（Raw Attention Logits）。第 $(i, j)$ 项表示第 $i$ 个 Token 对第 $j$ 个 Token 的原始匹配亲和度。</dd>
</dl>

</details>

---

## 第 4 步：历史渊源与技术演进（Where Did It Come From?） {: #step-4 }

人类是如何一步一步从僵硬的信息检索系统，演变出今天统治全球 AI 的 Q、K、V 注意力机制的？

<dl>
  <dt><time datetime="1970">1970</time> &mdash; <strong>Edgar F. Codd</strong>：关系型数据库与键值检索（Key-Value Store）</dt>
  <dd>
    图灵奖得主 Codd 奠定了现代数据库理论。在经典数据库中，检索是<strong>硬性二值离散的（Hard Discrete Matching）</strong>：给定一个明确的查询（<code>SELECT * WHERE Key = 'Bank'</code>），系统通过 B+ 树或哈希表精准抓取唯一的 Value。
    <br>
    <strong>为什么它不能直接用于神经网络？</strong> 离散的 <code>if Key == Query</code> 是不可导的阶梯函数，导数处处为零，梯度无法反向传播！
  </dd>

  <dt><time datetime="2014">2014</time> &mdash; <strong>Dzmitry Bahdanau, Kyunghyun Cho &amp; Yoshua Bengio</strong>：软对齐注意力机制的诞生</dt>
  <dd>
    在机器翻译的 RNN 瓶颈期，Bengio 团队发表了里程碑论文《通过联合学习对齐与翻译的神经机器翻译》。他们首次提出了<strong>连续可微的软寻址（Soft Addressing）</strong>：不再硬性挑选某一个词，而是给所有输入词赋予 0 到 1 之间的概率权重。
    <br>
    <strong>当时的局限：</strong> 解码器隐状态同时充当 Query，编码器隐状态直接充当 Key 和 Value，尚未建立角色分离的概念，计算严重绑定在单向循环的 RNN 链条中。
    <cite>"Neural Machine Translation by Jointly Learning to Align and Translate", ICLR 2015</cite>.
  </dd>

  <dt><time datetime="2015">2015</time> &mdash; <strong>Minh-Thang Luong, Hieu Pham &amp; Christopher D. Manning</strong>：点积注意力范式</dt>
  <dd>
    斯坦福大学 Manning 团队简化了 Bahdanau 复杂的加法多层感知机对齐评分，提出了极速的<strong>点积打分机制（Dot-Product Attention）</strong>：$\text{Score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^\top \mathbf{h}$，为后续的大规模矩阵化并行铺平了道路。
  </dd>

  <dt><time datetime="2017">2017</time> &mdash; <strong>Ashish Vaswani 等 8 位科学家（Google Brain &amp; Google Research）</strong>：Q、K、V 三位一体自注意力</dt>
  <dd>
    在传世论文《Attention Is All You Need》中，作者们彻底终结了循环神经网络（RNN）和卷积网络（CNN）在自然语言处理中的统治地位。他们正式借用数据库哲理，创立了由 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ 三个线性投影构成的<strong>自注意力机制（Self-Attention）</strong>。
    <br>
    <strong>为什么选择纯线性投影，而不是非线性 MLP？</strong>
    在数十层的大模型中，保持注意力寻址的高速至关重要。矩阵乘法在英伟达 GPU Tensor Core 上拥有无与伦比的计算吞吐量（每秒数百 TFLOPs）。非线性表达能力则由后续的 Softmax 归一化和 FFN/SwiGLU 模块充沛提供，形成了完美的架构解耦。
    <cite>"Attention Is All You Need", NeurIPS 2017</cite>.
  </dd>
</dl>

---

## 第 5 步：手算极简数值示例（Concrete Toy Example） {: #step-5 }

为了让你能够彻头彻尾验算每一个数字，我们设计一个**可以用草稿纸和笔完全手算验证的微型注意力玩具实验**。

### 实验设定
- 句子包含 3 个 Token：
  - $\text{Token}_1 =$ <kbd>"The"</kbd>
  - $\text{Token}_2 =$ <kbd>"river"</kbd>
  - $\text{Token}_3 =$ <kbd>"bank"</kbd>
- 模型隐藏层主维度设为极小值：$d_{\text{model}} = 4$；
- 注意力子空间维度设为：$d_k = 2, d_v = 2$。

---

### 1. 输入特征矩阵 $\mathbf{X} \in \mathbb{R}^{3 \times 4}$

假设经过词嵌入后，3 个词的 4 维特征行向量如下：

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1^\top \\
\mathbf{x}_2^\top \\
\mathbf{x}_3^\top
\end{bmatrix} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 2.0 & 0.0 & 1.0 \\
1.0 & 1.0 & 0.0 & 2.0
\end{bmatrix}
$$

其中：
- $\mathbf{x}_1 = [1, 0, 1, 0]$（虚词 "The"）
- $\mathbf{x}_2 = [0, 2, 0, 1]$（水利词 "river"）
- $\mathbf{x}_3 = [1, 1, 0, 2]$（待明确语境的多义词 "bank"）

---

### 2. 设定三个投影矩阵 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{4 \times 2}$

为了让算术过程一目了然，我们精心设计包含简单整数与小数的投影参数：

$$
\mathbf{W}_Q = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 1 \\
1 & 0
\end{bmatrix}, \quad
\mathbf{W}_K = \begin{bmatrix}
0 & 1 \\
1 & 0 \\
1 & 0 \\
0 & 1
\end{bmatrix}, \quad
\mathbf{W}_V = \begin{bmatrix}
1 & 0 \\
0 & 0 \\
0 & 2 \\
0 & 1
\end{bmatrix}
$$

---

### 3. 一步一步手算投影结果

#### 计算查询矩阵 $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{3 \times 2}$

对于第 1 行 $\mathbf{q}_1^\top = [1, 0, 1, 0] \mathbf{W}_Q$：
- 第 1 列：$1\times 1 + 0\times 0 + 1\times 0 + 0\times 1 = 1$
- 第 2 列：$1\times 0 + 0\times 1 + 1\times 1 + 0\times 0 = 1$
- $\mathbf{q}_1^\top = [1, 1]$

对于第 2 行 $\mathbf{q}_2^\top = [0, 2, 0, 1] \mathbf{W}_Q$：
- 第 1 列：$0\times 1 + 2\times 0 + 0\times 0 + 1\times 1 = 1$
- 第 2 列：$0\times 0 + 2\times 1 + 0\times 1 + 1\times 0 = 2$
- $\mathbf{q}_2^\top = [1, 2]$

对于第 3 行 $\mathbf{q}_3^\top = [1, 1, 0, 2] \mathbf{W}_Q$（关注多义词 `"bank"` 的查询意图！）：
- 第 1 列：$1\times 1 + 1\times 0 + 0\times 0 + 2\times 1 = 1 + 2 = 3$
- 第 2 列：$1\times 0 + 1\times 1 + 0\times 1 + 2\times 0 = 1$
- $\mathbf{q}_3^\top = [3, 1]$

汇总得出完整查询矩阵：

$$
\mathbf{Q} = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
3 & 1
\end{bmatrix}
$$

---

#### 计算键矩阵 $\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{3 \times 2}$

对于第 1 行 $\mathbf{k}_1^\top = [1, 0, 1, 0] \mathbf{W}_K$：
- 第 1 列：$1\times 0 + 0\times 1 + 1\times 1 + 0\times 0 = 1$
- 第 2 列：$1\times 1 + 0\times 0 + 1\times 0 + 0\times 1 = 1$
- $\mathbf{k}_1^\top = [1, 1]$

对于第 2 行 $\mathbf{k}_2^\top = [0, 2, 0, 1] \mathbf{W}_K$（词语 `"river"` 对外亮出的身份标签）：
- 第 1 列：$0\times 0 + 2\times 1 + 0\times 1 + 1\times 0 = 2$
- 第 2 列：$0\times 1 + 2\times 0 + 0\times 0 + 1\times 1 = 1$
- $\mathbf{k}_2^\top = [2, 1]$

对于第 3 行 $\mathbf{k}_3^\top = [1, 1, 0, 2] \mathbf{W}_K$：
- 第 1 列：$1\times 0 + 1\times 1 + 0\times 1 + 2\times 0 = 1$
- 第 2 列：$1\times 1 + 1\times 0 + 0\times 0 + 2\times 1 = 1 + 2 = 3$
- $\mathbf{k}_3^\top = [1, 3]$

汇总得出完整键矩阵：

$$
\mathbf{K} = \begin{bmatrix}
1 & 1 \\
2 & 1 \\
1 & 3
\end{bmatrix}
$$

---

#### 计算值矩阵 $\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{3 \times 2}$

对于第 1 行 $\mathbf{v}_1^\top = [1, 0, 1, 0] \mathbf{W}_V$：
- $\mathbf{v}_1^\top = [1\times 1 + 0 + 0 + 0, 0 + 0 + 1\times 2 + 0] = [1, 2]$

对于第 2 行 $\mathbf{v}_2^\top = [0, 2, 0, 1] \mathbf{W}_V$：
- $\mathbf{v}_2^\top = [0 + 0 + 0 + 0, 0 + 0 + 0 + 1\times 1] = [0, 1]$

对于第 3 行 $\mathbf{v}_3^\top = [1, 1, 0, 2] \mathbf{W}_V$：
- $\mathbf{v}_3^\top = [1\times 1 + 0 + 0 + 0, 0 + 0 + 0 + 2\times 1] = [1, 2]$

汇总得出完整值矩阵：

$$
\mathbf{V} = \begin{bmatrix}
1 & 2 \\
0 & 1 \\
1 & 2
\end{bmatrix}
$$

---

### 4. 关键高潮：手算原始注意力关联矩阵 $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{3 \times 3}$

现在，让我们见证奇迹时刻！
我们计算 $\mathbf{Q}$ 的每一行（各词的心愿）与 $\mathbf{K}^\top$ 的每一列（各词的标签）的点积：

$$
\mathbf{S} = \mathbf{Q} \mathbf{K}^\top = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
3 & 1
\end{bmatrix} \begin{bmatrix}
1 & 2 & 1 \\
1 & 1 & 3
\end{bmatrix}
$$

逐项相乘求和：
- 第 1 行（"The" 的注意力倾向）：
  - $S_{11} = 1\times 1 + 1\times 1 = 2$
  - $S_{12} = 1\times 2 + 1\times 1 = 3$
  - $S_{13} = 1\times 1 + 1\times 3 = 4$
- 第 2 行（"river" 的注意力倾向）：
  - $S_{21} = 1\times 1 + 2\times 1 = 3$
  - $S_{22} = 1\times 2 + 2\times 1 = 4$
  - $S_{23} = 1\times 1 + 2\times 3 = 7$
- 第 3 行（多义词 <mark>"bank"</mark> 的主动匹配结果！）：
  - $S_{31} = \mathbf{q}_3 \cdot \mathbf{k}_1 = 3\times 1 + 1\times 1 = 4$ （对虚词 "The" 的匹配分）
  - $S_{32} = \mathbf{q}_3 \cdot \mathbf{k}_2 = 3\times 2 + 1\times 1 = \mathbf{7}$ （对水利词 <mark>"river"</mark> 的匹配分！）
  - $S_{33} = \mathbf{q}_3 \cdot \mathbf{k}_3 = 3\times 1 + 1\times 3 = 6$ （对自身的自相关分）

$$
\mathbf{S} = \begin{bmatrix}
2 & 3 & 4 \\
3 & 4 & 7 \\
4 & \mathbf{7} & 6
\end{bmatrix}
$$

---

### 5. 结果透视与物理检验

请起立为第 3 行的结果鼓掌：
对于多义词 `"bank"`（第 3 行），它的原始注意力得分为：
- 对 `"The"`：得分 $4$
- 对自身 `"bank"`：得分 $6$
- 对 `"river"`：得分 **$7$**！

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 5.1：</strong> 多义词 "bank" 对上下文中各个词的注意力匹配量化表</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">目标词 (Key)</th>
      <th scope="col" align="center" width="25%">点积计算过程 $\mathbf{q}_3 \cdot \mathbf{k}_j$</th>
      <th scope="col" align="center" width="20%">原始匹配得分</th>
      <th scope="col" align="left" width="35%">直观语义解释</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><kbd>"The"</kbd></th>
      <td align="center">$3\times 1 + 1\times 1 = 4$</td>
      <td align="center">4.0</td>
      <td>基础冠词，提供极少上下文营养</td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"river"</kbd></th>
      <td align="center">$3\times 2 + 1\times 1 = \mathbf{7}$</td>
      <td align="center"><mark><strong>7.0</strong></mark></td>
      <td><strong>核心线索词！成功识别出“水利河流”语境！</strong></td>
    </tr>
    <tr>
      <th scope="row" align="left"><kbd>"bank"</kbd></th>
      <td align="center">$3\times 1 + 1\times 3 = 6$</td>
      <td align="center">6.0</td>
      <td>自我指涉，保留自身原有特征</td>
    </tr>
  </tbody>
</table>

可视化匹配强度仪表盘：
- 对 `"river"` 的关注度：
  <meter min="0" max="10" low="3" high="6" optimum="8" value="7.0">7.0 / 10</meter> (最高权重，决定将“河岸”含义注入自身)
- 对自身的关注度：
  <meter min="0" max="10" low="3" high="6" optimum="8" value="6.0">6.0 / 10</meter> (次高权重)
- 对 `"The"` 的关注度：
  <meter min="0" max="10" low="3" high="6" optimum="8" value="4.0">4.0 / 10</meter> (最低权重)

通过投影矩阵 $\mathbf{W}_Q$ 和 $\mathbf{W}_K$ 的协同，多义词 `"bank"` 成功跨越了自恋陷阱，把全场最高的注意力精准投射到了决定其真正语义的关键词 `"river"` 上！

在接下来的第 07 章中，我们将学习如何通过 **Softmax** 把这组得分 $[4, 7, 6]$ 转化为严格相加等于 100% 的概率权重，并最终加权融合 **Value 向量**。

---

## 第 6 步：核心精髓总结（Core Takeaway） {: #step-6 }

<fieldset>
<legend><strong>本章核心记忆卡片</strong></legend>
<p>
<strong>Query、Key、Value 是让静态词向量焕发动态智慧的三重人格分化：</strong><br>
原始词嵌入矩阵 $\mathbf{X}$ 是静态死板的；通过三个投影矩阵 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$，每个 Token 同时拥有了<strong>寻找线索的探针（Query）</strong>、<strong>展示身份的招牌（Key）</strong>以及<strong>提供知识的载荷（Value）</strong>。它们不仅彻底打破了点积的对称性与自恋诅咒，更为大模型动态重构词义、理解复杂人类语境提供了最坚固的线性代数底座。
</p>
</fieldset>
