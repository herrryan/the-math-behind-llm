# 第 05 章：大模型的全景图纸（Transformer 架构宏观巡览与圆桌会议）

<nav aria-label="目录导航">
  <p>
    <strong>目录导航：</strong> 
    <a href="#step-1">1. 3岁小孩直觉</a> &bull; 
    <a href="#step-2">2. 计算跨越的桥梁问题</a> &bull; 
    <a href="#step-3">3. 严谨数学公式与架构推导</a> &bull; 
    <a href="#step-4">4. 历史渊源与技术演进</a> &bull; 
    <a href="#step-5">5. 手算极简数值示例</a> &bull; 
    <a href="#step-6">6. 核心精髓总结</a>
  </p>
</nav>

---

<h2 id="step-1">第 1 步：3 岁小孩的直觉（3-Year-Old Intuition）</h2>

在上一章的实战工坊（Lab 01）中，我们用 80 行纯 Python 代码从零训练出了属于自己的首个神经网络大脑。它不仅学会了基础语法，还生成了有模有样的句子！

但与此同时，我们也目睹了它的致命软肋：**不可救药的严重短时失忆症**。因为它每次只能低头看紧挨着的前 1 个词，一旦陷入局部概率，就会陷入生成 <samp>"the rug the rug"</samp> 的死循环。

为了理解现代大模型是如何攻克这一难题的，让我们看看人类历史上让计算机处理文本序列的三种截然不同的方式：

<figure>
<pre>
1. 戴眼罩的马（Bengio 2003 固定窗口 MLP）：
   [词 1] [词 2] ... [词 98] | [词 99] ===&gt; [预测第 100 个词]
                             └─ 视野被眼罩遮蔽，对更早的一切完全失忆！

2. 悄悄话接力游戏（循环神经网络 RNN / LSTM）：
   [小朋友 1] ──耳语传话──&gt; [小朋友 2] ──耳语传话──&gt; ... ──耳语──&gt; [小朋友 100]
   “一只小狗...”           “一只小狗？”                           “一只土豆！”
   （记忆在漫长的传递链条中不断失真、稀释、消散）

3. 宏伟的圆桌会议（Transformer 架构）：
   ┌─────────────────────────────────────────────────────────────┐
   │                         小朋友 1                            │
   │                       （“那只小狗”）                        │
   │                          ▲         ▲                        │
   │       瞬间眼神交流       │         │ 瞬间眼神交流           │
   │                          ▼         ▼                        │
   │   小朋友 25 ◄──────────► 小朋友 50 ◄───────────► 小朋友 100 │
   │   （“毛茸茸”）           （“吠叫”）             （“大声地”）│
   └─────────────────────────────────────────────────────────────┘
   所有小朋友同时围坐在圆桌四周！
   任何两个人之间只需 0 秒就能完成毫无阻隔的直接对视。
</pre>
<figcaption><strong>图 5.1:</strong> 语言模型序列处理的三代演进：戴眼罩的固定窗口、悄悄话单列接力，以及所有词全平权落座的圆桌会议。</figcaption>
</figure>

### 比喻 1：戴眼罩的马（固定窗口前馈网络 MLP）
在 Lab 01 的模型中（基于 Bengio 2003 的经典设计），神经网络就像一匹双眼两侧扣着厚厚皮革眼罩的马。它唯一的视野就是眼前的一小步。

如果我们想让它看清过去的 10 个词，就必须把 10 个词的向量硬拼在一起，使得第一层权重矩阵的体积膨胀 10 倍。如果想要读完一整本包含 10 万个词的长篇小说，权重矩阵里需要填满数万亿个参数，哪怕计算一次都会瞬间耗尽整台超级计算机的内存！固定窗口在长文面前遭遇了根本性的不可扩展性。

### 比喻 2：悄悄话接力游戏（循环神经网络 RNN 与 LSTM）
在 2017 年之前，人工智能学界试图通过**循环神经网络（<abbr title="Recurrent Neural Network">RNN</abbr>）**与**长短期记忆网络（<abbr title="Long Short-Term Memory">LSTM</abbr>）**打破眼罩枷锁。

想象 100 个小朋友排成一条长长的单列纵队：
- 排在第 1 位的小朋友听到一句悄悄话：<samp>“一只长着毛茸茸耳朵的金毛小狗昨天在巴黎走失了。”</samp>
- 第 1 位小朋友在手里的小口袋笔记本上匆匆记下摘要（隐藏状态向量 $\mathbf{h}_1$），递给第 2 位小朋友；
- 第 2 位小朋友看一眼笔记，加上自己的理解，擦改笔记本，再悄悄传给第 3 位……
- 当这本笔记终于传递到第 100 位小朋友手里时，纸张早已被橡皮擦烂、被墨水浸污。最后一位小朋友听到的是什么？<samp>“一只土豆在巴黎走失了。”</samp>

这个模式暴露了两个致命的结构性死穴：
1. **信息压缩瓶颈（The Information Bottleneck）**：试图把成百上千个词的复杂语义，硬塞进一个固定长度的连续向量 $\mathbf{h}_t$ 中，久远的历史细节必然被不可逆地抹去（梯度消失与长程遗忘）。
2. **串行计算地狱（The Sequential Prison）**：排在第 50 位的小朋友**绝不能提前开工**，必须苦苦等待前面 49 个人一个接一个传完！尽管现代图形芯片（<abbr title="Graphics Processing Unit">GPU</abbr>）拥有数万个能够同时运转的高速计算核心，却只能大部分处于闲置饥饿状态，被迫排队做串行等待。

### 比喻 3：宏伟的圆桌会议（Transformer 架构）
2017 年，Google 的科学家们发表了一篇震撼整个人工智能史的划时代论文，标题极其激进自信：<cite>《Attention Is All You Need》（注意力就是你所需要的一切）</cite>。

他们彻底推翻了排队传话的单列纵队，把文章里的所有词一起请进了一座**宏伟的圆桌会议大厅**：
- **全员齐聚**：不管是第 1 个词还是第 1000 个词，在第 0 秒那一刻，全部同时围坐在圆桌四周；
- **全向对视（自注意力机制 Self-Attention）**：第 800 个词（代词 <kbd>"it"</kbd>）需要搞清楚自己指代谁时，根本不需要通过几百个人的层层转述。它只要抬起头，跨过圆桌，瞬间与第 12 个词（名词 <kbd>"puppy"</kbd>）完成眼神对视！
- **天生并行**：因为所有词全部就座，GPU 可以调动全部成千上万个算力核心，一次性并行计算出所有人与所有人之间的目光联系。

---

<h2 id="step-2">第 2 步：计算跨越的桥梁问题（The Bridging Question）</h2>

<fieldset>
<legend><strong>计算层面的核心挑战</strong></legend>
<p>
我们如何把这种“圆桌会议”转化为计算机与 GPU 能够精确计算的高效矩阵代数？
</p>
<p>
如果 1000 个词同时坐在大厅里大喊大叫，整个会议室岂不会陷入混乱不堪的杂音风暴？每个词怎么知道<em>应该专注倾听谁的发言</em>、<em>怎样吸收别人的信息</em>，又该<em>怎样在私下消化思考</em>并最终生成合乎语法的下一个词？
</p>
</fieldset>

为了把这一诗意的比喻变成现实，现代大语言模型设计了一套严丝合缝的交替节奏：
1. **交流阶段（自注意力机制 Self-Attention）**：词与词之间环顾圆桌、交换便签，把与自己相关的语境线索收集进来；
2. **思考阶段（前馈神经网络 Feed-Forward Network / SwiGLU）**：每个词退回自己的私人办公室，结合刚刚交流获得的新情报，检索深层事实记忆，独立提炼升级自己的语义。

让我们正式拆解这幅掌控着万亿参数模型的宏观建筑全景图。

---

<h2 id="step-3">第 3 步：严谨数学公式与架构推导（The Exact Math & Architecture）</h2>

当今所有顶尖大语言模型——包括 **GPT-4**、**LLaMA-3**、**Mistral**、**Gemma**、**Claude**、**DeepSeek-V2/V3** 以及 **Qwen-2.5**——都基于同一种主流范式构建：**仅包含解码器的自回归 Transformer（Decoder-Only Autoregressive Transformer）**。

整个模型在本质上是一条高维张量变换管道，将离散的整数 Token ID 映射为下一个词的概率分布：

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER 宏观数据流全景总线                     │
└────────────────────────────────────────────────────────────────────────┘

 [输入文本 prompt]: "The cat sat on the"
       │
       ▼ (分词器 Tokenizer)
 [Token IDs 整数序列]: [464, 3797, 3334, 319, 262]  (序列长度 T = 5)
       │
       ▼ (嵌入矩阵 E 查表 + 位置编码 P)
 [输入张量 X_0]  ───► 维度形状: [T × d_model]
       │
       ├───────────────────────────────────────────────────────┐
       ▼                                                       │
 ┌─────────────────────────────────────────────────────────┐   │
 │              第 1 层 TRANSFORMER 块 (Layer 1)           │   │
 │                                                         │   │
 │   X_in ──► [RMSNorm 归一化] ──► [因果自注意力机制] ──┐  │   │
 │     │                                                │  │   │
 │     └──────────────── (残差跳跃连接 +) ◄─────────────┘  │   │
 │                            │                            │   │
 │                            ▼ H_1                        │   │
 │     H_1 ──► [RMSNorm 归一化] ──► [SwiGLU 前馈网络] ──┐  │   │
 │     │                                                │  │   │
 │     └──────────────── (残差跳跃连接 +) ◄─────────────┘  │   │
 │                            │                            │   │
 └────────────────────────────┼────────────────────────────┘   │
                              ▼ X_1                            │ 重复堆叠
                              │                                │ 贯穿 L 层
                             ... (垂直重复堆叠 L 次)           │ 架构！
                              │                                │
 ┌────────────────────────────┼────────────────────────────┐   │
 │              第 L 层 TRANSFORMER 块 (Layer L)           │   │
 │                            │                            │   │
 └────────────────────────────┼────────────────────────────┘   │
                              ▼ X_L ◄──────────────────────────┘
                              │
                       [最终 RMSNorm 归一化]
                              │
                              ▼ X_final ───► 维度形状: [T × d_model]
                              │
             [输出解嵌入矩阵 E_U] ──► 维度形状: [d_model × |V|]
                              │
                              ▼
                       [未归一化对数几率 Z]  ───► 维度形状: [T × |V|]
                              │
                      [Softmax 归一化指数函数]
                              │
                              ▼
                        [词表概率分布 P]
             P("rug" | "The cat sat on the") = 78.4%
</pre>
<figcaption><strong>图 5.2:</strong> 现代 Decoder-Only Transformer 宏观计算全景管线。从底部的离散 Token 经由嵌入层进入残差主干道，历经 L 层“交流（Attention）与思考（FFN）”交替升级，最终在解嵌入投影层映射回词表完成概率预测。</figcaption>
</figure>

让我们从第一性原理出发，逐一推导其中的每一个数学要素。

### 1. 架构五大核心超参数（The Architectural Hyperparameters）

任何一个工业级大模型的整体骨架，完全由以下 5 个基础标量维度所统领：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">数学符号</th>
      <th align="left">官方术语</th>
      <th align="left">真实工业级典范（LLaMA-3-8B）</th>
      <th align="left">物理直觉与工程含义</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$|V|$</td>
      <td>词表大小（Vocabulary Size）</td>
      <td>$128,256$</td>
      <td>模型所能认识的离散子词（Subwords）总数。</td>
    </tr>
    <tr>
      <td>$T$</td>
      <td>上下文窗口长度（Context Length）</td>
      <td>$8,192$（可扩展至 $128\text{k}$）</td>
      <td>圆桌会议能够同时容纳落座的最大 Token 数量。</td>
    </tr>
    <tr>
      <td>$d_{\text{model}}$</td>
      <td>模型特征隐藏维度（Hidden Dim）</td>
      <td>$4,096$</td>
      <td>承载每个词丰富语义的连续几何向量宽度（残差流主干道宽度）。</td>
    </tr>
    <tr>
      <td>$L$</td>
      <td>网络层数（Number of Layers）</td>
      <td>$32$</td>
      <td>垂直堆叠的 Transformer 计算块数量。</td>
    </tr>
    <tr>
      <td>$d_{\text{ffn}}$</td>
      <td>前馈网络隐藏维度（FFN Hidden Dim）</td>
      <td>$14,336$（约为 $\frac{8}{3} d_{\text{model}}$）</td>
      <td>每个词进入私人思考室后，进行特征升维与深层记忆检索的通道宽度。</td>
    </tr>
  </tbody>
</table>

### 2. 阶段 1：序列输入表征（Input Representation）

给定用户输入的提示词文本，经分词器切分为由 $T$ 个整数构成的序列：

$$
\mathbf{w} = \begin{bmatrix} w_1 & w_2 & \dots & w_T \end{bmatrix}^\top \in \{1, \dots, |V|\}^T
$$

通过词嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$（我们在第 01 章已推导），每个词根据其编号索取专属的连续特征行向量，并附加上位置信息 $\mathbf{p}_t$（第 10 章详解）：

$$
\mathbf{x}_t^{(0)} = \mathbf{e}_{w_t}^\top \mathbf{E} + \mathbf{p}_t \in \mathbb{R}^{1 \times d_{\text{model}}}
$$

将当前序列的全部 $T$ 个行向量垂直堆叠，便铸就了贯穿整个模型的**输入张量**：

$$
\mathbf{X}^{(0)} = \begin{bmatrix}
\mathbf{x}_1^{(0)} \\
\mathbf{x}_2^{(0)} \\
\vdots \\
\mathbf{x}_T^{(0)}
\end{bmatrix} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

### 3. 阶段 2：Transformer 基础计算块（重复堆叠 $L$ 次）

张量 $\mathbf{X}^{(0)}$ 随后注入由 $L$ 个串联模块组成的深层网络中（$l = 1, 2, \dots, L$）。

每个块内部严格遵循由**残差流（Residual Stream）**串联的两级子架构：

#### 子层 A：全员交流室（自注意力机制 Self-Attention）
每个词环视圆桌，检索与自己最相关的上下文线索：

$$
\mathbf{H}^{(l)} = \mathbf{X}^{(l-1)} + \operatorname{SelfAttention}\left(\operatorname{RMSNorm}(\mathbf{X}^{(l-1)})\right)
$$

- $\operatorname{RMSNorm}(\cdot)$：层归一化操作，用于稳定极深网络的数值方差，防止信号发生指数级爆炸或衰减（第 13 章）；
- $\operatorname{SelfAttention}(\cdot)$：通过查询（Query）、键（Key）和值（Value）计算每对词之间的动态目光权重，实现信息跨词汇流动（第 06～09 章）；
- 核心符号 $+ \mathbf{X}^{(l-1)}$：**残差高速公路（Residual Highway）**（第 12 章）。它确保新交流得来的语境增量是以“追加修正”的方式写在原有表征上，绝不破坏词汇本身的根本特征，同时也让反向传播梯度得以无衰减地直接贯通到底层。

#### 子层 B：闭门思考室（前馈神经网络 FFN / SwiGLU）
各词在吸纳了邻居们提供的新语境后，分别走入各自独立的私人思考室闭门深造：

$$
\mathbf{X}^{(l)} = \mathbf{H}^{(l)} + \operatorname{FFN}\left(\operatorname{RMSNorm}(\mathbf{H}^{(l)})\right)
$$

- 在当今顶尖大模型中，该模块标配为我们在第 04 章深入剖析过的 **SwiGLU** 门控网络：
  
$$
\operatorname{FFN}(\mathbf{h}) = \left(\operatorname{Swish}(\mathbf{h}\mathbf{W}_{\text{gate}}) \odot (\mathbf{h}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}
$$

- **请务必洞察这里的本质分工：**
  - 注意力机制是**横向的（Horizontal）**：负责跨越时间序列维度 $T$，让词与词进行信息交换；
  - 前馈网络是**纵向的（Vertical）**：对每一个词完全独立计算，负责在单个词的特征维度 $d_{\text{model}}$ 上深挖逻辑关联、提取长程事实记忆。

<figure>
<pre>
Token 位置:           t = 1 ("The")         t = 2 ("bank")        t = 3 ("river")
                          │                      │                     │
子层 1 (交流):            ▼                      ▼                     ▼
[自注意力机制]     ◄── 横向跨词交互 ────────── 交流融合 ───────── 横向跨词交互 ──►
                          │                      │                     │
残差累加 (+):             ▼ (+)                  ▼ (+)                 ▼ (+)
                          │                      │                     │
子层 2 (思考):            ▼                      ▼                     ▼
[SwiGLU 前馈网络]  纵向独立深思           纵向独立深思           纵向独立深思
                   (闭门检索记忆)         (闭门检索记忆)         (闭门检索记忆)
                          │                      │                     │
残差累加 (+):             ▼ (+)                  ▼ (+)                 ▼ (+)
                          │                      │                     │
输出进入下一层:          X^(l)_1                X^(l)_2               X^(l)_3
</pre>
<figcaption><strong>图 5.3:</strong> Transformer 块的黄金节奏：注意力机制主导跨时序的横向交互（沟通）；前馈网络主导单词汇的纵向特征映射与知识检索（思考）。</figcaption>
</figure>

### 4. 阶段 3：输出解嵌入与词表概率投影

在完整历经 $L$ 轮高强度的“沟通”与“思考”后，输出张量中的每一个词向量都已被赋予了极度深邃的语境智慧：

$$
\mathbf{X}_{\text{final}} = \operatorname{RMSNorm}(\mathbf{X}^{(L)}) \in \mathbb{R}^{T \times d_{\text{model}}}
$$

为了将这些高维几何语义重新映射回人类可读的文字，模型使用**解嵌入矩阵（Unembedding Matrix）** $\mathbf{E}_U \in \mathbb{R}^{d_{\text{model}} \times |V|}$（在许多模型中与输入嵌入表 $\mathbf{E}$ 共享权重，称为 Weight Tying）：

$$
\mathbf{Z} = \mathbf{X}_{\text{final}} \mathbf{E}_U \in \mathbb{R}^{T \times |V|}
$$

张量 $\mathbf{Z}$ 中的最后一行 $\mathbf{z}_T \in \mathbb{R}^{1 \times |V|}$，代表了基于前 $T$ 个词的全部上下文信息，对紧随其后的**第 $T+1$ 个词**给出的全词表打分（称为对数几率 Logits）。

将其送入 **Softmax 函数**：

$$
P(w_{T+1} = v_i \mid w_{\le T}) = \frac{\exp(z_{T, i})}{\sum_{j=1}^{|V|} \exp(z_{T, j})}
$$

模型在这一概率分布中完成采样，挑选出最具合理性的词，并将其追加到现有序列尾部。紧接着，整个圆桌大厅以长度 $T+1$ 再次启动下一轮推理运算。这，就是现代大语言模型生成如泉涌般长篇大论的**自回归生成（Autoregressive Generation）**法则！

---

<h2 id="step-4">第 4 步：历史渊源与技术演进（Where Did It Come From?）</h2>

Transformer 的诞生并非凭空出现的神迹，而是人类在对抗“失忆”与“计算迟缓”的 15 年征程中，无数先驱思想碰撞出的胜利火花：

<dl>
  <dt><time datetime="2003">2003年</time> &mdash; <strong>Yoshua Bengio 等人</strong>：神经概率语言模型（NPLM）奠基</dt>
  <dd>
    首次证明了用连续向量 $\mathbf{E}$ 表征词义可以让模型学会语义泛化。但其采用的<strong>固定上下文窗口</strong>（如我们在 Lab 01 中所实现）导致其视界极度狭窄，对更早的语境完全盲目。<br>
    <cite>《A Neural Probabilistic Language Model》, JMLR 2003</cite>
  </dd>

  <dt><time datetime="2014">2014年</time> &mdash; <strong>Kyunghyun Cho / Ilya Sutskever 等人</strong>：Seq2Seq 机器翻译时代的辉煌与困境</dt>
  <dd>
    提出了基于 Encoder-Decoder 的循环神经网络架构，理论上允许读取任意长度的句子。然而它撞上了一堵不可逾越的高墙——<strong>固定向量压缩瓶颈</strong>：无论一句英文有 50 个词还是 100 个词，都必须被粗暴地强行压缩成一个长度固定的隐藏向量 $\mathbf{h}$，长句末期的严重失忆现象令翻译系统不堪重负。<br>
    <cite>《Sequence to Sequence Learning with Neural Networks》, NeurIPS 2014</cite>
  </dd>

  <dt><time datetime="2015">2015年</time> &mdash; <strong>Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio</strong>：注意力机制破晓</dt>
  <dd>
    首次提出了<strong>注意力机制（Attention Mechanism）</strong>：放弃将整句话硬塞进单个向量的执念，允许解码端在翻译时，能够主动“回过头看”编码端的<em>每一个历史中间状态</em>并计算加权平均。这一创举瞬间攻克了机器翻译的长句记忆瓶颈。<br>
    <cite>《Neural Machine Translation by Jointly Learning to Align and Translate》, ICLR 2015</cite>
  </dd>

  <dt><time datetime="2017">2017年</time> &mdash; <strong>Ashish Vaswani 等人（Google Brain / Research）</strong>：Attention Is All You Need</dt>
  <dd>
    提出了一个惊世骇俗的猜想：<em>既然注意力机制能够打破距离限制，为什么我们还要保留迟钝且无法并行的 RNN 循环连线？</em><br>
    他们彻底移除了所有循环链条，只依靠纯粹的自注意力网络，缔造了 <strong>Transformer</strong>。所有词之间的交互距离在一瞬间被压平到 $\mathcal{O}(1)$，人类首次在 GPU 上实现了前所未有的超高吞吐并行训练。<br>
    <cite>《Attention Is All You Need》, NeurIPS 2017</cite>
  </dd>

  <dt><time datetime="2018">2018年&ndash;至今</time> &mdash; <strong>Alec Radford 等人（OpenAI）与开源巨浪（Meta LLaMA）</strong>：仅解码器时代的统治</dt>
  <dd>
    OpenAI 敏锐地洞察到：如果核心目标是通用文本生成，原始论文中双边复杂的 Encoder-Decoder 结构实属冗余；仅需保留单向因果堆叠的<strong>纯解码器（Decoder-Only）</strong>架构，通过在大规模无标注文本上做“预测下一个词”，即可孕育出推理、代码编写与世界常识（GPT-1 至 GPT-4）。现代开源霸主 LLaMA、Mistral、DeepSeek 均在此经典架构上演进至今。<br>
    <cite>《Improving Language Understanding by Generative Pre-Training》, OpenAI 2018</cite>
  </dd>
</dl>

---

<h2 id="step-5">第 5 步：手算极简数值示例（Concrete Toy Example）</h2>

让我们用极小的数字，亲手走一遍一个 3 词短句穿越一层迷你 Transformer 的完整计算旅程！

### 场景设定
- 词表 $|V| = 4$：$\{\text{"the"}: 0, \text{"bank"}: 1, \text{"river"}: 2, \text{"flows"}: 3\}$；
- 输入序列长度 $T = 3$：<samp>["the", "bank", "river"]</samp>；
- 模型特征维度 $d_{\text{model}} = 2$；
- 目标：让模型在第 3 个位置，精准预测出第 4 个词是 <samp>"flows"</samp>（流动）。

<fieldset>
<legend><strong>运算跟踪检查清单</strong></legend>
<p><input type="checkbox" checked disabled> <strong>第 1 步：</strong> 查表获取静态词嵌入 $\mathbf{X}^{(0)} \in \mathbb{R}^{3 \times 2}$；</p>
<p><input type="checkbox" checked disabled> <strong>第 2 步：</strong> 自注意力交互（Communication），赋予多义词 "bank" 正确的水文含义；</p>
<p><input type="checkbox" checked disabled> <strong>第 3 步：</strong> 残差累加，保全词汇原有身份特征；</p>
<p><input type="checkbox" checked disabled> <strong>第 4 步：</strong> 前馈网络深思（Thinking），激活物理事实常识；</p>
<p><input type="checkbox" checked disabled> <strong>第 5 步：</strong> 解嵌入投影，计算全词表概率并命中 "flows"。</p>
</fieldset>

### 第 1 步：查表获取静态词嵌入
假设我们的词嵌入表为这 3 个词分配了初始的 2 维几何坐标：

$$
\mathbf{X}^{(0)} = \begin{bmatrix}
\mathbf{x}_1^{(0)} \\
\mathbf{x}_2^{(0)} \\
\mathbf{x}_3^{(0)}
\end{bmatrix} = \begin{bmatrix}
0.2 & 0.1 \\
0.5 & 0.5 \\
0.1 & 0.9
\end{bmatrix} \begin{matrix}
\leftarrow \text{"the"} \\
\leftarrow \text{"bank"（充满歧义：0.5 金融特征，0.5 河岸特征）} \\
\leftarrow \text{"river"（极其明显的水文地理特征：0.9）}
\end{matrix}
$$

请特别观察第 2 个位置的词 <samp>"bank"</samp>：它的向量是 $[0.5, 0.5]$，它既不知道自己是指华尔街的银行，还是水流旁的河岸。

### 第 2 步：自注意力机制的交流阶段（Communication）
在自注意力计算中（我们将在接下来的第 06～08 章详细学习具体公式），位置 2（<samp>"bank"</samp>）环顾圆桌，发现位置 3（<samp>"river"</samp>）蕴含着最核心的上下文线索！

模型计算出的目光分配权重为：给 <samp>"river"</samp> 投射 $0.8$ 的注意力，给自己保留 $0.2$ 的注意力：

$$
\Delta \mathbf{x}_2 = 0.2 \times \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} + 0.8 \times \begin{bmatrix} 0.1 & 0.9 \end{bmatrix} = \begin{bmatrix} 0.10 + 0.08 & 0.10 + 0.72 \end{bmatrix} = \begin{bmatrix} 0.18 & 0.82 \end{bmatrix}
$$

看！增量更新向量 $\Delta \mathbf{x}_2 = [0.18, 0.82]$ 吸收了邻居 <samp>"river"</samp> 身上极为强烈的“水流特征”（$0.82$）！

### 第 3 步：残差高速公路累加
我们将交流阶段获得的新线索 $\Delta \mathbf{x}_2$，通过残差连接与原本的嵌入向量叠加：

$$
\mathbf{h}_2 = \mathbf{x}_2^{(0)} + \Delta \mathbf{x}_2 = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix} + \begin{bmatrix} 0.18 & 0.82 \end{bmatrix} = \begin{bmatrix} 0.68 & 1.32 \end{bmatrix}
$$

残差连接确保了网络既没有忘记自己原本是 <samp>"bank"</samp>（保留了 $0.68$），又成功注入了浓郁的水文特征（激增至 $1.32$）。

### 第 4 步：前馈网络的思考阶段（Thinking）
向量 $\mathbf{h}_2 = [0.68, 1.32]$ 随后进入该词专属的前馈网络（FFN）闭门思考。

前馈网络的神经元权重沉淀着人类世界的事实常识：*“当特征 2 达到高位（$\ge 1.0$）且与 bank 共存时，该实体属于自然流动水体！”*

FFN 激活并给出了提炼后的增量思考，再次经由残差相加：

$$
\mathbf{x}_2^{(1)} = \mathbf{h}_2 + \operatorname{FFN}(\mathbf{h}_2) = \begin{bmatrix} 0.68 & 1.32 \end{bmatrix} + \begin{bmatrix} -0.18 & 0.68 \end{bmatrix} = \begin{bmatrix} 0.50 & 2.00 \end{bmatrix}
$$

此时此刻，该位置的向量彻底脱胎换骨，从一个摇摆不定的歧义词，升华为了一个高度聚焦、坚定不移指向水流地理意义的特征向量 $[0.50, 2.00]$！

### 第 5 步：解嵌入投影与下一个词概率输出
在句子的最终位置，模型将深思熟虑后的最终表征乘以解嵌入矩阵 $\mathbf{E}_U \in \mathbb{R}^{2 \times 4}$：

$$
\mathbf{E}_U = \begin{bmatrix}
-1.0 & 0.2 & 0.5 & 0.1 \\
-0.5 & -0.8 & -0.2 & 1.5
\end{bmatrix}
$$

将当前语义向量 $[0.50, 2.00]$ 与 $\mathbf{E}_U$ 进行矩阵乘法：

$$
\mathbf{z} = \begin{bmatrix} 0.50 & 2.00 \end{bmatrix} \begin{bmatrix}
-1.0 & 0.2 & 0.5 & 0.1 \\
-0.5 & -0.8 & -0.2 & 1.5
\end{bmatrix}
$$

让我们手算每个候选词的未归一化得分（Logit）：
- 对词 0（<samp>"the"</samp>）：$0.50(-1.0) + 2.00(-0.5) = -0.5 - 1.0 = \mathbf{-1.50}$；
- 对词 1（<samp>"bank"</samp>）：$0.50(0.2) + 2.00(-0.8) = 0.1 - 1.6 = \mathbf{-1.50}$；
- 对词 2（<samp>"river"</samp>）：$0.50(0.5) + 2.00(-0.2) = 0.25 - 0.4 = \mathbf{-0.15}$；
- 对词 3（<samp>"flows"</samp>）：$0.50(0.1) + 2.00(1.5) = 0.05 + 3.0 = \mathbf{+3.05}$。

将这组 Logits $\mathbf{z} = [-1.50, -1.50, -0.15, +3.05]$ 送入 Softmax 计算指数分布：
- $e^{-1.50} \approx 0.223$；
- $e^{-1.50} \approx 0.223$；
- $e^{-0.15} \approx 0.861$；
- $e^{+3.05} \approx 21.115$；
- 分母总和 $\sum = 0.223 + 0.223 + 0.861 + 21.115 = 22.422$。

最终计算得出的精确预测概率：
- $P(\text{"the"}) = \frac{0.223}{22.422} \approx 1.0\%$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.010">1.0%</meter>
- $P(\text{"bank"}) = \frac{0.223}{22.422} \approx 1.0\%$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.010">1.0%</meter>
- $P(\text{"river"}) = \frac{0.861}{22.422} \approx 3.8\%$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.038">3.8%</meter>
- $P(\text{"flows"}) = \frac{21.115}{22.422} \approx \mathbf{94.2\%}$
  <meter min="0" max="1" low="0.2" high="0.6" optimum="0.9" value="0.942">94.2%</meter>

<mark>Transformer 模型以压倒性的 94.2% 极高置信度，精准命中预测出下一个词应当是 <samp>"flows"</samp>（奔流）！</mark>

它之所以能攻克 Lab 01 微型大脑的死循环，正是因为圆桌会议允许多义词 <samp>"bank"</samp> 跨越时空阻隔直接凝视 <samp>"river"</samp>，在交流与思考的双重提炼下，彻底瓦解了孤立词汇的语义迷雾！

---

<h2 id="step-6">第 6 步：核心精髓总结（Core Takeaway）</h2>

<fieldset>
<legend><strong>本章核心记忆卡片</strong></legend>
<p>
<strong>Transformer 是一座“交流”与“思考”交替运转的智能加工厂：</strong><br>
1. <strong>自注意力机制是“交流阶段”</strong>：词与词跨越时间维度横向互通有无，彻底打破了固定窗口的短视失忆与循环链条的遗忘瓶颈；<br>
2. <strong>前馈神经网络是“思考阶段”</strong>：每个词在各自的特征通道上闭门深思、垂直计算，从模型参数中检索深层世界常识；<br>
3. <strong>残差主干道是“中央共享黑板”</strong>：新交流得来的线索和新思考出来的观点被源源不断地加写到主干向量上，使得万亿参数梯度的反向流动一路畅通无阻。
</p>
<p>
现在，你已经清晰俯瞰了整座圆桌会议大厦的宏观蓝图。下一个至关紧要的问题顺理成章浮出水面：<br>
<em>在交流阶段，每个词到底是如何精确计算出“应该看谁”、“看多大程度”的？</em><br>
让我们正式翻开第 06 章，揭开三大核心投影矩阵的神秘面纱——<strong>查询（Query）、键（Key）与值（Value）</strong>！
</p>
</fieldset>

---

<nav aria-label="章节导航">
  <p>
    <a href="../04b-lab-micro-brain/index.zh.html">&larr; 阶段实战工坊 01：80 行纯 Python 训练首个自研大脑</a> &bull;
    <a href="../index.html">课程主页</a> &bull;
    <a href="../06-queries-keys-values/index.zh.html">第 06 章：图书馆寻宝记（查询、键与值） &rarr;</a>
  </p>
</nav>
