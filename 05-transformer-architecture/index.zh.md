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
      <td>模型所能认识的离散子词（Subwords / Tokens）总数。</td>
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

<fieldset>
<legend><strong>深度拆解：BPE（字节对编码）到底是什么？它是如何炼成 128,256 个词表的？</strong></legend>
<p>
初学者常常产生一个直觉困惑：<em>《牛津英语词典》收录了 60 多万词，加上各种科技词汇、网络俚语更是无限开集，128,256 个词表项够用吗？能覆盖中文和世界其他语言吗？</em>
</p>
<p>
<strong>答案是：不仅能覆盖，而且在数学与工程上是 100% 绝对覆盖，永远不会遇到“无法表示的生词”！</strong>
</p>
<p>
这背后的功臣正是大模型分词的黄金法则 &mdash; <strong>BPE（Byte Pair Encoding，字节对编码）</strong>。
</p>

<h4>1. 3 岁孩子的直觉：冰箱字母磁贴与超级胶水</h4>
<p>
想象你的冰箱上只有最基础的单字母磁贴（<kbd>a</kbd>, <kbd>b</kbd>, <kbd>c</kbd> ... <kbd>z</kbd>）。
</p>
<ul>
  <li><strong>如果每次都一个字母一个字母地排</strong>：拼一个单词 <samp>"unbelievable"</samp> 要拿 12 个磁贴，拼一篇短文手指头都要累断了，而且整面冰箱很快就被密密麻麻的单字母占满了（<strong>序列过长，自注意力计算面临 $\mathcal{O}(T^2)$ 算力灾难</strong>）。</li>
  <li><strong>如果把字典里几十万个完整单词全做成现成的大磁贴</strong>：你的口袋根本装不下，而且一旦你想拼一个刚流行的新网络词，口袋里根本没有这个一体成型的磁贴（<strong>词表爆炸，且遭遇未登录词 OOV 崩溃</strong>）。</li>
</ul>
<p>
<strong>BPE 的胶水法则：</strong>拿出一瓶超级胶水，观察平时写故事的过程。只要发现哪两个相邻磁贴总是结对出现（比如 <kbd>t</kbd> 和 <kbd>h</kbd>），就滴一滴胶水粘成复合磁贴 <kbd>th</kbd>；接着发现 <kbd>th</kbd> 和 <kbd>e</kbd> 总是结对，再粘成 <kbd>the</kbd>！一直粘到你拥有 <strong>128,256 个</strong>最好用的磁贴积木为止。
</p>

<h4>2. BPE 算法严谨推导与微型手算实战</h4>
<p>
假设我们的训练语料库中只有 4 个单词，它们的出现频次如下：
</p>
<ul>
  <li><samp>"low"</samp>（5 次） &rarr; 初始切分：<code>l o w &lt;/w&gt;</code></li>
  <li><samp>"lower"</samp>（2 次） &rarr; 初始切分：<code>l o w e r &lt;/w&gt;</code></li>
  <li><samp>"newest"</samp>（6 次） &rarr; 初始切分：<code>n e w e s t &lt;/w&gt;</code></li>
  <li><samp>"widest"</samp>（3 次） &rarr; 初始切分：<code>w i d e s t &lt;/w&gt;</code></li>
</ul>

<p><strong>迭代统计与贪心合并：</strong></p>
<ol>
  <li><strong>第 1 轮合并</strong>：统计语料库中所有相邻二元组出现的次数，发现 <code>(e, s)</code> 在 <samp>"newest"</samp> (6) 与 <samp>"widest"</samp> (3) 中合计共现 **9 次**（最高频！）。因此执行合并规则 1：<code>e + s &rarr; es</code>，将新词元 <code>es</code> 加入词表。</li>
  <li><strong>第 2 轮合并</strong>：重新统计发现 <code>(es, t)</code> 同样共现 **9 次**。执行合并规则 2：<code>es + t &rarr; est</code>。</li>
  <li><strong>第 3 轮合并</strong>：发现 <code>(est, &lt;/w&gt;)</code> 共现 **9 次**。执行合并规则 3：<code>est + &lt;/w&gt; &rarr; est&lt;/w&gt;</code>。</li>
  <li><strong>第 4 轮合并</strong>：统计发现 <code>(l, o)</code> 共现 $5 + 2 = 7$ 次，<code>(o, w)</code> 共现 7 次。依次执行合并：<code>l + o &rarr; lo</code>，接着 <code>lo + w &rarr; low</code>。</li>
</ol>

<p><strong>见证奇迹的推理时刻（Inference）：</strong></p>
<p>
此时用户输入了一个在训练语料库中<strong>一辈子从未出现过的全新单词：<samp>"lowest"</samp></strong>。
</p>
<ol>
  <li>初始字母切分：<code>[l, o, w, e, s, t, &lt;/w&gt;]</code></li>
  <li>按顺序应用学到的合并规则：<code>e, s &rarr; es</code> &rarr; <code>es, t &rarr; est</code> &rarr; <code>est, &lt;/w&gt; &rarr; est&lt;/w&gt;</code> &rarr; <code>l, o &rarr; lo</code> &rarr; <code>lo, w &rarr; low</code></li>
  <li><strong>最终输出 Token 序列：</strong><kbd>"low"</kbd> + <kbd>"est&lt;/w&gt;"</kbd>（仅用 2 个已知子词优雅表征，<strong>零未登录词 Zero OOV</strong>！）。</li>
</ol>

<h4>3. 256 个原生字节终极兜底（Byte-level Fallback）</h4>
<p>
在现代大模型（GPT-4、LLaMA-3）使用的 <strong>Byte-level BPE</strong> 中，词表最底部完整保留了计算机最底层的 <strong>256 个原生 UTF-8 字节（<code>0x00</code> 到 <code>0xFF</code>）</strong>。哪怕遇到从未见过的极生僻汉字（如“龘”、“鱻”）、火星文、生僻符号或一段未经解码的乱码，分词器会自动退化为底层字节序列送入网络，保证<strong>100% 绝对能读写、永不报错崩溃</strong>！
</p>
</fieldset>

### 2. 阶段 1：序列输入表征（Input Representation）

给定用户输入的提示词文本，首先经过分词器（Tokenizer）转换为由 $T$ 个离散整数 ID 构成的序列向量：

$$
\mathbf{w} = \begin{bmatrix} w_1 & w_2 & \dots & w_T \end{bmatrix}^\top \in \{1, \dots, |V|\}^T
$$

<fieldset>
<legend><strong>符号深度解析：$\mathbf{w}$ 与 $w_t$ 的数学含义</strong></legend>
<ul>
  <li>$\mathbf{w}$：整段输入提示词的离散符号向量，长度为 $T$。它就像一列进入大厅的乘客名单；</li>
  <li>$w_t$（或 $w_i$）：序列中<strong>第 $t$ 个位置（或第 $i$ 个位置）上的具体 Token 整数编号</strong>。例如在句子 <samp>"The cat sat on the"</samp> 中，$w_1 = 464$（代表 <kbd>"The"</kbd>），$w_2 = 3797$（代表 <kbd>"cat"</kbd>）；</li>
  <li>$V = \{v_1, v_2, \dots, v_{|V|}\}$：词表全集，包含模型能够认识的全部候选子词；</li>
  <li>$|V|$：词表集合的元素总个数（例如 128,256），每个整数 $w_t$ 必须严格落在区间 $[1, |V|]$ 内；</li>
  <li>$T$：本次输入的总 Token 数量（时间步长 / 序列长度），即当前落座在圆桌周围的词汇总数。</li>
</ul>
</fieldset>

<fieldset>
<legend><strong>概念温故（Refresher）：什么是词嵌入矩阵 E？为什么它能把词变成数字？</strong></legend>
<p>
<strong>第一性原理思考：为什么计算机不能直接拿整数 ID 来计算？</strong>
</p>
<p>
如果直接把 <samp>"cat"</samp> 当成标量 $3797$、<samp>"dog"</samp> 当成标量 $3798$、<samp>"apple"</samp> 当成标量 $1549$：
</p>
<ul>
  <li>在算术上，计算机会得出极其荒唐的结论：$3798 = 3797 + 1$，即 $\text{dog} = \text{cat} + 1$；</li>
  <li>在距离上，<samp>"cat"</samp> 和 <samp>"dog"</samp> 的差值是 1，而 <samp>"cat"</samp> 和 <samp>"kitten"</samp>（小猫，ID 可能是 24000）的差值却高达两万！标量大小完全无法衡量人类词汇真实的语义亲疏。</li>
</ul>
<p>
<strong>几何宇宙的诞生：词嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$</strong>
</p>
<p>
正如我们在第 01 章所推导，解决这个问题的终极方法，是为每个词在多维空间中安放一个专属的几何坐标向量：
</p>
<ul>
  <li><strong>一本厚厚的语义字典</strong>：矩阵 $\mathbf{E}$ 共有 $|V| = 128,256$ 行，每一行代表一个词；每行包含 $d_{\text{model}} = 4,096$ 个连续浮点数，构成了该词的静态语义指纹。</li>
  <li><strong>语义亲近即空间亲近</strong>：在这个 4,096 维的几何世界中，<samp>"cat"</samp> 和 <samp>"dog"</samp> 的向量夹角极小（余弦相似度极高），而它们与 <samp>"refrigerator"</samp> 的向量近乎垂直正交。</li>
  <li><strong>代数查表机制（One-Hot 提取）</strong>：
    当我们用独热向量 $\mathbf{e}_{w_t}^\top = [0, \dots, 0, 1, 0, \dots, 0] \in \mathbb{R}^{1 \times |V|}$ 乘以矩阵 $\mathbf{E}$ 时，因为只有第 $w_t$ 列是 1、其余全为 0，乘法运算在代数上<strong>分毫不差地精确提取出了矩阵 $\mathbf{E}$ 的第 $w_t$ 行</strong>！
  </li>
  <li><strong>工程底层加速</strong>：在现代显卡底层代码（如 PyTorch 的 <code>nn.Embedding</code>）中，计算核心会跳过巨大的稀疏矩阵乘法，直接通过内存地址偏移（<code>E[w_t]</code>）以 $\mathcal{O}(1)$ 速度把这一行连续向量秒级拷贝给后续计算。</li>
</ul>
</fieldset>

如何将离散整数 $w_t$ 转化为连续几何向量并注入次序？通过词嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$ 与位置编码向量 $\mathbf{p}_t$ 完成映射：

$$
\mathbf{x}_t^{(0)} = \mathbf{e}_{w_t}^\top \mathbf{E} + \mathbf{p}_t \in \mathbb{R}^{1 \times d_{\text{model}}}
$$

<fieldset>
<legend><strong>符号深度解析：嵌入查找与位置注入</strong></legend>
<ul>
  <li>$\mathbf{e}_{w_t}$：词表空间内的独热（One-Hot）指示列向量，维度为 $|V| \times 1$。它在第 $w_t$ 行取值为 1，其余所有 $|V|-1$ 个位置全部为 0；</li>
  <li>$\mathbf{e}_{w_t}^\top$：转置后的独热行向量，维度为 $1 \times |V|$；</li>
  <li>$\mathbf{E} \in \mathbb{R}^{|V| \times d_{\text{model}}}$：全局词嵌入矩阵查找表。第 $k$ 行完整存储着第 $k$ 号 Token 在空间中的 $d_{\text{model}}$ 维连续语义坐标；</li>
  <li>$\mathbf{e}_{w_t}^\top \mathbf{E}$：这一矩阵乘法在代数上严格等价于<strong>“提取出矩阵 $\mathbf{E}$ 的第 $w_t$ 行”</strong>（我们在第 01 章已作手算验证）；</li>
  <li>$\mathbf{p}_t \in \mathbb{R}^{1 \times d_{\text{model}}}$：第 $t$ 个位置专属的<strong>位置编码行向量</strong>。因为矩阵乘法本身是无序的（无法区分“猫吃鱼”和“鱼吃猫”），必须通过 $\mathbf{p}_t$ 向词向量注入“这是第几个词”的次序印记（第 10 章详解）；</li>
  <li>上标 $(0)$：表示这是未经任何 Transformer 块处理的<strong>“第 0 层初始表征”</strong>；</li>
  <li>$\mathbf{x}_t^{(0)} \in \mathbb{R}^{1 \times d_{\text{model}}}$：位置 $t$ 处的词在融合了静态词义与位置信息后的初始特征行向量。</li>
</ul>
</fieldset>

将整段文本全部 $T$ 个词的行向量自上而下垂直堆叠，便铸就了贯穿整个模型的**输入特征张量**：

$$
\mathbf{X}^{(0)} = \begin{bmatrix}
\mathbf{x}_1^{(0)} \\
\mathbf{x}_2^{(0)} \\
\vdots \\
\mathbf{x}_T^{(0)}
\end{bmatrix} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

张量 $\mathbf{X}^{(0)}$ 的维度非常清晰：**一共有 $T$ 行（代表 $T$ 个时间步），每一行有 $d_{\text{model}}$ 个数字（代表该词的特征维度）**。

---

### 3. 阶段 2：Transformer 基础计算块（垂直堆叠 $L$ 次）

张量 $\mathbf{X}^{(0)}$ 随后注入由 $L$ 个串联模块组成的深层网络中（$l = 1, 2, \dots, L$）。

每个块内部严格遵循由**残差流（Residual Stream）**串联的两级子架构：

#### 子层 A：全员交流室（自注意力机制 Self-Attention）
每个词环视圆桌，跨越时间维度检索与自己最相关的上下文线索：

$$
\mathbf{H}^{(l)} = \mathbf{X}^{(l-1)} + \operatorname{SelfAttention}\left(\operatorname{RMSNorm}(\mathbf{X}^{(l-1)})\right)
$$

<fieldset>
<legend><strong>符号深度解析：自注意力层的前向传播</strong></legend>
<ul>
  <li>$l$：当前所处的网络层级数（Layer index），取值范围为 $l \in \{1, 2, \dots, L\}$；</li>
  <li>$\mathbf{X}^{(l-1)} \in \mathbb{R}^{T \times d_{\text{model}}}$：第 $l-1$ 层的输出张量，直接作为第 $l$ 层的输入输入（当 $l=1$ 时即为初始张量 $\mathbf{X}^{(0)}$）；</li>
  <li>$\operatorname{RMSNorm}(\cdot)$：均方根层归一化操作（我们在第 13 章将深入推导），独立对每一行的特征向量做均方根尺度缩放，保证信号在深层网络传递时方差始终稳定在标准范围，杜绝数值爆炸；</li>
  <li>$\operatorname{SelfAttention}(\cdot)$：因果多头自注意力机制（第 06～09 章核心）。它利用查询（Query）、键（Key）和值（Value）矩阵计算出词与词之间的动态目光分配权重，输出一个形状完全相同的更新张量（形状仍为 $T \times d_{\text{model}}$）；</li>
  <li>核心符号 $+$：<strong>残差跳跃连接（Residual Connection）</strong>（第 12 章详解）。它把交流得到的新语境以“增量修正”的方式加到原有主干特征上，不仅保全了原有词义，更让反向传播梯度获得了一条畅通无阻的高速通道；</li>
  <li>$\mathbf{H}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$：第 $l$ 层内完成横向全员交流后的<strong>“中间过渡隐藏状态张量”</strong>。</li>
</ul>
</fieldset>

#### 子层 B：闭门思考室（前馈神经网络 FFN / SwiGLU）
各词在吸纳了邻居们提供的新语境后，分别走入各自独立的私人思考室闭门深思、检索事实知识：

$$
\mathbf{X}^{(l)} = \mathbf{H}^{(l)} + \operatorname{FFN}\left(\operatorname{RMSNorm}(\mathbf{H}^{(l)})\right)
$$

在当今顶尖大模型中，该模块标配为我们在第 04 章深入剖析过的 **SwiGLU** 门控网络：

$$
\operatorname{FFN}(\mathbf{h}) = \left(\operatorname{Swish}(\mathbf{h}\mathbf{W}_{\text{gate}}) \odot (\mathbf{h}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}
$$

<fieldset>
<legend><strong>符号深度解析：SwiGLU 前馈网络的门控计算</strong></legend>
<ul>
  <li>$\mathbf{h} \in \mathbb{R}^{1 \times d_{\text{model}}}$：中间状态矩阵 $\mathbf{H}^{(l)}$ 中的某一行向量（代表单个词），矩阵计算时可直接对整个张量 $\mathbf{H}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$ 做批量计算；</li>
  <li>$\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$：<strong>门控投影权重矩阵</strong>，负责判断当前词汇的各个知识通路应当放行还是关死；</li>
  <li>$\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$：<strong>升维投影权重矩阵</strong>，将特征维度从 $d_{\text{model}}$（如 4096）升维放大到超宽的思考维度 $d_{\text{ffn}}$（如 14336）；</li>
  <li>$\operatorname{Swish}(u) = u \cdot \sigma(u) = \frac{u}{1 + e^{-u}}$：我们在第 04 章学过的平滑非线性激活函数；</li>
  <li>$\odot$：<strong>逐元素哈达玛积（Hadamard Product）</strong>，两个相同维度的向量对应元素一一相乘，实现平滑的“门控放行”；</li>
  <li>$\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$：<strong>降维投影权重矩阵</strong>，将高维思考空间提炼出的知识重新压缩投影回主干维度 $d_{\text{model}}$；</li>
  <li>$\mathbf{X}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$：第 $l$ 层完整运算后的最终输出张量，准备作为下一层（第 $l+1$ 层）的输入。</li>
</ul>
</fieldset>

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

---

### 4. 阶段 3：输出解嵌入与词表概率投影

在完整历经 $L$ 轮高强度的“沟通”与“思考”后，输出张量中的每一个词向量都已被赋予了极度深邃的语境智慧：

$$
\mathbf{X}_{\text{final}} = \operatorname{RMSNorm}(\mathbf{X}^{(L)}) \in \mathbb{R}^{T \times d_{\text{model}}}
$$

为了将这些高维几何语义重新映射回人类可读的文字，模型使用**解嵌入矩阵（Unembedding Matrix）** $\mathbf{E}_U \in \mathbb{R}^{d_{\text{model}} \times |V|}$：

$$
\mathbf{Z} = \mathbf{X}_{\text{final}} \mathbf{E}_U \in \mathbb{R}^{T \times |V|}
$$

<fieldset>
<legend><strong>符号深度解析：从隐藏几何空间映射回离散词表</strong></legend>
<ul>
  <li>$\mathbf{X}^{(L)} \in \mathbb{R}^{T \times d_{\text{model}}}$：最后一层（第 $L$ 层）输出的完整特征矩阵；</li>
  <li>$\mathbf{X}_{\text{final}} \in \mathbb{R}^{T \times d_{\text{model}}}$：经过最终全局 RMSNorm 尺度归一化后的输出张量；</li>
  <li>$\mathbf{E}_U \in \mathbb{R}^{d_{\text{model}} \times |V|}$：解嵌入投影矩阵（在很多架构中直接复用输入嵌入矩阵的转置，即 $\mathbf{E}_U = \mathbf{E}^\top$，称为权重绑定 Weight Tying）；</li>
  <li>$\mathbf{Z} \in \mathbb{R}^{T \times |V|}$：<strong>全序列对数几率矩阵（Logits Matrix）</strong>。矩阵共有 $T$ 行、每一行包含 $|V|$ 个数值；</li>
  <li>$\mathbf{z}_T \in \mathbb{R}^{1 \times |V|}$：矩阵 $\mathbf{Z}$ 的最后一行（第 $T$ 行）。它代表模型根据前 $T$ 个词的全部上下文信息，为词表中每一个候选词作为<strong>“第 $T+1$ 个词”</strong>所给出的原始未归一化打分。</li>
</ul>
</fieldset>

将向量 $\mathbf{z}_T$ 送入 **Softmax 函数**：

$$
P(w_{T+1} = v_i \mid w_{\le T}) = \frac{\exp(z_{T, i})}{\sum_{j=1}^{|V|} \exp(z_{T, j})}
$$

<fieldset>
<legend><strong>符号深度解析：Softmax 条件概率计算</strong></legend>
<ul>
  <li>$w_{\le T}$：已知的全部历史上下文 Token 序列 $(w_1, w_2, \dots, w_T)$；</li>
  <li>$w_{T+1}$：即将生成的下一个位置的目标 Token；</li>
  <li>$v_i$：词表全集 $V$ 中的第 $i$ 个具体候选词（$i \in \{1, 2, \dots, |V|\}$）；</li>
  <li>$z_{T, i}$：向量 $\mathbf{z}_T$ 中第 $i$ 个分量，即模型为候选词 $v_i$ 评估出的原始 Logit 得分；</li>
  <li>$\exp(z_{T, i}) = e^{z_{T, i}}$：指数放大操作，确保所有分值全部变为正数，并拉大高分词与低分词的差距；</li>
  <li>分母 $\sum_{j=1}^{|V|} \exp(z_{T, j})$：遍历全词表所有候选词的指数得分总和，作为归一化分母，确保全词表所有词的概率加起来严格等于 $1.0$（100%）。</li>
</ul>
</fieldset>

模型在这一概率分布中完成采样（或直接挑选概率最高的候选词），将其追加到现有序列尾部作为新的已知词，随后整个网络以长度 $T+1$ 启动下一轮计算。这就是现代大语言模型生生不息生成长篇大论的**自回归生成（Autoregressive Generation）**法则！

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
