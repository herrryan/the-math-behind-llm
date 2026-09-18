# 第 11 章：戴上不同的多棱眼镜（多头注意力机制 Multi-Head Attention）

---

## 步骤 1：3 岁孩子也能懂的直觉（小小侦探队与三副有色眼镜）

> [!INTUITION] 侦探俱乐部与神奇的有色眼镜
> 想象一个由三位聪明小朋友组成的“秘密侦探队”：天天、莉莉和山山。
>
> 某天下午，俱乐部收到了一张神秘的小纸条：
>
> *“勇敢的小狗昨天兴奋地追逐着飞舞的蝴蝶。”*
>
> 孩子们需要彻底弄明白故事里发生的所有细节。但有意思的是，每个小朋友鼻梁上都架着一副**特制的彩色魔法眼镜**：
>
> 1. **天天戴着“蓝色语法眼镜”**：
>    - 当天天看纸条时，蓝色镜片会自动过滤掉其他信息，专门高亮**谁对谁做了什么**。
>    - 他的目光瞬间把“小狗”、“追逐”和“蝴蝶”连在一起。他不在乎这件事是昨天发生的还是前天发生的，他只关心“谁是主角，谁被追了”。
>
> 2. **莉莉戴着“红色情绪眼镜”**：
>    - 当莉莉看纸条时，红色镜片专门捕捉**每个角色的心情与神态**。
>    - 她的目光立刻把“勇敢”和“兴奋”投射回“小狗”身上。她瞬间体会到了场景里活泼欢快的气氛！
>
> 3. **山山戴着“绿色时间轴眼镜”**：
>    - 当山山看纸条时，绿色镜片只盯着**事情发生的先后与时间因果**。
>    - 他的目光迅速锁定“昨天”，把追逐事件稳稳钉在日历的特定格子里。
>
> 现在，想象一下如果侦探队**只有一位小朋友**，而且非要把蓝、红、绿三副镜片硬生生叠在一起戴在眼睛上：
>
> 整个视野瞬间变成了一片模糊混沌的黑褐色！如果你试图用同一双眼睛在同一秒内同时看清语法动作、情绪起伏和时间坐标，所有线索就会互相干扰，揉成一团毫无意义的平均值。
>
> 聪明的做法是：三位小朋友各自在自己的桌子前专注研究，在特制卡片上写下自己专精视角的发现，最后大家围坐在一起，把三张卡片整整齐齐拼在大黑板上，拼成一份全知全能的超级破案报告！
>
> 在大语言模型中，这支分工协作的侦探队就叫做**多头注意力机制（Multi-Head Attention，简称 MHA）**。模型没有逼迫单一一套注意力逻辑去吃力地兼顾语言的所有维度，而是把自己的“大脑带宽”切分成好几个并行的“头”，让每个头戴上不同的眼镜去独立观察句子，最后再把大家的视角严丝合缝地缝合在一起！

<figure>
<pre>
单头注意力的视野瓶颈（一双眼睛试图看清万物）：

输入句子："河边的银行今天批准了这笔商业贷款。"
单一注意力头：
  "银行" ──► 试图同时关注 "河边"（地理特征）与 "贷款"（金融属性）
  灾难结果：两种截然不同的语义特征被迫混在一起，算出的点积只能折中成平庸的平均分！

多头注意力方案（专职侦探小分队）：

词向量输入 [1 x 4] ───┬──► 头 1（语法动作头）──► 极度专注锁定（"银行" ◄──► "批准"）
                       ├──► 头 2（行业领域头）──► 极度专注锁定（"银行" ◄──► "贷款"）
                       └──► 头 3（空间位置头）──► 极度专注锁定（"银行" ◄──► "河边"）
                                            │
                                            ▼
                       拼接所有头 Concat(头 1, 头 2, 头 3)
                                            │
                                            ▼
                       最终线性投影矩阵 (W_O) 融合输出
</pre>
<figcaption><strong>图 11.1：</strong> 多头注意力机制允许模型在相互正交的子空间中，同时捕捉语法、语义与长程依赖等多重关系。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

> [!BRIDGING] 单一子空间的“表达力压缩瓶颈”
> 在第 08 章中，我们推导了标准的缩放点积注意力公式：
>
> $$
> \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
> $$
>
> 对于任意一对词元 $i$ 和 $j$，内积项 $\mathbf{q}_i^\top \mathbf{k}_j$ 最终仅仅产生**一个单一标量**。
>
> 经过 Softmax 归一化后，这个标量变成了区间 $[0, 1]$ 上的单一注意力权重 $A_{ij}$。
>
> 然而，人类语言中的任意一个词，都同时身兼多种复杂的语义关联：
> - **主谓宾动宾关系**：动词指向宾语（“吃” $\to$ “苹果”）。
> - **代词指代关系**：代词溯源先行词（“它” $\to$ “小狗”）。
> - **多义词消歧关系**：根据上下文判定词义（“苹果”是水果还是手机品牌）。
> - **修饰限定关系**：副词形容动词，形容词限定名词。
>
> 如果模型只有一个注意力头，它就必须将这所有完全不同的关联硬生生压缩进唯一的概率分布里。如果某个词元将 $80\%$ 的注意力权重分配给了它的主语，那么它几乎就只剩下 $0$ 的注意力来关联情绪或者时态线索！
>
> 此时有人可能会天真地提出：*“既然 1 个头不够用，那我们直接并行跑 8 个全尺寸的注意力头不就好了？”*
> 但如果每个头都在完整隐层维度 $d_{\text{model}} = 4096$ 上全速运转，模型的显存占用和计算量将瞬间直接暴涨 8 倍，硬件根本不堪重负！
>
> “我们究竟如何在数学上，把一个高维向量空间拆分为 $h$ 个互不干扰的独立低维子空间，让模型能够同时戴上多副眼镜并行观察，却又能保证总计算量（FLOPs）和总参数量与原本的单头模型分毫不差？”

---

## 步骤 3：严谨数学推导与公式

### 1. 多头注意力架构公式

在《Attention Is All You Need》（Vaswani 等人，2017）的开创性设计中，<dfn id="def-mha-zh">多头注意力机制（Multi-Head Attention）</dfn>通过 $h$ 组互不相同的投影矩阵，将查询、键、值投影到低维子空间中分别计算注意力。

对于输入的词元表示序列 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{T \times d_{\text{model}}}$：

$$
\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\operatorname{head}_1, \operatorname{head}_2, \dots, \operatorname{head}_h)\mathbf{W}^O
$$

其中，每个独立的注意力头 $\operatorname{head}_i$（$i \in \{1, 2, \dots, h\}$）独立计算：

$$
\operatorname{head}_i = \operatorname{Attention}\left(\mathbf{Q}\mathbf{W}_i^Q, \, \mathbf{K}\mathbf{W}_i^K, \, \mathbf{V}\mathbf{W}_i^V\right) = \operatorname{softmax}\left(\frac{(\mathbf{Q}\mathbf{W}_i^Q)(\mathbf{K}\mathbf{W}_i^K)^\top}{\sqrt{d_k}}\right)(\mathbf{V}\mathbf{W}_i^V)
$$

---

### 2. 张量维度与子空间切分

多头注意力内部的每一个维度都经过了严格的代数设计：

<details>
<summary><strong>数学符号全景清单与维度映射</strong></summary>
<dl>
  <dt><strong>$d_{\text{model}}$（模型隐层主维度）</strong></dt>
  <dd>残差连接主干上的向量宽度（例如标准 Transformer 中为 $512$，LLaMA-7B 中为 $4096$，LLaMA-70B 中为 $8192$）。</dd>
  <dt><strong>$h$（注意力头数量）</strong></dt>
  <dd>并行子空间的数量（例如标准 Transformer 中 $h = 8$，LLaMA-7B 中 $h = 32$，LLaMA-70B 中 $h = 64$）。</dd>
  <dt><strong>$d_k$（键与查询的头子空间维度）</strong></dt>
  <dd>每个头内部的向量维度，定义为：$d_k = \frac{d_{\text{model}}}{h}$（业界通常固定为 $64$ 或 $128$）。</dd>
  <dt><strong>$d_v$（值的头子空间维度）</strong></dt>
  <dd>每个头内部的值向量维度，通常取 $d_v = d_k = \frac{d_{\text{model}}}{h}$。</dd>
  <dt><strong>$\mathbf{W}_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  <dd>第 $i$ 个注意力头的查询投影权重矩阵。</dd>
  <dt><strong>$\mathbf{W}_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$</strong></dt>
  <dd>第 $i$ 个注意力头的键投影权重矩阵。</dd>
  <dt><strong>$\mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$</strong></dt>
  <dd>第 $i$ 个注意力头的值投影权重矩阵。</dd>
  <dt><strong>$\mathbf{W}^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$</strong></dt>
  <dd>多头拼接后的最终输出线性投影矩阵，负责将所有头的视角融合还原回 $d_{\text{model}}$ 维度。</dd>
</dl>
</details>

---

### 3. “计算量守恒奇迹”：为什么多头并不比单头多消耗算力？

初学者常常陷入误区，以为拥有 8 个头就会让模型变慢 8 倍、参数膨胀 8 倍。然而数学证明：**总参数量和总计算浮点数（FLOPs）与单头全尺寸注意力完全严格等价！**

我们通过严密的代数推演予以证明：

#### A. 投影参数量守恒推导

假设我们构建一个单一的全尺寸头，直接在 $d_{\text{model}}$ 维度上进行投影：
- 查询矩阵 $\mathbf{W}^Q$ 的参数量为：$d_{\text{model}} \times d_{\text{model}} = d_{\text{model}}^2$。

现在计算 $h$ 个独立子空间头的总参数量，由于每个子空间维度缩小为 $d_k = \frac{d_{\text{model}}}{h}$：

$$
\sum_{i=1}^h \operatorname{size}(\mathbf{W}_i^Q) = h \times \left(d_{\text{model}} \times d_k\right) = h \times \left(d_{\text{model}} \times \frac{d_{\text{model}}}{h}\right) = d_{\text{model}}^2
$$

头数 $h$ 与子空间收缩比例 $\frac{1}{h}$ 在乘法中被精确抵消了！

同样，对于输出投影矩阵 $\mathbf{W}^O$：

$$
\operatorname{size}(\mathbf{W}^O) = (h \cdot d_v) \times d_{\text{model}} = \left(h \cdot \frac{d_{\text{model}}}{h}\right) \times d_{\text{model}} = d_{\text{model}}^2
$$

#### B. 注意力核心计算量（FLOPs）守恒

每个子头计算点积 $\mathbf{Q}_i \mathbf{K}_i^\top$ 需要 $T^2 d_k$ 次乘加运算。
将所有 $h$ 个头累加：

$$
\text{总点积运算量} = h \times (T^2 d_k) = h \times \left(T^2 \frac{d_{\text{model}}}{h}\right) = T^2 d_{\text{model}}
$$

这意味着：**多头注意力在数学上是将一个高维空间无损正交投影到了 $h$ 个平行的低维子流形上，没有多花一分钱的算力，却获得了 $h$ 倍的观察视角！**

---

### 4. 显存墙的破局演进：MHA、MQA 与 GQA

在长文本大模型推理（Inference）过程中，系统必须将所有历史词元的 Key 和 Value 矩阵驻留在显存中（即 <abbr title="Key-Value Cache">KV Cache</abbr>）。随着上下文长度跨入 32k、128k 甚至 1M，存储 $h$ 组 KV 矩阵所带来的显存压力成为了大模型落地的最大瓶颈。

为了击碎这一瓶颈，学术界和工业界经历了三次重大架构跃迁：

<figure>
<pre>
三种注意力架构的 KV 结构对比：

1. 多头注意力 (Multi-Head Attention, MHA) ── Vaswani 等人 (2017)
   Queries: [ Q_1 ] [ Q_2 ] [ Q_3 ] [ Q_4 ] [ Q_5 ] [ Q_6 ] [ Q_7 ] [ Q_8 ]
   Keys:    [ K_1 ] [ K_2 ] [ K_3 ] [ K_4 ] [ K_5 ] [ K_6 ] [ K_7 ] [ K_8 ]  ◄── 显存中保存 8 组 KV 头
   Values:  [ V_1 ] [ V_2 ] [ V_3 ] [ V_4 ] [ V_5 ] [ V_6 ] [ V_7 ] [ V_8 ]

2. 多查询注意力 (Multi-Query Attention, MQA) ── Shazeer (2019)
   Queries: [ Q_1 ] [ Q_2 ] [ Q_3 ] [ Q_4 ] [ Q_5 ] [ Q_6 ] [ Q_7 ] [ Q_8 ]
   Keys:    [                    全头共享的单组 Key (K)                   ]  ◄── 显存中仅存 1 组 KV 头
   Values:  [                   全头共享的单组 Value (V)                  ]      （显存暴降 8 倍！）

3. 分组查询注意力 (Grouped-Query Attention, GQA) ── Ainslie 等人 (2023) [LLaMA-3, Mistral]
   Queries: [ Q_1   Q_2 ] [ Q_3   Q_4 ] [ Q_5   Q_6 ] [ Q_7   Q_8 ]
   Keys:    [   K_1     ] [   K_2     ] [   K_3     ] [   K_4     ]        ◄── 显存中保存 4 组 KV 头
   Values:  [   V_1     ] [   V_2     ] [   V_3     ] [   V_4     ]          （精度与吞吐的最佳黄金分割点）
</pre>
<figcaption><strong>图 11.2：</strong> MHA、MQA 与现代工业标配 GQA 的结构对比示意图。</figcaption>
</figure>

1. **多查询注意力（Multi-Query Attention, MQA, Shazeer 2019）**：
   - 保留 $h$ 个不同的 Query 头，但让所有头**共享唯一的一组 Key 和 Value 头**。
   - 显存占用直接砍到原本的 $\frac{1}{h}$（显存减少 8 到 32 倍），极大释放了并发 Batch 空间。
2. **分组查询注意力（Grouped-Query Attention, GQA, Ainslie 等人 2023）**：
   - 将 $h$ 个 Query 头划分为 $g$ 个小组（$1 \lt g \lt h$），每组内部共享一对 KV 头。
   - 例如 LLaMA-3-70B 共有 $64$ 个 Query 头，按 $8$ 个一组分为 $8$ 组 KV 头（$g = 8$）。
   - 在几乎百分之百保留 MHA 丰富表达力的同时，将长文本推理的显存开销暴减 $8$ 倍，成为了当下开源与闭源前沿大模型无可争议的行业标准！

---

## 步骤 4：历史源流与思考演进

<figure>
<pre>
多视角注意力的演进脉络：

1990s-2015: 集成学习 (Ensemble) ──► 独立训练 N 个模型并对输出取平均。
                                    参数量呈 N 倍线性膨胀，工业代价极其沉重。
      │
      ▼
2016: 早期 NLP 自注意力机制 ──────► 单头自注意力。
                                    遭遇“子空间坍缩”危机；无法在同一步骤中分离多种语义关系。
      │
      ▼
2017: Vaswani 等人 (Google) ──────► 多头注意力 (MHA)
                                    提出子空间降维切分法 (d_k = d_model / h)；
                                    实现参数与算力零增长的多视角集成表达。
      │
      ▼
2019: Noam Shazeer (Google) ──────► 多查询注意力 (MQA)
                                    洞察到生成阶段的 KV Cache 显存带宽是吞吐瓶颈。
      │
      ▼
2023: Ainslie 等人 (Google) ──────► 分组查询注意力 (GQA)
                                    兼顾表征精度与长上下文极速推理；
                                    成为 LLaMA-2/3、Mistral、DeepSeek 的工业标配。
</pre>
<figcaption><strong>图 11.3：</strong> 从传统模型集成到现代分组查询注意力机制的演进历程。</figcaption>
</figure>

### 1. 历史触发点：单头注意力的“平均化灾难”

在多头注意力问世之前，早期的神经注意力网络（如 Bahdanau 2014, Lin 2017）普遍使用单向量注意力。研究人员在深入可视化注意力热力图时，发现了一个致命的**“语义平均化灾难”**：

当一个词同时具有语法功能（比如向后寻找助动词构成完成时态）和修饰功能（向前修饰中心名词）时，单一的 Softmax 概率分布被迫做出妥协。它只能在两个目标之间“各打五十大板”，使得两边的注意力权重都只剩下 $0.4 \sim 0.5$。这种折中的线性加权，导致输出的 Value 混合向量成了一杯“温吞水”，两头都没抓住！

Vaswani 等人敏锐地发现：**不需要增加整体维度，只需把空间切成几条互不相干的跑道**，头 1 就可以用 $1.0$ 的极致专注度去盯语法助动词，而头 2 则可以用 $1.0$ 的专注度去锁死修饰词！

---

### 2. 现代生产力架构横向评测

<fieldset>
<legend><strong>架构取舍权衡对比：MHA vs. MQA vs. GQA</strong></legend>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 11.1：</strong> 注意力头架构核心特性横向对比。</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">架构方案</th>
      <th align="center">Query 头数</th>
      <th align="center">KV 头数</th>
      <th align="center">KV Cache 显存比例</th>
      <th align="left">工业权衡分析与核心结论</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>多头注意力 (MHA)</strong></td>
      <td align="center">$h$</td>
      <td align="center">$h$</td>
      <td align="center">$1.0\times$ (基准线)</td>
      <td>
        <del>显存负担沉重！</del> 训练阶段特征表达能力极佳，但推理长文本时 KV 缓存轻松吞噬数十 GB 显存。
      </td>
    </tr>
    <tr>
      <td><strong>多查询注意力 (MQA)</strong></td>
      <td align="center">$h$</td>
      <td align="center">$1$</td>
      <td align="center">$\frac{1}{h}\times$ (最高缩减 32 倍)</td>
      <td>
        <del>微弱精度回落。</del> 推理速度极快，并发 Batch 吞吐惊人，但在高难度多步复杂推理任务上偶有性能轻微下降。
      </td>
    </tr>
    <tr bgcolor="#fdfdf0">
      <td><strong>分组查询注意力 (GQA)</strong></td>
      <td align="center">$h$</td>
      <td align="center">$g$ ($1 \lt g \lt h$)</td>
      <td align="center">$\frac{g}{h}\times$ (通常缩减 8 倍)</td>
      <td>
        <ins><strong>现代大模型无可争议的行业标准！</strong></ins> 几乎无损复现 MHA 的全部学术榜单精度，同时斩获接近 MQA 的极速推理体验与显存削减。
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

## 步骤 5：手把手超简单数字积木（$h = 2$ 时的微型纯算推导）

为了让你彻底看透矩阵是如何在指尖流转的，我们用最微型的数字积木，一步步完成全部乘法、除法与拼接计算。

### 1. 参数设定

设句子长度 $T = 2$（包含词元：<kbd>"The"</kbd> 和 <kbd>"cat"</kbd>）。
设模型隐层总维度 $d_{\text{model}} = 4$。
设注意力头数 $h = 2$。
则每个子头的空间维度为：

$$
d_k = d_v = \frac{d_{\text{model}}}{h} = \frac{4}{2} = 2
$$

假设经过词嵌入后，输入的词元序列特征矩阵为：

$$
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1^\top \\
\mathbf{x}_2^\top
\end{bmatrix} = \begin{bmatrix}
1.0 & 0.0 & 1.0 & 0.0 \\
0.0 & 1.0 & 0.0 & 1.0
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

为让推导最清晰直观，我们设 $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$。

---

### 2. 计算头 1（$i = 1$：专注观察前两维子空间）

设头 1 的投影矩阵专门提取前两个维度：

$$
\mathbf{W}_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad
\mathbf{W}_1^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad
\mathbf{W}_1^V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{4 \times 2}
$$

执行投影相乘，得到头 1 的子空间张量：

$$
\mathbf{Q}_1 = \mathbf{X}\mathbf{W}_1^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{K}_1 = \mathbf{X}\mathbf{W}_1^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{V}_1 = \mathbf{X}\mathbf{W}_1^V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

计算头 1 的未缩放内积点阵（缩放系数 $\sqrt{d_k} = \sqrt{2} \approx 1.414$）：

$$
\mathbf{Q}_1 \mathbf{K}_1^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}
$$

除以 $\sqrt{2}$ 后的缩放得分 $\mathbf{S}_1$：

$$
\mathbf{S}_1 = \begin{bmatrix} \frac{1}{\sqrt{2}} & 0 \\ 0 & \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 & 0.0000 \\ 0.0000 & 0.7071 \end{bmatrix}
$$

逐行应用 Softmax 归一化：
- 第 1 行：$e^{0.7071} \approx 2.0281$，$e^{0} = 1.0000$，分母和为 $3.0281$。
  $A_{11} = \frac{2.0281}{3.0281} \approx 0.67$，$A_{12} = \frac{1.0}{3.0281} \approx 0.33$。
- 第 2 行根据对称性：$A_{21} \approx 0.33$，$A_{22} \approx 0.67$。

$$
\mathbf{A}_1 \approx \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
$$

与值矩阵 $\mathbf{V}_1$ 相乘，得出头 1 的输出结果：

$$
\operatorname{head}_1 = \mathbf{A}_1 \mathbf{V}_1 = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 0.67 & 0.66 \\ 0.33 & 1.34 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

---

### 3. 计算头 2（$i = 2$：专注观察后两维并施加交叉关联）

现在观察头 2，其投影矩阵专门提取后两维并进行交叉重组：

$$
\mathbf{W}_2^Q = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{W}_2^K = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
\mathbf{W}_2^V = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 3 & 0 \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 2}
$$

执行线性变换：

$$
\mathbf{Q}_2 = \mathbf{X}\mathbf{W}_2^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{K}_2 = \mathbf{X}\mathbf{W}_2^K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad
\mathbf{V}_2 = \mathbf{X}\mathbf{W}_2^V = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

计算点积矩阵：

$$
\mathbf{Q}_2 \mathbf{K}_2^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \end{bmatrix}
$$

除以 $\sqrt{2}$ 后的得分矩阵 $\mathbf{S}_2$：

$$
\mathbf{S}_2 \approx \begin{bmatrix} 0.0000 & 0.7071 \\ 0.7071 & 0.0000 \end{bmatrix}
$$

Softmax 权重矩阵 $\mathbf{A}_2$：

$$
\mathbf{A}_2 \approx \begin{bmatrix} 0.33 & 0.67 \\ 0.67 & 0.33 \end{bmatrix}
$$

<mark>看，惊人的差异出现了：头 1 主要聚焦自身（主对角线 0.67），而头 2 在另一个子空间中精准地捕获了对侧词元（反对角线 0.67）！</mark>

计算头 2 的加权输出：

$$
\operatorname{head}_2 = \mathbf{A}_2 \mathbf{V}_2 = \begin{bmatrix} 0.33 & 0.67 \\ 0.67 & 0.33 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.99 & 0.67 \\ 2.01 & 0.33 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

---

### 4. 拼合多头结果（Concat）

将两个 2 维子头的输出在特征维度上水平拼合：

$$
\mathbf{H}_{\text{cat}} = \operatorname{Concat}(\operatorname{head}_1, \operatorname{head}_2) = \begin{bmatrix}
0.67 & 0.66 & 0.99 & 0.67 \\
0.33 & 1.34 & 2.01 & 0.33
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

现在，每个词元对应的行向量中，前两列承载了头 1 的自我专注见解，后两列承载了头 2 的跨词交互洞察！

---

### 5. 输出线性投影矩阵融合（$\mathbf{W}^O$）

最后，将拼合矩阵乘以上层输出投影矩阵 $\mathbf{W}^O \in \mathbb{R}^{4 \times 4}$。为简化计算，取 $\mathbf{W}^O = \mathbf{I}_4$（单位矩阵）：

$$
\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{H}_{\text{cat}} \mathbf{W}^O = \begin{bmatrix}
0.67 & 0.66 & 0.99 & 0.67 \\
0.33 & 1.34 & 2.01 & 0.33
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

两个词元在此刻同时吸收了多个正交子空间的特征信息，而且整个过程在硬件上没有产生任何额外的参数暴涨！

---

## 步骤 6：核心精要（一句话记住核心奥秘）

> [!TIP] 多头注意力机制的终极心法
> **多头注意力将高维隐藏特征空间切分为多个平行的正交低维子空间，让模型能够同时戴上多副有色眼镜，独立捕捉语法、指代与时序逻辑。**
>
> 通过设定每个头的维度 $d_k = d_{\text{model}} / h$，Transformer 在未增加任何额外参数与计算量的前提下，优雅实现了类似多模型集成的超级表征能力。
