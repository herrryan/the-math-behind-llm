# 第 09 章：戴上时间盲盒（因果掩码 Causal Masking）


## 第 1 步：3 岁小孩直觉（绘本滑块与防偷看挡板） {: #step-1 }

!!! note "3岁小孩的直觉: 绘本接龙游戏与防偷看硬纸板"
    想象你坐在小木桌前，和老师一起玩一个有趣的绘本猜词游戏。

    面前的桌上摊开着一本精彩的故事书。但你们不是单纯地朗读故事，而是在玩侦探挑战：老师要求你在读到下一个词之前，必须先凭自己的脑子猜出它是什么。

    页面上印着这样一句话：

    *“勇敢的小狗在草地上追逐飞舞的蝴蝶。”*

    假设你现在正读到 **“追逐”** 这个词。老师转过头问你：*“快猜猜看，小狗追逐的后面会是什么？”*

    如果整页书就这样毫无遮拦地摊在眼前，会发生什么？

    你根本不需要动脑子！你既不需要思考小狗喜欢玩什么，也不需要回忆前面的故事情节。你只需要眼睛轻轻往右一瞟，瞄到后面印着的 **“飞舞”**，然后大声念出来就行了。

    你根本没有学会怎么讲故事，你的大脑也没有得到任何锻炼。你只是在光明正大地偷看明天的标准答案！

    为了让你真正学会思考，老师拿来了一块厚厚的黑硬纸板做成**滑块**，盖在书页上。

    当你的手指从左往右滑动时，硬纸板滑块只允许露出你的手指已经摸过的词。手指右边所有还没读到的内容，都被死死封印在漆黑的挡板后面：

    1. **昨天的足迹完全公开**：你可以反复回头看已经读过的词，尽情搜集上下文线索。
    2. **明天的剧透彻底封死**：你绝对看不见未来的任何一个字，哪怕一个字母都不行。
    3. **逼出真正的智慧**：正因为无法偷看剧透，你的大脑才被迫全力运转，去深刻理解词语之间的逻辑关联，真正学会语言的规律！

    在大语言模型中，这块黑色的硬纸板滑块就叫做**因果掩码（Causal Mask）**。它是一副严密的时间盲盒，让模型在充分吸收过去知识的同时，绝对无法偷窥未来！

<figure>
<pre>
阅读时间轴（词语从左向右逐步处理）：

第 1 步：读到 "勇敢"
大脑能看到的文字：[勇敢] ───────► 被挡板遮挡：[???] [???] [???] [???]
（大脑注意力只能放在 "勇敢" 上）

第 2 步：读到 "小狗"
大脑能看到的文字：[勇敢] [小狗] ─► 被挡板遮挡：[???] [???] [???]
（大脑注意力可以在 "勇敢" 和 "小狗" 之间建立联系）

第 3 步：读到 "追逐"
大脑能看到的文字：[勇敢] [小狗] [追逐] ──► 被挡板遮挡：[???] [???]
（大脑综合 "勇敢"、"小狗"、"追逐" 预测下一个词）

不加挡板（作弊）："勇敢" 直接偷看后面的 "蝴蝶" ──► 大脑偷懒，推理能力彻底崩溃！
戴上挡板（诚实）：每个词只能回望过去与当下 ────► 深度推理，学到真正的语言智慧！
</pre>
<figcaption><strong>图 9.1：</strong> 滑块挡板确保每个词元只能向左回望过去与当下的足迹，绝不允许向右偷窥未来的剧透。</figcaption>
</figure>

---

## 第 2 步：承前启后的关键过渡 {: #step-2 }

!!! question "计算连接问题: 从全向注意力迈向时间单向箭头的必然跃迁"
    在第 08 章中，我们推导出了完整的缩放点积注意力公式：



    $$
    \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
    $$



    请仔细观察矩阵相乘的核心部分 $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$。对于长度为 $T$ 的文本序列，这个矩阵计算出了**任意两个位置** $(i, j)$ 之间的匹配得分：

    - 行号 $i$ 代表发起检索的词元（**Query 查询**）。
    - 列号 $j$ 代表提供线索的词元（**Key 键**）。

    在普通的无约束矩阵乘法中，第 1 个词元（$i = 1$）会与第 5 个词元（$j = 5$）计算点积。进入 Softmax 后，第 1 个词元就会从第 5 个词元那里分得可观的注意力百分比权重，导致第 5 个词元的信息直接渗透进了第 1 个词元的输出向量中！

    对于双向编码任务（例如用于理解文本分类的 <abbr title="Bidirectional Encoder Representations from Transformers">BERT</abbr>），前后互看是完全合法的。但在生成式大语言模型（<abbr title="Large Language Model">LLM</abbr>，如 GPT-4、LLaMA-3、Gemini）中，模型的核心使命是**自回归下一个词预测**：



    $$
    P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_{\lt t})
    $$



    在现实推理生成时，当模型正在吐出第 2 个词时，第 5 个词在物理宇宙中根本还不存在！如果模型在训练期间被允许偷看第 5 个词，它就会形成致命的“剧透依赖症”。一旦上线独立推理，未来的剧透骤然消失，模型就会瞬间语无伦次，彻底瘫痪。

    但我们该如何施加这条“时间单向性”戒律？

    - 如果用 `for` 循环按时间一个词一个词地串行运算，GPU 的数万个并行核心将被迫闲置，训练一个大模型需要上千年。
    - 我们极度渴望把整个序列的 $T$ 个词元一次性整块喂进 GPU，在单次巨大的矩阵乘法中全部算完，同时又绝对禁止信息从未来逆流回过去。

    *“我们究竟如何在数学上对注意力打分矩阵实施手术，使得未来的词元获得严格为零的注意力权重（0.0000%）与严格为零的反向传播梯度，同时在单次统一的 GPU 运算中保持极速并行？”*

---

## 第 3 步：严谨数学公式与架构推导 {: #step-3 }

### 1. 带因果掩码的注意力公式

在现代自回归 Transformer 架构中，注意力机制引入了一个加性<dfn id="def-causal-mask-zh">因果掩码矩阵（Causal Mask Matrix）</dfn> $\mathbf{M}$：



$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}
$$



因果掩码矩阵 $\mathbf{M} \in \mathbb{R}^{T \times T}$ 针对行索引 $i$ 与列索引 $j$ 的分段定义如下：



$$
M_{ij} = \begin{cases}
0 & \text{当 } j \le i \quad \text{（过去与当前位置：允许关注）} \\
-\infty & \text{当 } j > i \quad \text{（未来位置：绝对禁止）}
\end{cases}
$$



将其展开为完整的 $T \times T$ 矩阵形式：



$$
\mathbf{M} = \begin{bmatrix}
0 & -\infty & -\infty & \dots & -\infty \\
0 & 0 & -\infty & \dots & -\infty \\
0 & 0 & 0 & \dots & -\infty \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & 0
\end{bmatrix} \in \mathbb{R}^{T \times T}
$$



<details>
<summary><strong>数学符号速查与张量维度清单</strong></summary>
<dl>
  <dt><strong>$T$（序列长度 Sequence Length）</strong></dt>
  <dd>当前输入上下文窗口包含的词元总数（如 $T = 2048$ 或 $T = 8192$）。</dd>
  <dt><strong>$d_k$（注意力头维度 Head Dimension）</strong></dt>
  <dd>Query 与 Key 向量的特征维度（通常为 $d_k = 64$ 或 $d_k = 128$）。</dd>
  <dt><strong>$d_v$（数值向量维度 Value Dimension）</strong></dt>
  <dd>Value 向量的内容承载维度（通常 $d_v = d_k = 64$）。</dd>
  <dt><strong>$\mathbf{Q} \in \mathbb{R}^{T \times d_k}$</strong></dt>
  <dd>打包的 Query 矩阵。第 $i$ 行 $\mathbf{q}_i^\top$ 代表第 $i$ 个词元发出的检索需求。</dd>
  <dt><strong>$\mathbf{K} \in \mathbb{R}^{T \times d_k}$</strong></dt>
  <dd>打包的 Key 矩阵。第 $j$ 行 $\mathbf{k}_j^\top$ 代表第 $j$ 个词元提供的检索索引标签。</dd>
  <dt><strong>$\mathbf{V} \in \mathbb{R}^{T \times d_v}$</strong></dt>
  <dd>打包的 Value 矩阵。第 $j$ 行 $\mathbf{v}_j^\top$ 代表第 $j$ 个词元携带的实际语义内容。</dd>
  <dt><strong>$\mathbf{M} \in \mathbb{R}^{T \times T}$</strong></dt>
  <dd>因果掩码矩阵。主对角线及以下为 $0$，主对角线以上严格为 $-\infty$ 的上三角遮罩。</dd>
  <dt><strong>$i$（行索引 / Query 位置）</strong></dt>
  <dd>正在观察上下文的当前词元位置（$1 \le i \le T$）。</dd>
  <dt><strong>$j$（列索引 / Key 位置）</strong></dt>
  <dd>被观察的来源词元位置（$1 \le j \le T$）。</dd>
  <dt><strong>$-\infty$（负无穷大）</strong></dt>
  <dd>指数运算的数学零吸收元：$\lim_{x \to -\infty} \exp(x) = 0$。</dd>
</dl>
</details>

---

### 2. 负无穷大 $-\infty$ 在 Softmax 中的神奇数学机制

为什么在进入 Softmax 之前直接给打分加上 $-\infty$，就能在数学上天衣无缝地消除未来信息？

设 $S_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}$ 为第 $i$ 个 Query 与第 $j$ 个 Key 之间的原始缩放打分。加上掩码后的有效打分为：



$$
\widetilde{S}_{ij} = S_{ij} + M_{ij}
$$



对第 $i$ 行的所有打分执行标准 Softmax 概率归一化：



$$
A_{ij} = \frac{\exp(\widetilde{S}_{ij})}{\sum_{k=1}^T \exp(\widetilde{S}_{ik})} = \frac{\exp(S_{ij} + M_{ij})}{\sum_{k=1}^T \exp(S_{ik} + M_{ik})}
$$



我们分两种情况严格推演输出结果：

#### 情况 A：未来词元（$j > i$）

对于任何出现在当前位置 $i$ 之后的未来词元，掩码项为 $M_{ij} = -\infty$：



$$
S_{ij} + M_{ij} = S_{ij} + (-\infty) = -\infty
$$



代入自然指数函数 $\exp(x)$ 中：



$$
\exp(S_{ij} + M_{ij}) = \exp(-\infty) = 0
$$



由于分子为严格的绝对零，未来词元分得的注意力概率精确为零：



$$
A_{ij} = \frac{0}{\sum_{k=1}^T \exp(\widetilde{S}_{ik})} = 0.0000 \quad (\forall j > i)
$$



未来词元分到的注意力比例是**纯粹的 0%**！未来词元的任何信息 $\mathbf{v}_j$ 都不可能渗透进当前词元 $i$ 的输出表示中。

#### 情况 B：过去与当前词元（$j \le i$）

对于任何出现在当前位置或历史位置的合法词元，掩码项为 $M_{ij} = 0$：



$$
S_{ij} + M_{ij} = S_{ij} + 0 = S_{ij}
$$





$$
\exp(S_{ij} + M_{ij}) = \exp(S_{ij})
$$



现在审视 Softmax 的分母部分。原本跨越整个序列 $T$ 的累加和，自然分裂为两段：



$$
\sum_{k=1}^T \exp(\widetilde{S}_{ik}) = \sum_{k=1}^i \exp(S_{ik}) + \sum_{k=i+1}^T \underbrace{\exp(-\infty)}_{= 0} = \sum_{k=1}^i \exp(S_{ik})
$$



未来所有项全部自然湮灭为 0！因此，对于所有合法历史词元（$j \le i$），注意力权重自动简化为：



$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^i \exp(S_{ik})} \quad (\text{当 } j \le i)
$$



这展现了令人惊叹的数学优雅性：
1. 未来的权重严格归零（$A_{ij} = 0$）。
2. 合法历史词元的权重之和自动严格等于 $1.0$（$100\%$）：


   $$
   \sum_{j=1}^T A_{ij} = \sum_{j=1}^i A_{ij} + \sum_{j=i+1}^T 0 = \frac{\sum_{j=1}^i \exp(S_{ij})}{\sum_{k=1}^i \exp(S_{ik})} = 1.0
   $$


3. 计算机完全不需要额外的截断、动态切片或二次归一化操作！

---

### 3. 注意力矩阵的下三角几何形态

由于对所有 $j > i$ 都有 $A_{ij} = 0$，经过掩码计算后的注意力权重矩阵 $\mathbf{A} \in \mathbb{R}^{T \times T}$ 必然是一个标准的**下三角矩阵（Lower-Triangular Matrix）**：



$$
\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) = \begin{bmatrix}
1.0 & 0 & 0 & \dots & 0 \\
A_{21} & A_{22} & 0 & \dots & 0 \\
A_{31} & A_{32} & A_{33} & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
A_{T1} & A_{T2} & A_{T3} & \dots & A_{TT}
\end{bmatrix}
$$



<figure>
<pre>
注意力权重矩阵 A 的下三角几何布局：

       j = 1     j = 2     j = 3     j = 4     j = 5
      (Key 1)   (Key 2)   (Key 3)   (Key 4)   (Key 5)
   ┌─────────────────────────────────────────────────┐
i=1│  100%   │    0%   │    0%   │    0%   │    0%   │  ◄── 词元 1 只能看见自己
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=2│   35%   │   65%   │    0%   │    0%   │    0%   │  ◄── 词元 2 能看见词元 1, 2
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=3│   10%   │   20%   │   70%   │    0%   │    0%   │  ◄── 词元 3 能看见词元 1, 2, 3
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=4│    5%   │   15%   │   30%   │   50%   │    0%   │  ◄── 词元 4 能看见词元 1..4
   ├─────────┼─────────┼─────────┼─────────┼─────────┤
i=5│    2%   │    8%   │   15%   │   25%   │   50%   │  ◄── 词元 5 能看见词元 1..5
   └─────────────────────────────────────────────────┘
     ▲──────────────────▲ └─────────────────────────┘
        合法历史区                 非法未来区
     （各行概率和为 1）          （严格归零：0%）
</pre>
<figcaption><strong>图 9.2：</strong> 因果注意力矩阵的下三角几何形态。矩阵的每一行都是针对当前词元及其历史词元的合法概率分布。</figcaption>
</figure>

---

### 4. 反向传播梯度绝缘（零梯度泄漏）

不仅前向传播中的未来信息被完全封锁，反向传播中的梯度也绝对不会向后泄漏。

在反向传播链式求导中，损失函数 $\mathcal{L}$ 对掩码前原始打分 $S_{ij}$ 的偏导数为：



$$
\frac{\partial \mathcal{L}}{\partial S_{ij}} = \sum_{k=1}^T \frac{\partial \mathcal{L}}{\partial A_{ik}} \frac{\partial A_{ik}}{\partial S_{ij}}
$$



对于任何未来位置 $j > i$，由于 $A_{ij} \equiv 0$ 恒等于常数零，且 $\exp(\widetilde{S}_{ij}) = 0$，Softmax 的局部导数处处严格为零：



$$
\frac{\partial A_{ik}}{\partial S_{ij}} = 0 \quad (\forall j > i, \, \forall k)
$$



因此：



$$
\frac{\partial \mathcal{L}}{\partial S_{ij}} = 0 \quad (\forall j > i)
$$



梯度根本无法穿透因果掩码这堵高墙。神经网络在优化参数时，绝对不会利用任何来自于未来的信息梯度，因果逻辑在数学上形成了绝对的单向闭环。

---

## 第 4 步：历史渊源与技术演进 {: #step-4 }

<figure>
<pre>
时间因果序列建模的技术演进简史：

1913年：安德雷·马尔可夫（Andrey Markov） ──► 离散条件概率链：P(w_t | w_{t-1})
                                             （在普希金诗歌《欧根·奥涅金》中统计字母接龙）
        │
        ▼
1948年：克劳德·香农（Claude Shannon） ────► 信息论：语言文本的自回归熵测算
        │
        ▼
1990年：杰弗里·埃尔曼（Jeffrey Elman） ───► 循环神经网络（RNN）：
                                             利用物理时间循环维持因果性：h_t = f(h_{t-1}, x_t)
                                             致命缺陷：O(T) 串行步数！GPU 核心大面积闲置。
        │
        ▼
2017年：Vaswani 等人 ──────────────────────► 《Attention Is All You Need》（第 3.2.3 节）：
                                             因果掩码：把时间箭头投影为空间下三角矩阵！
                                             惊人飞跃：O(1) 串行步数，GPU 算力 100% 满载迸发。
        │
        ▼
2018年：Radford 等人 ──────────────────────► GPT（Generative Pre-trained Transformer）：
                                             确立 Decoder-only 纯因果大模型为当代 AI 主流范式。
</pre>
<figcaption><strong>图 9.3：</strong> 从时间物理循环到空间矩阵掩码的技术跨越。</figcaption>
</figure>

### 1. 从时间循环到空间下三角掩码

在传统的循环神经网络（RNN、LSTM、GRU）中，因果性是通过**物理执行顺序**来强行保证的：



$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)
$$



因为计算第 $t$ 步时必须依赖第 $t-1$ 步输出的隐藏状态 $\mathbf{h}_{t-1}$，词元 $t$ 在物理上绝对不可能偷看词元 $t+1$，因为词元 $t+1$ 根本还没被送入计算机内存！

然而，这种因果保证方式付出了极其惨重的硬件代偿：
- 如果一个文本序列有 $T = 2048$ 个词元，GPU 必须像老式齿轮一样，一步一步串行循环 $2048$ 次。
- GPU 上原本专为巨型并行矩阵运算设计的数千个张量核心（Tensor Cores），大部分时间都只能处于饥饿等待状态。

Vaswani 等人（2017）做出了一个天才般的思想飞跃：**把物理上的时间顺序，转化为几何空间中的矩阵坐标！**
1. 把 $T$ 个词元一次性整块喂进 GPU，打包成单个巨大的输入矩阵 $\mathbf{X} \in \mathbb{R}^{T \times d}$。
2. 用一次密集的 GPU 矩阵乘法，瞬间并行算出所有 $T \times T$ 个点积匹配度。
3. 通过一个全是 $-\infty$ 的上三角掩码矩阵 $\mathbf{M}$，在数学上一次性抹杀所有未来信息。

这项创新把原本需要 $O(T)$ 步的串行训练时间，骤降为 **$O(1)$ 的单步大规模并行操作**，彻底释放了现代超级计算机的吞吐潜能，直接催生了现代百亿、千亿参数大模型的爆发。

---

### 2. 为什么必须是加性 $-\infty$？那些被淘汰的朴素尝试

在探索因果注意力的过程中，研究人员曾尝试过几种看似更直观的方案，但它们都在严谨的数学检验中遭遇了惨败：

<fieldset>
<legend><strong>三次失败的方案与最终的数学胜利者</strong></legend>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 9.1：</strong> 注意力机制中实现因果时序限制的各类方案对比。</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">实现方案</th>
      <th align="center">数学公式</th>
      <th align="left">致命缺陷与数学崩溃原因</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>尝试 1：打分直接乘 0</strong></td>
      <td align="center">$\mathbf{S} \odot \mathbf{M}_{\text{01}}$</td>
      <td>
        <del>灾难级逻辑错误！</del> 如果把未来打分置为 $0$，进入 Softmax 后会算出 $\exp(0) = 1.0$！未来的词元不仅没有消失，反而获得了高达 $1.0$ 的基准注意力权重，严重污染真实上下文！
      </td>
    </tr>
    <tr>
      <td><strong>尝试 2：Softmax 之后置 0</strong></td>
      <td align="center">$\mathbf{A} \odot \mathbf{M}_{\text{01}}$</td>
      <td>
        <del>概率公理被破坏！</del> 抹掉未来权重后，剩下的历史权重之和小于 $1.0$（$\sum_{j=1}^i A_{ij} < 1$）。导致输出向量的范数变小，破坏层间方差平衡，且必须增加一次昂贵的全行重归一化除法。
      </td>
    </tr>
    <tr>
      <td><strong>尝试 3：按长度动态切片</strong></td>
      <td align="center">循环 $t=1 \dots T$：$\mathbf{q}_t \mathbf{K}_{1:t}^\top$</td>
      <td>
        <del>硬件性能雪崩！</del> 每次运算的张量维度都在变大，需要连续发起 $T$ 次不同尺寸的 CUDA 算子调度。无法饱和调用显卡张量核心，显存带宽利用率极低。
      </td>
    </tr>
    <tr bgcolor="#fdfdf0">
      <td><strong>胜利者：加性 $-\infty$ 掩码</strong></td>
      <td align="center">$\operatorname{softmax}(\mathbf{S} + \mathbf{M})$</td>
      <td>
        <ins><strong>数学与工程的双重完美！</strong></ins> 借助 $\exp(-\infty) = 0$ 的数学性质，未来权重天然绝对归零，历史权重自动完美求和为 100%，整个过程在 GPU 内部单次极速算子中并行完成！
      </td>
    </tr>
  </tbody>
</table>
</fieldset>

---

### 3. 工程实战：PyTorch 中的浮点数与 FlashAttention 优化

在纯数学推导中，我们可以随意书写 $-\infty$。但在计算机微处理器和显卡硬件中，实数是用 IEEE 754 浮点数表示的。

在真实工业界落地中：
- 在 32 位单精度浮点（`torch.float32`）中，`float('-inf')` 运作良好。
- 但在现代大模型普遍采用的 16 位半精度浮点（`torch.float16` 或 `torch.bfloat16`）中，字面量无穷大会引发潜在数值风险：
  - 如果整行被打上掩码（在某些交叉注意力 Padding 场景中），分母为 0，会导致 $\frac{0}{0} = \text{NaN}$。
  - 在半精度浮点下，浮点数可表示的极值有限。

因此，工业级开源大模型代码（如 Hugging Face Transformers、Megatron-LM、vLLM）通常采用以下工程策略：

1. **大绝对值有限负数替代**：
   在 `torch.float16` 中，能表示的最小有限负数为 $-65504$。工程中通常使用：
   ```python
   mask_value = torch.finfo(torch.float16).min  # 即 -65504.0
   ```
   因为 $\exp(-65504)$ 在半精度下直接发生下溢，硬件精确输出 `0.0`，既完成了清零，又彻底规避了无穷大带来的算术异常。

2. **算子级下三角计算剪枝（FlashAttention）**：
   在现代推理与训练标配的 **FlashAttention-2**（Dao, 2023）中，显存里甚至根本**不需要分配**那个巨大的 $T \times T$ 掩码矩阵！CUDA 线程在计算点积时，只要发现当前线程坐标满足 $j > i$，就直接在片上高速缓存（SRAM）中跳过算力消耗与显存写入。这不仅完全杜绝了显存浪费，还将内存带宽开销降低了一个数量级。

---

## 第 5 步：手算极简数值示例（3 个词元的完整推导） {: #step-5 }

为了让每一个加减乘除都清晰可查，我们选用一个只有 3 个词元的微型句子，进行全流程纸笔手算。

### 1. 初始数值设定

假设微型模型正在处理一个由 3 个词元构成的句子：

<p align="center">
  <kbd>词元 1: "The"</kbd> &emsp;
  <kbd>词元 2: "cat"</kbd> &emsp;
  <kbd>词元 3: "sat"</kbd>
</p>

这里序列长度 $T = 3$，注意力特征维度设定为 $d_k = 2$。

假设我们已经通过第 08 章的方法算好了原始的点积打分矩阵 $\mathbf{S} \in \mathbb{R}^{3 \times 3}$：



$$
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} = \begin{bmatrix}
2.0 & 1.0 & 4.0 \\
0.0 & 3.0 & 1.0 \\
1.0 & 2.0 & 5.0
\end{bmatrix}
$$



同时，3 个词元各自携带的内容 Value 矩阵 $\mathbf{V} \in \mathbb{R}^{3 \times 2}$ 为：



$$
\mathbf{V} = \begin{bmatrix}
10.0 & 0.0 \\
0.0 & 20.0 \\
30.0 & 30.0
\end{bmatrix}
$$



---

### 2. 注入因果掩码矩阵

构建 $3 \times 3$ 的标准因果掩码矩阵 $\mathbf{M}$：



$$
\mathbf{M} = \begin{bmatrix}
0 & -\infty & -\infty \\
0 & 0 & -\infty \\
0 & 0 & 0
\end{bmatrix}
$$



将掩码矩阵加到原始得分矩阵上：$\widetilde{\mathbf{S}} = \mathbf{S} + \mathbf{M}$：



$$
\widetilde{\mathbf{S}} = \begin{bmatrix}
2.0 + 0 & 1.0 + (-\infty) & 4.0 + (-\infty) \\
0.0 + 0 & 3.0 + 0 & 1.0 + (-\infty) \\
1.0 + 0 & 2.0 + 0 & 5.0 + 0
\end{bmatrix} = \begin{bmatrix}
2.0 & -\infty & -\infty \\
0.0 & 3.0 & -\infty \\
1.0 & 2.0 & 5.0
\end{bmatrix}
$$



---

### 3. 逐行手算 Softmax 概率分布

现在，我们对掩码打分矩阵 $\widetilde{\mathbf{S}}$ 的每一行分别执行 Softmax 归一化。

<fieldset>
<legend><strong>第 1 行：词元 1（"The"）</strong></legend>

词元 1 位于位置 $i = 1$。按因果律，它只能关注自己（$j \le 1$），绝对不能看词元 2 和 3。

1. **求自然指数**：


   $$
   \begin{aligned}
   \exp(\widetilde{S}_{11}) &= \exp(2.0) \approx 7.3891 \\
   \exp(\widetilde{S}_{12}) &= \exp(-\infty) = 0 \\
   \exp(\widetilde{S}_{13}) &= \exp(-\infty) = 0
   \end{aligned}
   $$



2. **指数总和**：


   $$
   \sum_{k=1}^3 \exp(\widetilde{S}_{1k}) = 7.3891 + 0 + 0 = 7.3891
   $$



3. **归一化概率**：


   $$
   \begin{aligned}
   A_{11} &= \frac{7.3891}{7.3891} = 1.0000 \quad (100.0\%) \\
   A_{12} &= \frac{0}{7.3891} = 0.0000 \quad (0.0\%) \\
   A_{13} &= \frac{0}{7.3891} = 0.0000 \quad (0.0\%)
   \end{aligned}
   $$



词元 1 将 100% 的注意力完全倾注在自身之上，未来的两个词元彻底不可见！
</fieldset>

<fieldset>
<legend><strong>第 2 行：词元 2（"cat"）</strong></legend>

词元 2 位于位置 $i = 2$。它可以关注自身与历史词元（$j \in \{1, 2\}$），但不能看未来的词元 3。

1. **求自然指数**：


   $$
   \begin{aligned}
   \exp(\widetilde{S}_{21}) &= \exp(0.0) = 1.0000 \\
   \exp(\widetilde{S}_{22}) &= \exp(3.0) \approx 20.0855 \\
   \exp(\widetilde{S}_{23}) &= \exp(-\infty) = 0
   \end{aligned}
   $$



2. **指数总和**：


   $$
   \sum_{k=1}^3 \exp(\widetilde{S}_{2k}) = 1.0000 + 20.0855 + 0 = 21.0855
   $$



3. **归一化概率**：


   $$
   \begin{aligned}
   A_{21} &= \frac{1.0000}{21.0855} \approx 0.0474 \quad (4.74\%) \\
   A_{22} &= \frac{20.0855}{21.0855} \approx 0.9526 \quad (95.26\%) \\
   A_{23} &= \frac{0}{21.0855} = 0.0000 \quad (0.00\%)
   \end{aligned}
   $$



概率总和验证：$0.0474 + 0.9526 + 0.0 = 1.0000$（严格 100%）。词元 2 高度聚焦于自己（$95.26\%$），分出少量精力关照开头的 "The"（$4.74\%$），未来的 "sat" 权重严格为 0。
</fieldset>

<fieldset>
<legend><strong>第 3 行：词元 3（"sat"）</strong></legend>

词元 3 位于序列末尾 $i = 3$。所有位置 $j \in \{1, 2, 3\}$ 都属于历史或当前位置，没有任何遮蔽！

1. **求自然指数**：


   $$
   \begin{aligned}
   \exp(\widetilde{S}_{31}) &= \exp(1.0) \approx 2.7183 \\
   \exp(\widetilde{S}_{32}) &= \exp(2.0) \approx 7.3891 \\
   \exp(\widetilde{S}_{33}) &= \exp(5.0) \approx 148.4132
   \end{aligned}
   $$



2. **指数总和**：


   $$
   \sum_{k=1}^3 \exp(\widetilde{S}_{3k}) = 2.7183 + 7.3891 + 148.4132 = 158.5206
   $$



3. **归一化概率**：


   $$
   \begin{aligned}
   A_{31} &= \frac{2.7183}{158.5206} \approx 0.0171 \quad (1.71\%) \\
   A_{32} &= \frac{7.3891}{158.5206} \approx 0.0466 \quad (4.66\%) \\
   A_{33} &= \frac{148.4132}{158.5206} \approx 0.9363 \quad (93.63\%)
   \end{aligned}
   $$



概率总和验证：$0.0171 + 0.0466 + 0.9363 = 1.0000$（严格 100%）。词元 3 可以自由纵览已经发生的全句上下文。
</fieldset>

---

### 4. 汇编注意力权重矩阵

把这三行手算结果拼装在一起，就得到了最终完美的下三角注意力矩阵 $\mathbf{A} \in \mathbb{R}^{3 \times 3}$：



$$
\mathbf{A} = \begin{bmatrix}
1.0000 & 0.0000 & 0.0000 \\
0.0474 & 0.9526 & 0.0000 \\
0.0171 & 0.0466 & 0.9363
\end{bmatrix}
$$



我们通过直观的概率计量进度条来审视这一分布：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 9.2：</strong> 各词元的注意力概率分布计量表。</caption>
  <thead>
    <tr bgcolor="#f0eee6">
      <th align="left">查询词元（Query）</th>
      <th align="center">j=1 ("The")</th>
      <th align="center">j=2 ("cat")</th>
      <th align="center">j=3 ("sat")</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>词元 1 ("The")</strong></td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="1.0"></meter><br>
        <strong>100.0%</strong>
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0"></meter><br>
        <samp>0.0% (掩码遮蔽)</samp>
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0"></meter><br>
        <samp>0.0% (掩码遮蔽)</samp>
      </td>
    </tr>
    <tr>
      <td><strong>词元 2 ("cat")</strong></td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0474"></meter><br>
        4.74%
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.9526"></meter><br>
        <strong>95.26%</strong>
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0"></meter><br>
        <samp>0.0% (掩码遮蔽)</samp>
      </td>
    </tr>
    <tr>
      <td><strong>词元 3 ("sat")</strong></td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0171"></meter><br>
        1.71%
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.0466"></meter><br>
        4.66%
      </td>
      <td align="center">
        <meter min="0" max="1" low="0.33" high="0.66" optimum="0.9" value="0.9363"></meter><br>
        <strong>93.63%</strong>
      </td>
    </tr>
  </tbody>
</table>

---

### 5. 提取上下文内容向量（$\mathbf{O} = \mathbf{A}\mathbf{V}$）

最后一步，将注意力权重矩阵 $\mathbf{A} \in \mathbb{R}^{3 \times 3}$ 与内容矩阵 $\mathbf{V} \in \mathbb{R}^{3 \times 2}$ 相乘，计算出融合了合法历史信息的最终表示 $\mathbf{O} \in \mathbb{R}^{3 \times 2}$：



$$
\mathbf{O} = \mathbf{A}\mathbf{V} = \begin{bmatrix}
1.0000 & 0.0000 & 0.0000 \\
0.0474 & 0.9526 & 0.0000 \\
0.0171 & 0.0466 & 0.9363
\end{bmatrix}
\begin{bmatrix}
10.0 & 0.0 \\
0.0 & 20.0 \\
30.0 & 30.0
\end{bmatrix}
$$



我们通过线性和的方式分别计算每一行输出向量：

#### 第 1 行输出向量（$\mathbf{o}_1^\top$）：


$$
\mathbf{o}_1^\top = 1.0000 \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} + 0.0 \begin{bmatrix} 0.0 & 20.0 \end{bmatrix} + 0.0 \begin{bmatrix} 30.0 & 30.0 \end{bmatrix} = \begin{bmatrix} 10.0000 & 0.0000 \end{bmatrix}
$$


词元 1 的输出纯粹来自于其自身的语义沉淀。

#### 第 2 行输出向量（$\mathbf{o}_2^\top$）：


$$
\begin{aligned}
\mathbf{o}_2^\top &= 0.0474 \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} + 0.9526 \begin{bmatrix} 0.0 & 20.0 \end{bmatrix} + 0.0 \begin{bmatrix} 30.0 & 30.0 \end{bmatrix} \\
&= \begin{bmatrix} 0.4740 & 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 & 19.0520 \end{bmatrix} \\
&= \begin{bmatrix} 0.4740 & 19.0520 \end{bmatrix}
\end{aligned}
$$


词元 2 的输出融合了 "The" 和 "cat" 的双重线索。

#### 第 3 行输出向量（$\mathbf{o}_3^\top$）：


$$
\begin{aligned}
\mathbf{o}_3^\top &= 0.0171 \begin{bmatrix} 10.0 & 0.0 \end{bmatrix} + 0.0466 \begin{bmatrix} 0.0 & 20.0 \end{bmatrix} + 0.9363 \begin{bmatrix} 30.0 & 30.0 \end{bmatrix} \\
&= \begin{bmatrix} 0.1710 & 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 & 0.9320 \end{bmatrix} + \begin{bmatrix} 28.0890 & 28.0890 \end{bmatrix} \\
&= \begin{bmatrix} 0.1710 + 28.0890 & 0.9320 + 28.0890 \end{bmatrix} \\
&= \begin{bmatrix} 28.2600 & 29.0210 \end{bmatrix}
\end{aligned}
$$


词元 3 综合了整句话迄今为止所有的语义特征。

整体验算输出矩阵为：



$$
\mathbf{O} = \begin{bmatrix}
10.0000 & 0.0000 \\
0.4740 & 19.0520 \\
28.2600 & 29.0210
\end{bmatrix}
$$



每一个数字都可以用纸笔精准验证。在整个计算流程中，未来就像从未存在过一样，被数学高墙彻底阻隔！

---

## 第 6 步：核心精髓总结 {: #step-6 }

!!! tip "核心要点: 因果掩码的核心精髓"
    **因果掩码利用自然指数的数学零吸收性质 $\exp(-\infty) = 0$，在单次矩阵加法中为未来词元铸造了一道不可逾越的时间单向高墙。**

    它将物理世界中必须一步步串行等待的时间顺序，优雅地投影为几何空间中的上三角掩码矩阵。这一飞跃让大语言模型得以兼收并蓄两项极致优势：既能利用 GPU 的数万核心以单次大矩阵并行吞吐整个序列，又能严格保证模型永远在真诚地预测未来，而非投机作弊。
