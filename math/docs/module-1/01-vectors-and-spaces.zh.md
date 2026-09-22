# 第01章：词汇地图（向量与词嵌入）



## Step 1: 3岁小孩的直觉（巨型操场地图） {: #step-1 }

!!! note "3岁小孩的直觉: 3岁小孩的直觉：巨型操场地图"
    想象一下，你把心爱的所有玩具统统带到了一个巨大的绿草地操场上。

    你在操场上划分出不同的沙坑，让相似的玩具聚在相邻的沙盒里：泰迪熊、小猫咪和小狗坐在软绵绵、毛茸茸的沙盒里；木头飞机和小赛车坐在呼呼飞驰的机器沙盒里；香蕉和红苹果坐在美味零食沙盒里。

    现在，想象你在操场地面上用粉笔画出整齐的方格坐标线：
    - 向北走 2 步，向东走 8 步：你找到了毛茸茸沙盒里的小猫咪。
    - 向北走 9 步，向东走 1 步：你找到了机器沙盒里的巨型飞机。

    这样，操场上的每一个玩具都拥有了一个独一无二的**专属门牌地址**。

    如果两个玩具很像，它们的门牌号就在同一个或隔壁沙盒里，彼此挨得很近；如果两个玩具毫无共同点（比如一只毛茸茸的小猫和一架庞大的波音747客机），它们的地址就隔了十万八千里。

    这就是大语言模型（LLM）对待人类语言的方式。它给每个单词都在一张巨大的数学操场地图上分配一个门牌地址，让含义相近的词并肩坐在一起！

---

## Step 2: 承上启下的问题（从文字到数字） {: #step-2 }

在第00章中，我们了解到大模型通过计算词表上的概率分布来预测下一个词。

然而，计算机芯片和 GPU 是由硅片与电子线路构成的物理器件。硬件根本无法理解英文字母 <kbd>"c"</kbd>, <kbd>"a"</kbd>, <kbd>"t"</kbd> 或汉字“猫”。计算机处理器不可能拿单词 <kbd>"kitten"</kbd> 直接去乘以 $0.90$。

**我们究竟该如何把人类的自然语言单词，转换为计算机芯片能够存储、相加、相乘的物理坐标（数字列表）？**

### 失败的捷径：为什么不能简单给单词编号 1, 2, 3？

初学者最容易想到的直觉，是给词典里的每个词编一个顺序整数 ID：
- $\text{"cat"} = 1$
- $\text{"dog"} = 2$
- $\text{"kitten"} = 3$
- $\text{"airplane"} = 4$

这种简单的标量整数编号会遭遇灾难性的彻底失败，根源在于两大数学漏洞：

1. **虚假的数值大小序关系**：  
   在计算机眼里，数字是天然带有大小关系的：$1 < 2 < 3 < 4$。难道能说飞机比猫“更大”，或者狗的语义恰好位于猫和小猫的“中间半路”吗？显然不能！词汇是互不相同的语义范畴，绝非排位高低的标量。

2. **荒谬的虚假代数等式**：  
   在数学中，$1 + 2 = 3$。如果采用标量编号，计算机在做运算时就会得出荒唐的结论：


   $$
   \text{"cat"} (1) + \text{"dog"} (2) = \text{"kitten"} (3)
   $$


   这是毫无意义的胡话。一只猫加一只狗，绝对变不出一只小猫咪！

### 正确的破局之道：多维向量（Vectors）

我们不再用单一的标量数字，而是使用一整串**多维坐标列表**来刻画一个词，其中每个坐标轴衡量一个独立的语义特征（例如：毛茸茸程度、物理体积、是否有生命、机械属性等）。

这一组坐标数值在数学上被称为**向量**（$\mathbf{x} \in \mathbb{R}^d$），而所有这些向量栖息的几何空间，被称为**嵌入空间**（Embedding Space）。

---

## Step 3: 精确数学公式 {: #step-3 }

### 1. 独热编码向量（$\mathbf{e}_i$）

在词语被映射到多维几何坐标之前，它最初在词汇集合 $V$（大小为 $|V|$）中以离散形式存在。

位于词汇表第 $i$ 个位置的词（$i \in \{1, 2, \dots, |V|\}$），形式化表示为**独热向量**（One-Hot Vector）$\mathbf{e}_i \in \{0, 1\}^{|V|}$。这是一个维度高达 $|V|$ 的细长列向量，除了在第 $i$ 个位置为数字 $1$ 之外，其余所有位置全为 $0$：



$$
\mathbf{e}_i = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} \leftarrow \text{第 } i \text{ 个位置}
$$



严格形式化下，独热向量 $\mathbf{e}_i$ 的第 $j$ 个分量由克罗内克函数（Kronecker delta）定义：



$$
(\mathbf{e}_i)_j = \begin{cases} 1 & \text{若 } j = i \\ 0 & \text{若 } j \neq i \end{cases}
$$



---

### 2. 词嵌入矩阵（$\mathbf{E}$）

整个大模型的词汇总地图，存储在一张巨大的二维可学习参数矩阵中，称为**词嵌入矩阵**（Embedding Matrix），记作 $\mathbf{E}$：



$$
\mathbf{E} \in \mathbb{R}^{|V| \times d}
$$



其中：
- $|V|$ 是词表中词元的总数量（即矩阵的行数）。
- $d$ 是**词嵌入维度**（Embedding Dimension）或称**隐藏层大小**（Hidden Size，即矩阵的列数）。

矩阵 $\mathbf{E}$ 的每一行 $\mathbf{E}[i, :]$，就代表了第 $i$ 个词元在连续语义空间里的稠密向量坐标：



$$
\mathbf{E} = \begin{bmatrix}
\text{---} & \mathbf{x}_1^\top & \text{---} \\
\text{---} & \mathbf{x}_2^\top & \text{---} \\
& \vdots & \\
\text{---} & \mathbf{x}_{|V|}^\top & \text{---}
\end{bmatrix} \in \mathbb{R}^{|V| \times d}
$$



---

### 3. 嵌入查表方程

神经网络究竟是如何为第 $i$ 个词提取出连续坐标向量 $\mathbf{x}_i \in \mathbb{R}^d$ 的？

在严谨的线性代数中，查表操作被精确定义为转置独热向量 $\mathbf{e}_i^\top$ 与嵌入矩阵 $\mathbf{E}$ 的矩阵乘法：



$$
\mathbf{x}_i^\top = \mathbf{e}_i^\top \mathbf{E} \in \mathbb{R}^{1 \times d}
$$



若采用标准教科书的列向量表示法，则为：



$$
\mathbf{x}_i = \mathbf{E}^\top \mathbf{e}_i \in \mathbb{R}^{d \times 1}
$$



由于 $\mathbf{e}_i$ 除了第 $i$ 个位置为 1 之外其余分量全为 0，这个矩阵乘法在物理上就像一个**光学选通开关**：它把矩阵里的所有其他行全部清零，精准地把第 $i$ 行完整提取出来：

<figure>
<pre>
独热向量 e_i^T (1 × |V|)             嵌入矩阵 E (|V| × d)                  输出向量 x_i (1 × d)
┌───────────────────────────────┐     ┌───────────────────────────────────┐     ┌────────────────────────┐
│ [ 0,  ...,  1,  ...,  0 ]     │  ×  │ 第 1 行:  [  0.12,  -0.45,  ... ] │  =  │ [  0.95,   0.15,  ... ] │
└──────────────┬────────────────┘     │ ...                               │     └────────────────────────┘
               │                      │ 第 i 行:  [  0.95,   0.15,  ... ] │  ◄── 精准抽取出第 i 行！
               └──────────────────────► ...                               │
                                      │ 第|V|行:  [ -0.80,   0.21,  ... ] │
                                      └───────────────────────────────────┘
</pre>
<figcaption><strong>图 1.1:</strong> 独热向量矩阵乘法查表：孤立的数字 1 充当了行选择器。</figcaption>
</figure>

!!! note "注解: 工程实践注解：查表 vs 矩阵乘法"
    在纯粹的数学理论中，词查表被写作 $\mathbf{e}_i^\top \mathbf{E}$，这是为了保持神经网络作为端到端可微分矩阵映射系统的优雅与统一。

    但在生产代码（如 PyTorch 的 `torch.nn.Embedding`）中，在 GPU 上做 100,000 维的稀疏向量乘法是极大的算力浪费。因此底层硬件库直接通过数组内存索引 `E[i]` 在 $O(1)$ 常数时间内完成提取，两者数学结果分毫不差。

---

### 数学符号拆解速查表

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 1.1:</strong> 词嵌入向量的形式化数学符号、维度与物理含义</caption>
  <thead>
    <tr bgcolor="#f0f0f0">
      <th align="left">符号</th>
      <th align="left">正式名称</th>
      <th align="left">形状 / 维度</th>
      <th align="left">3岁小孩的理解</th>
      <th align="left">具象实例</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>$V$</td>
      <td>词汇集合（Vocabulary）</td>
      <td>词元构成的集合</td>
      <td>装着所有已知单词的积木箱</td>
      <td>$V = \{\text{"cat"}, \text{"kitten"}, \text{"dog"}, \text{"airplane"}\}$</td>
    </tr>
    <tr>
      <td>$|V|$</td>
      <td>词表大小（Cardinality）</td>
      <td>标量整数</td>
      <td>积木箱里总共有几块积木</td>
      <td>玩具模型中 $|V| = 4$（现代大模型中为 $32{,}000 \sim 128{,}000$）</td>
    </tr>
    <tr>
      <td>$d$</td>
      <td>嵌入维度（Hidden Size）</td>
      <td>标量整数</td>
      <td>草地操场地图上总共有几条粉笔方向</td>
      <td>玩具算例中 $d = 2$（LLaMA-3-8B 中 $d = 4{,}096$）</td>
    </tr>
    <tr>
      <td>$\mathbf{e}_i$</td>
      <td>独热向量（One-Hot）</td>
      <td>$\mathbb{R}^{|V| \times 1}$</td>
      <td>只按下了第 $i$ 个开关的配电盘</td>
      <td>$\mathbf{e}_2 = [0, 1, 0, 0]^\top$（选中第2个词）</td>
    </tr>
    <tr>
      <td>$\mathbf{E}$</td>
      <td>嵌入矩阵（Embedding Matrix）</td>
      <td>$\mathbb{R}^{|V| \times d}$</td>
      <td>登记了每个玩具在操场上坐标的门牌名册</td>
      <td>形状为 $4 \times 2$ 的参数表格</td>
    </tr>
    <tr>
      <td>$\mathbf{x}_i^\top$（或 $\mathbf{x}_i$）</td>
      <td>词嵌入向量（Embedding Vector）</td>
      <td>$\mathbb{R}^{1 \times d}$（行）/ $\mathbb{R}^{d \times 1}$（列）</td>
      <td>某个具体单词在操场上的 GPS 坐标</td>
      <td>$\mathbf{x}_{\text{kitten}}^\top = [0.95, 0.15]$</td>
    </tr>
  </tbody>
</table>

<br>

<details>
<summary><strong>符号词汇表（定义列表）</strong></summary>

<dl>
  <dt><strong>稠密向量（Dense Vector）</strong></dt>
  <dd>绝大部分分量均为非零实数（$\mathbb{R}$）的向量，能够用紧凑的维度编码极度丰富的多维连续语义信息。</dd>
  
  <dt><strong>稀疏向量（Sparse Vector）</strong></dt>
  <dd>绝大部分分量全部为零的向量（如 100,000 维里仅包含一个 1 其余全为 0 的独热向量）。</dd>
  
  <dt><strong>嵌入空间（Embedding Space, $\mathbb{R}^d$）</strong></dt>
  <dd>由 $d$ 个连续坐标轴张成的几何空间，自然语言中的概念以空间点的形式分布其中。</dd>
  
  <dt><strong>克罗内克函数（Kronecker Delta, $\delta_{ij}$）</strong></dt>
  <dd>一种二值分段函数：当下标 $i=j$ 时取值为 1，当下标 $i \neq j$ 时取值为 0。</dd>
</dl>
</details>

---

## Step 4: 公式从何而来？ {: #step-4 }

!!! quote "历史渊源与设计必然: 历史演进之路：从弗斯到 Word2Vec"
    人工智能研究者是如何领悟到“单词可以作为地图上的几何点”这一真谛的？

### 1. 分布假说（1957）
1957年，英国语言学家约翰·鲁珀特·弗斯（John Rupert Firth）写下了语言学史上最著名的论断之一：

> *“观其伴，知其意。”（You shall know a word by the company it keeps.）*  
> &mdash; **J. R. Firth (1957)**

弗斯敏锐地指出，含义相似的词必然会频繁出现在相似的上下文语境中：
- <kbd>“毛茸茸的【小猫咪】在小碟子里喝着温牛奶。”</kbd>
- <kbd>“毛茸茸的【猫】在小碟子里喝着温牛奶。”</kbd>

既然 <kbd>“kitten”</kbd> 和 <kbd>“cat”</kbd> 的前后邻居几乎完全一致，那么它们在数学地图上的坐标就理所应当紧紧挨在一起。

### 2. 独热编码的正交性灾难
在稠密嵌入诞生前，早期的计算语言学完全依赖独热向量 $\mathbf{e}_i$。

这引发了致命的数学缺陷——**正交性灾难（Orthogonality Catastrophe）**：

任取两个不同的独热单词 $\mathbf{e}_i$ 和 $\mathbf{e}_j$（其中 $i \neq j$），计算它们的点积：



$$
\mathbf{e}_i \cdot \mathbf{e}_j = \sum_{k=1}^{|V|} (\mathbf{e}_i)_k (\mathbf{e}_j)_k = 0
$$



在线性代数中，非零向量点积为零意味着它们彼此**垂直（正交）**。

在独热空间里，每一个词都与其他所有词保持着冰冷的 90 度垂直直角：
- <kbd>“小狗”</kbd> 到 <kbd>“狗”</kbd> 的欧氏距离：$\sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1.414$
- <kbd>“小狗”</kbd> 到 <kbd>“核反应堆”</kbd> 的欧氏距离：$\sqrt{1^2 + (-1)^2} = \sqrt{2} \approx 1.414$

计算机对语义毫无感知：在它眼里，一只小狗和小狗的亲近程度，竟然和一只小狗与一座核电站完全一样！

### 3. 稠密向量革命：Word2Vec（2013）
2013年，Tomas Mikolov 带领 Google 团队提出了轰动学术界的 **Word2Vec**。他们抛弃了庞大正交的独热编码，训练神经网络把词汇压缩进低维连续的稠密空间（如 $d=300$）。

当研究人员把这些向量投影出来时，整个 AI 界都被震撼了：高维几何空间竟然自发呈现出平行的语义向量结构！

其中最举世闻名的发现莫过于向量类比代数：



$$
\mathbf{x}_{\text{king}} - \mathbf{x}_{\text{man}} + \mathbf{x}_{\text{woman}} \approx \mathbf{x}_{\text{queen}}
$$



从“国王”向量中减去“男人”向量，剔除了“男性特征”，剩下的纯粹是“王权君主”的概念；再加上“女人”向量，计算出的坐标分毫不差地落在了“女王”的身旁！

在现代 Transformer 中，嵌入矩阵 $\mathbf{E}$ 是在海量数据预训练中通过反向传播自动学出的，最终凝聚为人类知识的连续几何宇宙。

---

## Step 5: 具象微型算例（纸笔手算验证） {: #step-5 }

!!! tip "超简单玩具算例: 具象微型算例：2维宠物与载具操场"
    让我们用具体的微型数字在纸上一步步完成手算。

### 场景设定
- **微型词表（$|V| = 4$）**：
  1. $w_1 = \text{"cat"}$
  2. $w_2 = \text{"kitten"}$
  3. $w_3 = \text{"dog"}$
  4. $w_4 = \text{"airplane"}$

- **嵌入维度（$d = 2$）**：
  - **第 1 维（$x_1$）&mdash; “毛茸茸 / 软萌度”**：衡量概念的可爱温和程度（$0.0 = \text{冰冷金属}, 1.0 = \text{极致蓬松}$）。
  - **第 2 维（$x_2$）&mdash; “物理体型”**：衡量现实物体的体积极限（$0.0 = \text{掌心大小}, 1.0 = \text{庞然巨机}$）。

### 嵌入矩阵（$\mathbf{E} \in \mathbb{R}^{4 \times 2}$）



$$
\mathbf{E} = \begin{bmatrix}
0.90 & 0.30 \\
0.95 & 0.15 \\
0.85 & 0.55 \\
0.02 & 0.98
\end{bmatrix} \quad \begin{matrix}
\leftarrow \text{第 1 行: "cat"} \\
\leftarrow \text{第 2 行: "kitten"} \\
\leftarrow \text{第 3 行: "dog"} \\
\leftarrow \text{第 4 行: "airplane"}
\end{matrix}
$$



---

### 为 $w_2 = \text{"kitten"}$ 查表的逐步计算

单词 <kbd>"kitten"</kbd>（索引号为 2）的独热向量为：



$$
\mathbf{e}_2^\top = \begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix}
$$



将 $\mathbf{e}_2^\top$ 与矩阵 $\mathbf{E}$ 相乘：



$$
\mathbf{x}_{\text{kitten}}^\top = \mathbf{e}_2^\top \mathbf{E} = \begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix}
0.90 & 0.30 \\
0.95 & 0.15 \\
0.85 & 0.55 \\
0.02 & 0.98
\end{bmatrix}
$$



手算每个坐标：



$$
\text{坐标 } 1 = (0 \times 0.90) + (1 \times 0.95) + (0 \times 0.85) + (0 \times 0.02) = \mathbf{0.95}
$$





$$
\text{坐标 } 2 = (0 \times 0.30) + (1 \times 0.15) + (0 \times 0.55) + (0 \times 0.98) = \mathbf{0.15}
$$





$$
\mathbf{x}_{\text{kitten}}^\top = [0.95, 0.15]
$$



独热选择器精确无误地提取出了矩阵的第 2 行！

---

### 词向量坐标及语义对照表

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 1.2:</strong> 空间坐标、可视化特征标尺与现实语义档案</caption>
  <thead>
    <tr bgcolor="#f0f0f0">
      <th align="left">词元（$w_i$）</th>
      <th align="center">行索引</th>
      <th align="right">毛茸茸度（$x_1$）</th>
      <th align="center">毛茸茸标尺</th>
      <th align="right">体型大小（$x_2$）</th>
      <th align="center">体型标尺</th>
      <th align="left">语义特征剖析</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><kbd>"cat"</kbd></td>
      <td align="center">1</td>
      <td align="right">0.90</td>
      <td align="center"><meter min="0" max="1" value="0.90">90%</meter></td>
      <td align="right">0.30</td>
      <td align="center"><meter min="0" max="1" value="0.30">30%</meter></td>
      <td align="left">十分软萌，中小型宠物</td>
    </tr>
    <tr>
      <td align="left"><kbd>"kitten"</kbd></td>
      <td align="center">2</td>
      <td align="right"><mark><strong>0.95</strong></mark></td>
      <td align="center"><meter min="0" max="1" value="0.95">95%</meter></td>
      <td align="right"><mark><strong>0.15</strong></mark></td>
      <td align="center"><meter min="0" max="1" value="0.15">15%</meter></td>
      <td align="left">极致软萌，巴掌大幼崽</td>
    </tr>
    <tr>
      <td align="left"><kbd>"dog"</kbd></td>
      <td align="center">3</td>
      <td align="right">0.85</td>
      <td align="center"><meter min="0" max="1" value="0.85">85%</meter></td>
      <td align="right">0.55</td>
      <td align="center"><meter min="0" max="1" value="0.55">55%</meter></td>
      <td align="left">毛发温热，中型伴侣动物</td>
    </tr>
    <tr>
      <td align="left"><kbd>"airplane"</kbd></td>
      <td align="center">4</td>
      <td align="right">0.02</td>
      <td align="center"><meter min="0" max="1" value="0.02">2%</meter></td>
      <td align="right">0.98</td>
      <td align="center"><meter min="0" max="1" value="0.98">98%</meter></td>
      <td align="left">毫无毛发，巨型飞行机器</td>
    </tr>
  </tbody>
</table>

---

### 空间几何距离验算

在这个 2 维操场地图上，词语之间的间隔有多远？我们采用经典的**欧几里得距离公式**：



$$
\text{Distance}(\mathbf{u}, \mathbf{v}) = \sqrt{(u_1 - v_1)^2 + (u_2 - v_2)^2}
$$



#### 1. “kitten”（小猫咪）与“cat”（猫）之间的空间距离：



$$
\begin{aligned}
\text{Distance}(\text{kitten}, \text{cat}) &= \sqrt{(0.95 - 0.90)^2 + (0.15 - 0.30)^2} \\
&= \sqrt{(0.05)^2 + (-0.15)^2} \\
&= \sqrt{0.0025 + 0.0225} \\
&= \sqrt{0.0250} \\
&\approx \mathbf{0.158}
\end{aligned}
$$



#### 2. “kitten”（小猫咪）与“airplane”（飞机）之间的空间距离：



$$
\begin{aligned}
\text{Distance}(\text{kitten}, \text{airplane}) &= \sqrt{(0.95 - 0.02)^2 + (0.15 - 0.98)^2} \\
&= \sqrt{(0.93)^2 + (-0.83)^2} \\
&= \sqrt{0.8649 + 0.6889} \\
&= \sqrt{1.5538} \\
&\approx \mathbf{1.246}
\end{aligned}
$$



对比两者相距的距离：



$$
\frac{1.246}{0.158} \approx \mathbf{7.89\times \text{（飞机足足远了近 8 倍！）}}
$$



<kbd>"airplane"</kbd> 到 <kbd>"kitten"</kbd> 的距离，是 <kbd>"cat"</kbd> 到 <kbd>"kitten"</kbd> 距离的 **近 8 倍**！

---

### 2维操场地图几何点阵示意图

<figure>
<pre>
体型大小 (第2维)
  ▲
  │
1.00│  ["airplane"] (0.02, 0.98)
  │     (庞大的高空金属机器)
0.80│
  │
0.60│                                     ["dog"] (0.85, 0.55)
  │
0.40│                                     ["cat"] (0.90, 0.30)
  │                                           │ (距离 ≈ 0.158)
0.20│                                     ["kitten"] (0.95, 0.15)
  │                                       (毛茸茸萌宠聚合簇！)
0.00└─────────────────────────────────────────────────────────────► 毛茸茸度 (第1维)
   0.00        0.20        0.40        0.60        0.80       1.00
</pre>
<figcaption><strong>图 1.2:</strong> 2维几何操场地图。注意看，“cat”、“kitten” 和 “dog” 紧密抱团形成萌宠聚类，而 “airplane” 孤零零落在左上方。</figcaption>
</figure>

---

## Step 6: 核心精髓 {: #step-6 }

!!! tip "核心要点: 核心精髓"
    嵌入矩阵 $\mathbf{E}$ 是大语言模型的**通用语言转译桥梁**：它将孤立离散的人类文字符号，映射为多维空间中具有生命力的几何坐标。

    通过让相似含义的词在空间中比邻而居，神经网络就能单纯凭借纯粹的线性代数来理解概念、类比常识和推演语境。

    但这引发了下一个关键的数学问题：**当大语言模型需要衡量两个词之间的关联程度到底有多紧密时，它在底层究竟使用了哪把精准的数学标尺？**

---

