# 第 21 章：显存之墙（KV Cache 机制与 GQA / MLA 压缩数学）

> [!INTUITION] 步骤 1：3 岁小孩直觉
> 想象你是一个在黑板前写长篇作文的小学生。
> 
> 如果你手头没有任何便签草稿本，那么每当你打算写下第 50 个词时，你都必须立刻停笔，跑去图书馆把前 1 到 49 个词从头到尾重读一遍，重新理解整篇文章的含义，然后再写下第 50 个词。写第 51 个词时，你又得跑一趟图书馆把前 50 个词再重读一遍。还没写完 10 页纸，你的双腿就已经累瘫，放学铃声响了，你连一个自然段都没写完。
> 
> 为了拯救你的双腿，老师给了你一个随身小本子——这就是 **键值缓存（KV Cache）**。
> 每当你写完一个新词，你就把这个词的线索卡片（Key）与上下文含义（Value）直接抄在桌上的小本子里。写第 51 个词时，你只需看一眼桌上的小本子，再也不用往图书馆跑了。
> 
> 但这引发了一个新的危机：如果你的作文写了 100 页呢？
> 你的课桌上堆满了成千上万本草稿本。渐渐地，草稿本从地板一直堆到天花板，挤满了整间教室，你连坐进椅子的空间都没有了！这堵由堆积如山的草稿本筑起的物理高墙，就是 **显存之墙（The Memory Wall）**。

---

## 步骤 2：承前启后的关键过渡

我们如何把小学生桌上的“便签草稿本”，转化成严谨的张量切片、计算其在 GPU 显存（<abbr title="高带宽显存">HBM</abbr>）中的精确字节数，并通过数学投影将草稿本缩小 8 倍而不丢失文章记忆？

在 Transformer 自回归生成中，前序已经生成的词元向量 $\mathbf{x}_1, \dots, \mathbf{x}_{t-1}$ 一旦确定就永远不会改变。如果不作缓存，每次计算第 $t$ 步时都需要对所有前序词元重复计算键（Key）和值（Value）投影，生成长度为 $T$ 的文本总共需要 $O(T^3)$ 的立方级总算力，导致长文本生成在计算上彻底瘫痪。

承前启后的关键过渡问题是：
$$\text{如何将历史键和值优雅地持久化存储，使得每步生成的计算复杂度从 } O(t^2) \text{ 降至 } O(t)\text{？这一缓存结构在物理显存中究竟占据多大空间？}$$

---

## 步骤 3：严谨数学公式与推导

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        自回归生成中的 KV CACHE 动态累加                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 时间步 t：                                                                             │
│ 当前最新词元：     x_t ──► [ W_q ] ──► q_t  [1 x d_head]                               │
│                    x_t ──► [ W_k ] ──► k_t  [1 x d_head] ──┐                           │
│                    x_t ──► [ W_v ] ──► v_t  [1 x d_head] ──┼──┐                        │
│                                                            │  │                        │
│ 历史 KV 缓存：                                             ▼  │                        │
│   K_{1:t-1} [(t-1) x d_head] ─────────────► 张量拼接(Cat) ─► K_{1:t} [t x d_head]      │
│                                                               │                        │
│   V_{1:t-1} [(t-1) x d_head] ─────────────► 张量拼接(Cat) ─► V_{1:t} [t x d_head] ◄─┘  │
│                                                               │                        │
│ 注意力输出向量：                                              ▼                        │
│   o_t = softmax( (q_t @ K_{1:t}.T) / sqrt(d_head) ) @ V_{1:t}  [1 x d_head]           │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 21.1:</strong> KV Cache 迭代追加机制。在时间步 t，仅需投影最新词元的单行 q, k, v 向量；k 和 v 追加存入历史缓存矩阵中。</figcaption>
</figure>

### 1. 无缓存生成的计算灾难

在标准自注意力中，对于长度为 $t$ 的完整序列：
$$
\mathbf{Q}_t = \mathbf{X}_{1:t} \mathbf{W}_Q, \quad \mathbf{K}_t = \mathbf{X}_{1:t} \mathbf{W}_K, \quad \mathbf{V}_t = \mathbf{X}_{1:t} \mathbf{W}_V
$$
$$\mathbf{O}_t = \operatorname{softmax}\left(\frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}}\right) \mathbf{V}_t$$

若不保存中间激活值，在生成第 $t$ 个词时，前序所有词元的注意力投影都要从头算起。对于长度为 $T$ 的生成序列：
$$
\text{总浮点运算量} = \sum_{t=1}^T 2 P \cdot t \propto O(T^2 \cdot d_{\text{model}})
$$
叠加自注意力内部矩阵乘法后，整条序列生成的总体时间复杂度高达 $O(T^3)$。

### 2. KV Cache 递归递推公式

由于前序词元 $\mathbf{x}_1, \dots, \mathbf{x}_{t-1}$ 的表示形式具有时间不变性：
$$
\mathbf{k}_\tau = \mathbf{x}_\tau \mathbf{W}_K, \quad \mathbf{v}_\tau = \mathbf{x}_\tau \mathbf{W}_V \quad (\forall \tau < t)
$$

在时间步 $t$，模型**仅仅**对最新的单行词元向量 $\mathbf{x}_t \in \mathbb{R}^{1 \times d_{\text{model}}}$ 进行一次矩阵乘法投影：

$$
\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_{\text{head}}}
$$
$$
\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K \in \mathbb{R}^{1 \times d_{\text{head}}}
$$
$$
\mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V \in \mathbb{R}^{1 \times d_{\text{head}}}
$$

随后将 $\mathbf{k}_t$ 与 $\mathbf{v}_t$ 沿着时序维度拼接（Concatenate）追加到历史缓存中：

$$
\mathbf{K}_{1:t} = \begin{bmatrix} \mathbf{K}_{1:t-1} \\ \mathbf{k}_t \end{bmatrix} \in \mathbb{R}^{t \times d_{\text{head}}}, \quad \mathbf{V}_{1:t} = \begin{bmatrix} \mathbf{V}_{1:t-1} \\ \mathbf{v}_t \end{bmatrix} \in \mathbb{R}^{t \times d_{\text{head}}}
$$

当前步的注意力输出仅需一次向量-矩阵乘法：

$$
\mathbf{o}_t = \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{1:t}^\top}{\sqrt{d_{\text{head}}}}\right) \mathbf{V}_{1:t} \in \mathbb{R}^{1 \times d_{\text{head}}}
$$

单步计算复杂度瞬间从 $O(t^2)$ 压降至 $O(t)$！

---

### 3. KV Cache 显存容量精确计算公式

设：
- $n_{\text{layers}}$：模型的 Transformer 堆叠总层数。
- $n_{\text{kv\_heads}}$：每层中 Key/Value 注意力头的实际数量。
- $d_{\text{head}}$：每个注意力头的内部特征维度。
- $T_{\text{ctx}}$：总上下文长度（提示词词元数 + 已生成词元数）。
- $b$：并发批处理大小（服务同时处理的并发请求数）。
- $p$：每个浮点数所占用的字节数（FP16/BF16 取 $p = 2$，FP8 取 $p = 1$，INT4 取 $p = 0.5$）。

KV Cache 所消耗的物理显存总量 $\text{RAM}_{\text{KV}}$ 精确为：

$$
\text{RAM}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{ctx}} \times b \times p \quad [\text{Bytes}]
$$

公式开头的常数 $2$ 严格对应于同时缓存的两张独立张量：**键（Key）张量与值（Value）张量**。

---

### 4. 显存墙危机：128k 超长上下文下的显存暴击

以当今工业界主流标杆模型 **LLaMA-3-70B** 为例：
- 模型层数：$n_{\text{layers}} = 80$
- 隐藏维度：$d_{\text{model}} = 8192$
- Query 头数：$H_q = 64$
- 每一头的维度：$d_{\text{head}} = 128$
- 数值精度：16 位半精度（$p = 2\text{ 字节}$）
- 上下文窗口：$T_{\text{ctx}} = 128{,}000\text{ 词元}$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 21.1:</strong> 单用户并发（$b=1$）在 128k 上下文下，各注意力架构的 KV Cache 显存占用对比（LLaMA-3-70B 规模）。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">注意力架构</th>
      <th align="center">Query 头数（$H_q$）</th>
      <th align="center">KV 头数（$H_{\text{kv}}$）</th>
      <th align="center">缓存压缩比例</th>
      <th align="right">单用户 128k 缓存显存开销</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>标准多头注意力（MHA）</strong></td>
      <td align="center">64</td>
      <td align="center">64</td>
      <td align="center">$1\times$（基线）</td>
      <td align="right"><strong>312.50 GB</strong>（直接撑爆显存！）</td>
    </tr>
    <tr>
      <td><strong>分组查询注意力（GQA）</strong></td>
      <td align="center">64</td>
      <td align="center">8</td>
      <td align="center"><strong>$8\times$ 压缩</strong></td>
      <td align="right"><strong>39.06 GB</strong>（单卡可容纳）</td>
    </tr>
    <tr>
      <td><strong>多查询注意力（MQA）</strong></td>
      <td align="center">64</td>
      <td align="center">1</td>
      <td align="center"><strong>$64\times$ 压缩</strong></td>
      <td align="right"><strong>4.88 GB</strong></td>
    </tr>
    <tr>
      <td><strong>多头潜在注意力（MLA）</strong></td>
      <td align="center">128</td>
      <td align="center">潜在维度 $d_c=512$</td>
      <td align="center"><strong>低秩潜在投影</strong></td>
      <td align="right"><strong>~11.20 GB</strong>（DeepSeek-V2/V3）</td>
    </tr>
  </tbody>
</table>

如果采用传统的标准多头注意力（MHA），仅仅为一个拥有 128k 长度的用户提供服务，**光是存放 KV Cache 就需要 312.5 GB 显存**！这相当于需要将近 4 张 80GB 的 NVIDIA A100 GPU 仅仅用来装载这一名用户的对话记忆！

---

### 5. 架构级压缩救星：MQA、GQA 与 DeepSeek MLA

为了击碎这堵显存墙，学术界与工业界相继发明了三代革命性的架构级压缩方案：

#### A. 分组查询注意力（GQA）
不再为每个 Query 头独立分配专属的 Key/Value 头，而是将 $H_q$ 个 Query 头均匀划分为 $H_{\text{kv}}$ 个组（组大小 $G = H_q / H_{\text{kv}}$）：

$$
\text{kv\_idx}(h) = \left\lfloor \frac{h}{G} \right\rfloor
$$

在保留 Query 多样化表征能力的同时，将 KV 缓存体积直接削减至原来的 $\frac{1}{G}$（LLaMA-3 中直降 8 倍）。

#### B. DeepSeek 多头潜在注意力（MLA）
DeepSeek-V2 与 DeepSeek-V3 彻底革新了缓存范式。模型不再独立缓存各个头的 Key 和 Value，而是将隐藏状态 $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ 投影至一个极小维度的共享低秩潜在向量 $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$（其中 $d_c \ll H_q \cdot d_{\text{head}}$）：

$$
\mathbf{c}_t^{KV} = \mathbf{x}_t \mathbf{W}_{DKV} \in \mathbb{R}^{1 \times d_c}
$$

在推理时，服务引擎**仅仅在显存中缓存低维潜在向量 $\mathbf{c}_t^{KV}$**（外加一个解耦的 64 维 RoPE 旋转位置编码向量 $\mathbf{k}_t^R$）。多头 Key 和 Value 的还原利用矩阵结合律直接吸收进 Query 端的投影运算中：

$$
\mathbf{q}_{t, h} \mathbf{k}_{\tau, h}^\top = \mathbf{q}_{t, h} (\mathbf{c}_\tau^{KV} \mathbf{W}_{UK, h})^\top = (\mathbf{q}_{t, h} \mathbf{W}_{UK, h}^\top) \mathbf{c}_\tau^{KV \top}
$$

通过只存低维隐变量 $\mathbf{c}_t^{KV}$，DeepSeek MLA 将 KV 显存占用压低到传统 MHA 的极小零头，以极高效率支撑起百万级超长上下文。

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2019 年">2019 年</time> &mdash; <strong>多查询注意力（MQA）诞生</strong>（<cite>Noam Shazeer，Google</cite>）</dt>
  <dd>Transformer 奠基人之一 Noam Shazeer 发表著名论文《Fast Transformer Decoding: One Write-Head is All You Need》，首次证明所有 Query 头可以共享单一对 KV 头，大幅砍去 95% 的解码访存开销。</dd>
  <dt><time datetime="2023 年">2023 年</time> &mdash; <strong>分组查询注意力（GQA）成为通用标准</strong>（<cite>Ainslie 等，Google Research</cite>）</dt>
  <dd>Google 团队提出 GQA，完美平衡了 MHA 的强表征能力与 MQA 的低显存开销，随即被 LLaMA-2-70B、LLaMA-3、Mistral 及 Gemma 2 普遍采纳。</dd>
  <dt><time datetime="2024 年">2024 年</time> &mdash; <strong>多头潜在注意力（MLA）横空出世</strong>（<cite>DeepSeek-AI</cite>）</dt>
  <dd>DeepSeek 在 DeepSeek-V2 与 V3 中推出 MLA，将低秩矩阵压缩技术深度融入注意力内核，从根本上重塑了长上下文大模型推理的显存经济学。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

我们通过一组极简参数，手算 KV Cache 的拼接过程与注意力分数计算。

### 微型模型配置
- 层数：$n_{\text{layers}} = 1$
- Query 头数：$H_q = 2$
- Key/Value 头数：$H_{\text{kv}} = 1$（组大小 $G = 2$）
- 头维度：$d_{\text{head}} = 2$
- 精度：16 位半精度（$p = 2\text{ 字节}$）

---

### 第 1 步：摄入提示词词元 1（$t = 1$）
词元 1 投影生成其 Key 和 Value 向量：
$$
\mathbf{k}_1 = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix} 0.5 & 1.5 \end{bmatrix}
$$

初始化缓存矩阵：
$$
\mathbf{K}_{1:1} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}, \quad \mathbf{V}_{1:1} = \begin{bmatrix} 0.5 & 1.5 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
$$
$$\text{显存开销} = (1 \text{ 词元} \times 2 \text{ 维度} \times 2 \text{ 张量}) \times 2 \text{ 字节} = 8\text{ Bytes}$$

---

### 第 2 步：生成新词元 2（$t = 2$）
词元 2 到达，仅投影自身单行向量：
$$
\mathbf{k}_2 = \begin{bmatrix} 3.0 & 0.0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2.0 & 1.0 \end{bmatrix}
$$
将其追加拼接进缓存：
$$
\mathbf{K}_{1:2} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}, \quad \mathbf{V}_{1:2} = \begin{bmatrix} 0.5 & 1.5 \\ 2.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$
$$\text{当前缓存显存} = (2 \text{ 词元} \times 2 \text{ 维度} \times 2 \text{ 张量}) \times 2 \text{ 字节} = 16\text{ Bytes}$$

设 Query 头 1 在当前步产生的查询向量为：
$$
\mathbf{q}_{2, 1} = \begin{bmatrix} 1.0 & 0.0 \end{bmatrix}
$$

跨历史缓存计算注意力未归一化打分：
$$
\mathbf{s} = \frac{\mathbf{q}_{2, 1} \mathbf{K}_{1:2}^\top}{\sqrt{2}} = \frac{\begin{bmatrix} 1.0 & 0.0 \end{bmatrix} \begin{bmatrix} 1.0 & 3.0 \\ 2.0 & 0.0 \end{bmatrix}}{\sqrt{2}} = \frac{\begin{bmatrix} 1.0 & 3.0 \end{bmatrix}}{\sqrt{2}} \approx \begin{bmatrix} 0.707 & 2.121 \end{bmatrix}
$$

执行 Softmax 概率归一化：
$$
e^{0.707} \approx 2.028, \quad e^{2.121} \approx 8.339 \implies \text{分母和} = 10.367
$$
$$
\mathbf{a} = \operatorname{softmax}(\mathbf{s}) \approx \begin{bmatrix} \frac{2.028}{10.367} & \frac{8.339}{10.367} \end{bmatrix} \approx \begin{bmatrix} 0.196 & 0.804 \end{bmatrix}
$$

对缓存中的 Value 矩阵进行加权聚合：
$$
\mathbf{o}_{2, 1} = \mathbf{a} \mathbf{V}_{1:2} = \begin{bmatrix} 0.196 & 0.804 \end{bmatrix} \begin{bmatrix} 0.5 & 1.5 \\ 2.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.098 + 1.608 & 0.294 + 0.804 \end{bmatrix} = \begin{bmatrix} 1.706 & 1.098 \end{bmatrix}
$$

见证奇迹的时刻：**我们完全没有重新计算词元 1 的 Key 与 Value！** 历史信息直接从缓存读出，没有任何多余的重复算力开销。

---

## 步骤 6：核心精髓总结

<fieldset>
<legend><strong>核心教学要点</strong></legend>
<p><strong>KV Cache</strong> 实现了典型的“以空间换时间”，将大模型自回归解码的单步算力开销从灾难性的 $O(t^2)$ 砍至线性的 $O(t)$。</p>
<p>但由于缓存体积随上下文长度 $T_{\text{ctx}}$ 刚性线性膨胀，长文本推理必然遭遇冰冷残酷的 <strong>显存之墙（Memory Wall）</strong>。现代前沿架构通过 <strong>分组查询注意力（GQA）</strong> 与 <strong>多头潜在注意力（MLA）</strong>，在算法结构层面对缓存实施了 $8\times$ 至 $20\times$ 的降维压缩，成为了撑起百万级长窗口服务的最核心支柱。</p>
</fieldset>
