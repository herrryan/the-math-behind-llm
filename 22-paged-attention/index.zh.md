# 第 22 章：分页注意力与虚拟显存（PagedAttention 与零碎片革命）

> [!INTUITION] 步骤 1：3 岁小孩直觉
> 想象你开了一家拥有 500 间客房的大酒店。
> 
> 一位旅行团导游冲进大堂，大声喊道：*“我们的旅行团刚刚进城！我们可能只需要 5 间房，但如果所有远房亲戚都来，我们最多可能需要 500 间房！而且，我们所有的房间必须在同一层楼、连成一条绝对不许中断的直线长廊！”*
> 
> 为了满足他最坏情况下“必须连续占用 500 间房”的要求，你只能把 1 楼到 5 楼的整整 500 间房全部上锁留给他。然而实际上，最后只来了 10 位游客！整整 490 间客房空空荡荡、铁将军把门。此时酒店门外的暴雨中，站满了想要入住的其他散客，但你只能无奈地把他们拒之门外，因为房间已经被“预订”了——这就是传统深度学习中的 **连续显存预分配（Contiguous Allocation）**。
> 
> 后来，一位聪明的酒店经理发明了动态智能房卡：**分页注意力（PagedAttention）**。
> 经理对导游说：*“你不需要一开始就霸占 500 间房，你们的房间也完全不需要挨在一起！这是 102 房的钥匙；待会儿你下一个朋友到了，我再给他 307 房；再下一位到了，我给他 412 房。我的电脑系统能清清楚楚记录你们团所有人的房号，大家随时能自由走动串门。”*
> 
> 顷刻之间，酒店里再也没有任何一间被浪费的闲置客房，每一间空房都能随时接待新客人，同一家酒店能接待的游客总数暴增了 4 倍！

---

## 步骤 2：承前启后的关键过渡

我们如何将这位酒店经理的“动态房卡系统”，转化为 GPU 显存底层的张量寻址指令，让注意力计算核心能够在零额外开销下直接读取物理上不连续的内存块？

在 PyTorch 等原生深度学习框架中，张量在物理显存中必须占据**一块严格连续的内存空间**。如果一个用户的生成请求最多可能产生 $T_{\text{max}} = 2048$ 个词元，推理引擎就不得不一开始就预先分配一个形状为 $[B, 2, n_{\text{layers}}, n_{\text{kv\_heads}}, T_{\text{max}}, d_{\text{head}}]$ 的连续显存大数组。

因为实际生成的文本长度事先完全无法预知，这种机制导致了两种极其昂贵的显存浪费：
1. **内部碎片（Internal Fragmentation）**：为未来可能生成的词元预留、但最终根本没用上的大量空白显存。
2. **外部碎片（External Fragmentation）**：不同请求释放后遗留下的零碎显存空隙，因为空间太小而无法容纳任何新的连续 $T_{\text{max}}$ 请求。

在工业界真实生产环境中，**高达 60% 至 80% 的 GPU 显存被这种空白连续填充物白白浪费**！

承前启后的核心过渡问题是：
$$\text{当 Key 和 Value 张量被随意打散存储在显存中完全不连续的物理碎块中时，注意力核心如何直接计算 } \operatorname{softmax}\left(\frac{\mathbf{q} \mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}\text{？}$$

---

## 步骤 3：严谨数学公式与推导

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PAGEDATTENTION 虚拟显存映射架构                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 请求 1（提示词：7 个词元，逻辑块大小 B = 4）                                           │
│ 逻辑 KV 块（序列视角）：                                                               │
│   逻辑块 0：[ 词元 0, 词元 1, 词元 2, 词元 3 ] （已满）                                │
│   逻辑块 1：[ 词元 4, 词元 5, 词元 6,   ___  ] （偏移量 = 3）                          │
│                                                                                        │
│ 页表（Page Table - 请求 1）：                                                          │
│   逻辑块 0 ──► 物理显存块 7                                                            │
│   逻辑块 1 ──► 物理显存块 2                                                            │
│                                                                                        │
│ GPU HBM 显存物理块池（任意分散存储）：                                                 │
│   [ 物理块 0 ]   [ 物理块 1 ]   [ 物理块 2: 请求 1 (词元 4-6) ]   [ 物理块 3 ]         │
│   [ 物理块 4 ]   [ 物理块 5 ]   [ 物理块 6 ]                      [ 物理块 7: 请求 1 (0-3)]
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 22.1:</strong> PagedAttention 将逻辑上连续的时序词元映射到 GPU 物理显存中离散的页块中，彻底根除了显存碎片。</figcaption>
</figure>

### 1. 逻辑块与物理块的划分

将单个请求的时序 KV 向量序列划分为固定大小的 **逻辑块（Logical Blocks）**，设每个块容纳 $B$ 个词元（工业界通常取 $B = 16$ 或 $B = 32$）。

对于时序位置为 $t \in \{0, 1, \dots, T - 1\}$ 的任意词元：
- **逻辑块索引**：
  $$
  b_{\text{logical}} = \left\lfloor \frac{t}{B} \right\rfloor
  $$
- **块内时序偏移量**：
  $$
  o = t \bmod B
  $$

GPU 物理显存被统一划分为一个平坦的 **物理块池（Physical Block Pool）** $\mathcal{P} = \{P_0, P_1, \dots, P_{N-1}\}$。每个物理块的张量形状为：
$$
P_i \in \mathbb{R}^{B \times 2 \times n_{\text{kv\_heads}} \times d_{\text{head}}}
$$

### 2. 页表映射函数（The Page Table Mapping）

对于每个正在运行的服务请求 $r$，推理运行时维护一个动态数组——<dfn id="def-page-table-zh">页表（Page Table）</dfn> $\mathcal{T}_r$：

$$
\mathcal{T}_r(b_{\text{logical}}) = P_{\text{physical}} \in \mathcal{P}
$$

词元 $t$ 的 Key 或 Value 向量在物理显存中的实际字节基地址 $\text{Addr}(t)$ 为：

$$
\text{Addr}(t) = \text{BaseAddr}\left(\mathcal{T}_r\left(\left\lfloor \frac{t}{B} \right\rfloor\right)\right) + (t \bmod B) \times S_{\text{token}}
$$

其中 $S_{\text{token}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times p$ 为单词元所需的显存跨步字节数。

---

### 3. PagedAttention 算子执行机制

传统注意力算子要求 $\mathbf{K}$ 是单个连续内存矩阵。
在 PagedAttention 中，CUDA 算子通过传入的页表指针直接在内层循环中迭代物理块，在完全不复制数据的前提下流式完成分块注意力加权：

$$
\mathbf{o}_t = \sum_{j=0}^{\lceil t / B \rceil - 1} \operatorname{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{P_j}^\top}{\sqrt{d_{\text{head}}}}\right) \mathbf{V}_{P_j}
$$

其中 $P_j = \mathcal{T}_r(j)$ 是存放该请求第 $j$ 个逻辑块数据的具体物理显存块。

---

### 4. 显存碎片率的数学清零

设某请求的实际生成序列长度为 $T$，单块容积为 $B$。
- **外部碎片**：由于物理显存池中任何空闲块都可以分配给任意请求的任意位置，**外部碎片严格为 $0\%$**。
- **内部碎片**：由于只有序列最末尾的一个逻辑块可能未填满，显存浪费严格被限制在最末块：
  $$\text{浪费槽位数} \le B - 1$$
  其显存浪费率上限为：
  $$
  W_{\text{internal}} < \frac{B}{T}
  $$

若服务采用块大小 $B = 16$，上下文长度为 $T = 2048$：
$$
W_{\text{internal}} < \frac{16}{2048} \approx 0.78\%
$$

PagedAttention 将大模型推理的显存浪费率从原先的 **60%～80% 骤降至 1% 以下**！

---

### 5. 写时复制（Copy-on-Write, CoW）与零拷贝共享

在并行采样（Parallel Sampling）、束搜索（Beam Search）或公共系统提示词缓存（Prefix Caching）中，多个并发请求共享完全相同的前缀提示词。

PagedAttention 引入操作系统的 **写时复制（CoW）** 机制：
- 每个物理块维护一个原子引用计数器 $\text{ref\_count}(P_i) \in \mathbb{N}$。
- 当两个请求共享相同的 Prompt 块时，它们的页表直接指向**同一个物理块地址**：
  $$\mathcal{T}_{r_1}(0) = \mathcal{T}_{r_2}(0) = P_7 \implies \text{ref\_count}(P_7) = 2$$
- 只要请求没有向该共享块追加写入新内容，就不消耗任何额外显存（**零显存拷贝**）。当某请求需要写入新词元时，系统仅在引用计数 $>1$ 时触发实际物理拷贝，并更新该请求的独占页表指针。

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 22.1:</strong> 传统连续显存预分配与 PagedAttention（vLLM）的技术特性全景对比。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">对比维度</th>
      <th align="left">传统连续预分配（HuggingFace/原生）</th>
      <th align="left">分页注意力（PagedAttention / vLLM）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>显存分配模式</strong></td>
      <td>按可能的最长序列 $T_{\text{max}}$ 预先占满连续空间</td>
      <td>按需动态分配固定大小为 $B$ 的非连续物理页块</td>
    </tr>
    <tr>
      <td><strong>内部碎片浪费</strong></td>
      <td>60% – 80%（大量填充无效空白 Padding）</td>
      <td><strong>&lt; 1%</strong>（严格由 $(B-1)/T$ 界定）</td>
    </tr>
    <tr>
      <td><strong>外部碎片浪费</strong></td>
      <td>严重（释放后的小孔洞无法容纳新大请求）</td>
      <td><strong>严格为 0%</strong>（任意碎片空间均可复用）</td>
    </tr>
    <tr>
      <td><strong>提示词前缀共享</strong></td>
      <td>必须深拷贝多份完整张量，显存翻倍</td>
      <td><strong>零拷贝共享</strong>（通过页表多对一映射指针）</td>
    </tr>
    <tr>
      <td><strong>真实服务吞吐量</strong></td>
      <td>1x（基准）</td>
      <td><strong>2x – 4x 吞吐量暴增</strong></td>
    </tr>
  </tbody>
</table>

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="1962 年">1962 年</time> &mdash; <strong>虚拟内存与分页系统诞生</strong>（<cite>曼彻斯特大学 Ferranti Atlas 计算机</cite>）</dt>
  <dd>Tom Kilburn 团队首次提出了虚拟内存与分页机制，将软件层面的连续逻辑地址空间与底层离散的物理磁芯内存彻底解耦，奠定了现代操作系统的基石。</dd>
  <dt><time datetime="2023 年">2023 年</time> &mdash; <strong>PagedAttention 与 vLLM 开创推理新纪元</strong>（<cite>Woosuk Kwon 等，加州大学伯克利分校 Sky Computing 实验室，SOSP 2023</cite>）</dt>
  <dd>Woosuk Kwon 敏锐洞察到大模型推理的 KV Cache 碎片问题与半个世纪前的操作系统完全同构。他们发明了 PagedAttention 并打造了开源推理引擎 <strong>vLLM</strong>，以摧枯拉朽之势重构了整个 AI 工业界的大模型部署架构。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

我们通过手算，推演两个请求在页表与物理块池之间的动态分配轨迹。

### 系统初始配置
- 块大小：$B = 2\text{ 词元}$
- 物理显存池：共有 4 个可用空闲块 $\{P_0, P_1, P_2, P_3\}$
- 每个块内部可装载 2 个词元的 Key 和 Value 向量。

---

### 第 1 步：请求 1 带着 3 个提示词到达
请求 1 输入提示词：`"the dog barked"`（$T = 3$）。
- 需要块数：$\lceil 3 / 2 \rceil = 2\text{ 个块}$。
- 显存池分配空闲物理块 $P_1$ 与 $P_3$。

初始化请求 1 的页表：
$$
\mathcal{T}_1 = [P_1, \; P_3]
$$

物理块装载状态：
- **$P_1$（对应逻辑块 0）**：装载词元 0（`"the"`）与词元 1（`"dog"`）。**完全填满**（2/2 槽位）。
- **$P_3$（对应逻辑块 1）**：装载词元 2（`"barked"`）。**部分填满**（1/2 槽位，尚有 1 槽位空闲）。

$$\text{内部碎片率} = \frac{1\text{ 空闲槽位}}{4\text{ 已分配槽位}} = 25\%$$

---

### 第 2 步：请求 1 生成词元 3（`"loudly"`）
模型自回归产生词元 3。
- 时序索引：$t = 3$。
- 逻辑块索引：$\lfloor 3 / 2 \rfloor = 1$。
- 块内偏移量：$3 \bmod 2 = 1$。

逻辑块 1 已经映射在 $P_3$ 上，且槽位 1 恰好空闲！
CUDA 算子直接将词元 3 写入 $P_3$ 的偏移量 1 处，**无需向系统申请任何新显存块**。
- 此时 $P_3$ 达到完全满载（2/2 槽位）。
$$\text{内部碎片率} = 0\%$$

---

### 第 3 步：请求 1 继续生成词元 4（`"."`）
模型产生词元 4。
- 时序索引：$t = 4$。
- 逻辑块索引：$\lfloor 4 / 2 \rfloor = 2$。

逻辑块 2 尚未分配！系统从物理池取出下一个空闲物理块 $P_0$，追加写入页表：
$$
\mathcal{T}_1 = [P_1, \; P_3, \; P_0]
$$
词元 4 写入 $P_0$ 的偏移量 0 处。
虽然物理显存分散在 $P_1 \to P_3 \to P_0$ 三个互不相连的区域，但模型完全无感地跨越它们完成了注意力计算！

---

## 步骤 6：核心精髓总结

<fieldset>
<legend><strong>核心教学要点</strong></legend>
<p><strong>PagedAttention</strong> 借用了计算机体系结构中最伟大的发明之一——<strong>操作系统虚拟分页（Virtual Memory Paging）</strong>，彻底化解了大模型自回归解码的显存危机。</p>
<p>通过页表将时序逻辑块映射至离散的物理 DRAM 页块中，PagedAttention 将大模型部署的显存碎片率从 80% 压降至 1% 以下，在相同硬件上直接实现了 2 至 4 倍的吞吐量爆发。</p>
</fieldset>
