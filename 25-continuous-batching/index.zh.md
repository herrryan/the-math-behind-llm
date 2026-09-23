# 第 25 章：细胞级调度（连续批处理与分块预填充数学）

> [!INTUITION] 步骤 1：3 岁小孩直觉
> 想象游乐园里有一辆只有 4 个座位的过山车。
> 
> 四位游客坐了进去。1 号游客只想坐 1 分钟体验一下；2 号游客想坐 2 分钟；而 4 号游客是个极限狂人，他买了一张要连续坐 50 分钟马拉松的超级套票！
> 
> 在传统的 **静态批处理（Static Batching）** 中：
> 过山车轰鸣出发。1 分钟后，1 号游客已经头晕目眩想要下车。但安全压杠被死死锁死！1 号游客只能被迫系着安全带，在座位上干坐整整 49 分钟无所事事。与此同时，护栏外有 100 个满心期待的小朋友在烈日下排成长队，但检票员坚决不准任何人上车，理由是：“必须等 4 号游客坐满 50 分钟，整辆车才能停下开门！”一整天下来，大部分座位上坐的都是发呆的空人，90% 的运力被白白浪费。
> 
> 后来，一位天才工程师发明了 **连续批处理（Continuous Batching）**：
> 每一圈过山车经过站台时，只轻巧减速 1 秒钟。
> - 谁坐够了，立刻解开安全带走下站台；
> - 排在队首的小朋友顺势坐进空出的座位，咔哒一声扣紧卡扣；
> - 过山车瞬间加速重新冲上轨道——**整辆车的 4 个座位在每一圈都坐得满满当当！**
> 
> 如果此时来了一头带了 10 个大箱子的巨型大象（超长提示词）怎么办？
> 检票员采用 **分块预填充（Chunked Prefill）**：每一圈只让大象搬 2 个箱子上去。过山车永远不中断运行，后面的游客也再也不会被堵死在站外！

---

## 步骤 2：承前启后的关键过渡

我们如何将游乐园过山车站台的“随到随上下车”机制，抽象为在 GPU 上以单个词元生成步（Iteration）为粒度的状态转移数学方程，又如何让高计算密度的 Prefill 与高访存受限的 Decode 在同一个前向传播中搭上“顺风车”？

在常规的深度学习离线训练中，批处理完全是静态且规整的：张量拥有固定的矩形尺寸 $[B, T]$，所有样本齐步走，经历相同次数的前向与反向传播。

但在现实的在线大模型推理服务中，进入系统的请求展现出三重极端的动态不确定性：
1. **到达时间不确定**：用户请求随时到达，遵循典型的泊松分布。
2. **输入长度不确定**：提示词长度天差地别，有的只有 5 个词，有的长达 32,000 个词。
3. **输出长度不确定**：生成过程是由模型内部何时吐出特殊的终止符 `<eos>` 动态决定的，事先根本无法预知。

如果推理引擎像训练一样将请求打包为固定批次，短请求早早结束后所留下的空槽位就会产生极其严重的 **显存气泡（Bubble Waste）**：GPU 算力核心只能被迫空转，对全零的空白 Padding 执行无意义的乘加运算。

承前启后的核心过渡问题是：
$$\text{推理调度器如何在每生成一个词元的离散时钟步内动态更新批次成员，又如何通过切分超长 Prompt 让访存受限的解码请求“顺手搭车”，彻底粉碎气泡浪费？}$$

---

## 步骤 3：严谨数学公式与推导

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        静态批处理 VS. 连续批处理调度对比                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 静态批处理（粗粒度请求级锁定）：                                                       │
│   槽位 0: [ 请求 A (生成 3 词) ] [ 无效空白填充 PADDING PADDING PADDING ] &lt;── 60% 气泡│
│   槽位 1: [ 请求 B (生成 5 词) ] [ 无效空白填充 PADDING ]                              │
│   槽位 2: [ 请求 C (生成 8 词 - 最长请求) ]                                            │
│   ──► 槽位 0 和 1 只能全程陪跑空转！新请求全被堵在等待队列外。                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 连续批处理（细粒度词元迭代级更新）：                                                   │
│   槽位 0: [ 请求 A (生成 3 词) ] ──► 立即驱逐 ──► [ 请求 D 立即插入并开跑！ ]          │
│   槽位 1: [ 请求 B (生成 5 词) ] ──► 立即驱逐 ──► [ 请求 E 立即插入并开跑！ ]          │
│   槽位 2: [ 请求 C 持续正常生成... ]                                                   │
│   ──► 全程零填充气泡！GPU 每一个前向迭代的所有槽位始终保持 100% 满载饱和。             │
└────────────────────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 25.1:</strong> 静态批处理被最长请求完全绑架，产生巨大的显存气泡；连续批处理在每一个迭代步完成动态换乘，实现算力满载。</figcaption>
</figure>

### 1. 静态批处理的气泡浪费率方程

设一个静态批次包含 $B$ 个请求，各自的真实生成词元长度为 $T_{\text{gen}}^{(1)}, T_{\text{gen}}^{(2)}, \dots, T_{\text{gen}}^{(B)}$。
该批次整体被最长请求所绑架，总共必须执行的迭代步数为：
$$
T_{\text{max}} = \max_{i=1}^B T_{\text{gen}}^{(i)}
$$

系统在此期间分配给 GPU 计算的槽位总量为 $B \times T_{\text{max}}$，而实际产出有用词元的槽位仅为 $\sum_{i=1}^B T_{\text{gen}}^{(i)}$。

<dfn id="def-bubble-zh">气泡浪费率</dfn> $\eta_{\text{bubble}}$ 定义为：

$$
\eta_{\text{bubble}} = 1 - \frac{\sum_{i=1}^B T_{\text{gen}}^{(i)}}{B \times \max_{i=1}^B T_{\text{gen}}^{(i)}}
$$

在互联网真实生产流量中，由于长短文本混合，$\eta_{\text{bubble}}$ 通常高达 **70% 以上**！这意味着昂贵的 GPU 集群有三分之二以上的算力都在徒劳地计算 Padding 掩码！

---

### 2. 连续批处理（Iteration-Level）的状态转移方程

设 $\mathcal{R}_{\text{waiting}}$ 为等待服务的请求先进先出队列，$\mathcal{B}_k = \{r_1, r_2, \dots, r_{B_k}\}$ 为第 $k$ 个迭代步正在执行的活跃批次集合。

在每个离散时钟步 $k \to k + 1$ 发生如下细胞级状态跃迁：
1. **统一前向计算**：GPU 对当前批次 $\mathcal{B}_k$ 统一执行一步前向传播。
2. **终止符检测**：检查各请求吐出的新词元 $y_k^{(i)}$：
   $$\text{Finished}(r_i) = \left(y_k^{(i)} = \langle\text{eos}\rangle\right) \lor \left(\text{len}(r_i) \ge T_{\text{limit}}\right)$$
3. **动态驱逐与注入（Eviction &amp; Injection）**：
   $$\mathcal{B}_{\text{surviving}} = \left\{r_i \in \mathcal{B}_k \mid \neg \text{Finished}(r_i)\right\}$$
   空出的槽位数量为 $N_{\text{free}} = B_{\text{max}} - |\mathcal{B}_{\text{surviving}}|$，调度器立刻从队列队首补足新请求：
   $$
   \mathcal{B}_{k+1} = \mathcal{B}_{\text{surviving}} \cup \operatorname{Dequeue}\left(\mathcal{R}_{\text{waiting}}, N_{\text{free}}\right)
   $$

每个词元生成瞬间都可以自由进出，从而将气泡浪费率数学上压缩至 $\eta_{\text{bubble}} \to 0$。

---

### 3. 分块预填充（Chunked Prefill）与“搭便车”访存数学

连续批处理虽然消除了气泡，却引入了新的“恶霸”：当一个拥有 8,000 词元的超长 Prompt 新请求进入 $\mathcal{B}_{k+1}$ 时，其庞大的 Prefill 运算会强行霸占 GPU 数百毫秒，导致同批次正在流畅解码的存量请求发生严重的卡顿抖动（<dfn id="def-tpot-zh">输出词元间隔时间 TPOT</dfn> 剧烈恶化）。

**分块预填充（Chunked Prefill）**（源自 Sarathi-Serve）通过设定每个迭代步的总词元预算上限 $K_{\text{budget}}$（例如 $K_{\text{budget}} = 512$）来化解这一矛盾。
当某 Prompt 长度超出剩余预算时，将其拆分成按需切片：
$$
C = \min\left(T_{\text{remaining}}, \; K_{\text{budget}} - N_{\text{decode}}\right)
$$

此时，我们在同一前向步中构建一个**混合批次（Mixed Batch）**，包含 $N_{\text{decode}}$ 个解码单词元以及 $C$ 个切片预填充词元：
- **浮点计算量**：
  $$\text{FLOPs} \approx 2 \times (N_{\text{decode}} + C) \times P$$
  （其中 $P$ 为模型参数总量）。
- **显存权重读取量**：
  整层模型权重从 HBM 读入 SRAM **仅仅一次**：
  $$\text{Bytes} \approx 2 \times P \times p$$
- **算术强度（计算访存比）**：
  $$
  I_{\text{mixed}} = \frac{2 \times (N_{\text{decode}} + C) \times P}{2 \times P \times p} = \frac{N_{\text{decode}} + C}{p} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]
  $$

若在 16 位半精度下（$p = 2$ 字节），令 $N_{\text{decode}} = 32, C = 480$：
$$
I_{\text{mixed}} = \frac{32 + 480}{2} = 256 \text{ FLOPs/Byte} \approx I^*
$$

<mark>混合批次的“搭便车”奇迹：</mark> 原本深陷显存带宽受限区的 32 个逐字解码词元，**顺便蹭上了** 为 480 个预填充词元载入 SRAM 的那一整套模型权重！解码阶段以近乎零边际访存成本完成了推理，彻底消除了交互卡顿与延迟抖动。

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 25.1:</strong> 三代大模型服务调度范式全方位对比。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">调度范式</th>
      <th align="center">调度粒度</th>
      <th align="center">显存气泡浪费</th>
      <th align="center">解码时延抖动（TPOT）</th>
      <th align="right">服务吞吐倍率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>静态批处理（传统）</strong></td>
      <td align="center">请求级（粗粒度）</td>
      <td align="center">50% – 80%</td>
      <td align="center">严重</td>
      <td align="right">1.0x（基准）</td>
    </tr>
    <tr>
      <td><strong>连续批处理（Orca / vLLM）</strong></td>
      <td align="center">迭代步级（细粒度）</td>
      <td align="center"><strong>&lt; 5%</strong></td>
      <td align="center">中等（新入 Prefill 偶发阻塞）</td>
      <td align="right"><strong>2.5x – 3.5x</strong></td>
    </tr>
    <tr>
      <td><strong>分块预填充（Sarathi）</strong></td>
      <td align="center">迭代步 + 词元切片</td>
      <td align="center"><strong>&lt; 1%</strong></td>
      <td align="center"><strong>近乎为零（平稳 TPOT）</strong></td>
      <td align="right"><strong>4.0x+</strong></td>
    </tr>
  </tbody>
</table>

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2022 年">2022 年</time> &mdash; <strong>Orca 提出迭代级调度</strong>（<cite>Gyeong-In Yu 等，首尔大学与微软亚洲研究院，OSDI 2022</cite>）</dt>
  <dd>打破了深度学习自发明以来以完整请求为批处理单元的铁律，首次提出了连续批处理架构，将大模型在线生成吞吐量暴拉了 36 倍。</dd>
  <dt><time datetime="2024 年">2024 年</time> &mdash; <strong>Sarathi-Serve 提出分块预填充</strong>（<cite>Amey Agrawal 等，微软研究院，OSDI 2024</cite>）</dt>
  <dd>敏锐发现连续批处理中巨型 Prefill 造成的解码卡顿，创立了 Chunked-Prefill 与混合计算模式，使大模型推理在维持极致高吞吐的同时，保证了毫秒级的丝滑交互时延。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

我们通过手算，对比静态批处理与连续批处理在容量为 $B_{\text{max}} = 2$ 槽位的服务系统中的执行轨迹。

### 请求属性
- **请求 A**：$t=0$ 到达，需生成 1 个词元。
- **请求 B**：$t=0$ 到达，需生成 3 个词元。
- **请求 C**：$t=1$ 到达，需生成 2 个词元。

---

### 方法 1：静态批处理执行轨迹
在 $t=0$，系统打包请求 A 和 B 组成静态批次：
- **迭代 0**：A 产生第 1 词（已完成！）；B 产生第 1 词。
- **迭代 1**：A 已经完成，但被锁在槽位里填充 Padding；B 产生第 2 词。
- **迭代 2**：A 继续填充 Padding；B 产生第 3 词（完成！）。

静态批次在 $t=3$ 彻底结束。
消耗总槽位 = $2 \times 3 = 6$ 个。
产出有效词元 = $1 + 3 = 4$ 个。
$$\text{气泡浪费率} = 1 - \frac{4}{6} = 33.3\%$$

请求 C 必须在队列中苦苦等待到 $t=3$ 才能开始！

---

### 方法 2：连续批处理执行轨迹
- **迭代 0（$t=0$）**：
  - 槽位状态：[请求 A, 请求 B]。
  - A 吐出第 1 词 $\to$ 立即标记完成！
  - B 吐出第 1 词。
- **迭代 1（$t=1$）**：
  - **A 被立刻驱逐出槽位**。
  - 刚刚在 $t=1$ 到达的请求 C **无缝直接插进槽位 0**！
  - 槽位状态：[请求 C, 请求 B]。
  - C 吐出第 1 词；B 吐出第 2 词。
- **迭代 2（$t=2$）**：
  - 槽位状态：[请求 C, 请求 B]。
  - C 吐出第 2 词 $\to$ 完成！
  - B 吐出第 3 词 $\to$ 完成！

所有 3 个请求在 $t=3$ 瞬间全部圆满完成！
消耗总槽位 = 6 个。
产出有效词元 = $1 (\text{A}) + 3 (\text{B}) + 2 (\text{C}) = 6\text{ 个}$。
$$\text{气泡浪费率} = 1 - \frac{6}{6} = 0\%$$
气泡浪费彻底归零，请求 C 的排队等待时间直接被压缩掉了整整 2 个周期！

---

## 步骤 6：核心精髓总结

<fieldset>
<legend><strong>核心教学要点</strong></legend>
<p><strong>连续批处理（Continuous Batching）</strong> 彻底废除了粗放的请求级封装，将大模型服务重构为以单个词元步为最小转移单位的细粒度流式状态机，使请求完成即可下车、新请求立刻插槽。</p>
<p>配合 <strong>分块预填充（Chunked Prefill）</strong>，它在数学上将算力受限的高密度 Prompt 与带宽受限的稀疏 Decode 巧妙编织在同一个混合批次中，让解码运算无偿搭上权重访存的顺风车，达成了零气泡浪费与低时延抖动的完美统一。</p>
</fieldset>
