# 动手实战 Lab 05：流式 KV 与连续批处理引擎（200 行纯 Python 实现 vLLM 核心内核）

<fieldset id="evolution">
<legend><strong>Python 极简大脑演进链 &bull; 工业级推理前沿（第 5 阶段，共 6 阶段）</strong></legend>
<p>在 Lab 04 中，我们完成了带有基础 KV Cache 与核采样（Top-p）的单请求交互式大模型。然而该引擎每次只能服务一位用户，且依赖连续线性内存数组。在真实的生产级高并发场景中，数以千计的用户会在不同时刻发送长度迥异的请求。</p>
<p>在第五个动手实战中，我们用约 200 行标准库纯 Python 代码从零构建<strong>流式 KV 与连续批处理引擎</strong>。绝无外部依赖：不使用 PyTorch，不使用 vLLM，不使用 NumPy。我们将亲手实现现代高并发大模型推理服务的两项基石技术：<strong>PagedAttention 虚拟显存分页管理</strong>（第 21 与 22 章）与<strong>连续批处理（Continuous Dynamic Batching）</strong>（第 25 章）。</p>
<pre>
[Python 极简大模型演进全景路线]
[阶段 1]  80 行纯 Python: Bengio 2003 神经网络前馈语言模型 (词嵌入, 隐藏层投影, 手写反向传播)
       │
       ▼ (失忆缺陷: 仅 1 词上下文窗口)
[阶段 2] 140 行纯 Python: 注意力大脑 (自注意力 Q, K, V 投影与因果下三角掩码)
       │
       ▼ (数值不稳: 深层堆叠梯度弥散与爆炸)
[阶段 3] 220 行纯 Python: 现代 Transformer 块 (Pre-RMSNorm, 残差直连高速公路, SwiGLU 门控)
       │
       ▼ (采样死板: 贪婪循环死锁与 O(T^2) 冗余重算)
[阶段 4] 300 行纯 Python: 交互式端到端大模型 (KV Cache 增量缓存与核采样套件)
       │
       ▼ (多租户吞吐缺陷: 静态批处理产生 60%+ 气泡浪费; 连续内存导致显存严重碎片化)
[阶段 5 (当前)] 200 行纯 Python: 流式 KV 与连续批处理引擎 (PagedAttention 分页管理与单步迭代调度)
       │
       ▼ (物理带宽瓶颈: 单字解码仍受显存带宽墙 O(T) 限制)
[阶段 6 (下一阶段)] 220 行纯 Python: 推测解码与 INT4 权重量化引擎
</pre>
</fieldset>

---

## 第 1 步：3 岁小孩直觉（行李寄存柜与永不停歇的旋转转盘）

想象你正在管理机场的行李提取大厅，旅客在全天各个时刻随机抵达：

<figure>
<pre>
[传统静态批处理：死板的旅游大巴]
旅客 1 (极快：只有 1 个背包) ───► [第 1 分钟完成] ───► 必须在座位傻坐干等 59 分钟！
旅客 2 (极慢：有 60 件大行李) ───► 正在搬运...    ───► 正在搬运... ───► 第 60 分钟才完成
旅客 3 (中等：有 10 件行李)   ───► [第 10 分钟完成] ──► 必须在座位傻坐干等 50 分钟！
                                └───────────────────────────────────────────────┘
                                       算力气泡浪费：超过 60% 的座位处于空转闲置！

[连续批处理：永不停歇的旋转转盘]
迭代 01: [请求 A: 步 1] [请求 B: 步 1] [请求 C: 步 1]
迭代 02: [请求 A: 完成] [请求 B: 步 2] [请求 C: 步 2]
               │
               ▼ (瞬间释放槽位 &amp; 动态接入新请求)
迭代 03: [请求 D: 预填] [请求 B: 步 3] [请求 C: 完成]
                                               │
                                               ▼
迭代 04: [请求 D: 步 1] [请求 B: 步 4] [请求 E: 预填]  (彻底消灭空等气泡！)
</pre>
<figcaption><strong>图 25b.1：</strong> 连续批处理将调度粒度从“粗暴的整批等待”下沉到“精细的单步迭代”，彻底消灭了 GPU 计算槽位空转的气泡浪费。</figcaption>
</figure>

1. **霸道的整排占座（连续内存分配的困局）**：
   在早期的推理系统中，当用户开启一段对话时，系统会询问：*“你这通对话最多可能说多长？”* 如果用户回答 4,096 个词，系统就会在显存里硬生生划出一整排连续的 4,096 个专属空位。如果用户最终只问了句 *“1+1 等于几？”* 并得到了 *“2”*，那么 99.9% 预留的显存位置就全程空置，但其他用户完全无权使用！

2. **自动储物柜（PagedAttention 的分页哲学）**：
   系统不再霸道地为每个人包下一整间大库房，而是采购了一批标准规格的 4 格储物柜（Page）。随着用户的回答逐字吐出，只有当当前这组 4 格柜子彻底装满时，系统才会从空闲池中申请下一个 4 格储物柜。这些储物柜可以散落在显存的任意角落（非连续物理内存），调度台只需持有一张小巧的索引清单（**Block Table 块表**）即可完成寻址。

3. **永不停歇的旋转转盘（连续批处理）**：
   系统不再要求大巴车上的所有旅客必须同时等最慢的人。转盘每转动一圈（执行一次前向推理），旅客 A 拿到自己的单张小卡片（生成 `<eos>`）就立刻离场；其所占用的储物柜瞬间清空退回空闲池，大门外刚到的旅客 D 立即登台补位！

---

## 第 2 步：承前启后的关键过渡

在标准的 PyTorch 矩阵运算中，多头注意力机制默认接收规整的连续张量：

$$
\mathbf{K} \in \mathbb{R}^{B \times L \times d_k}, \quad \mathbf{V} \in \mathbb{R}^{B \times L \times d_v}
$$

然而在 PagedAttention 下，历史词元的 KV 向量被切分成大小为 $B$（如 4）的分页，物理上离散地散落在显存的不同区块中。我们该如何通过块表完成非连续内存的注意力寻址？调度器又是如何在单次循环中同时统筹预填（Prefill）与解码（Decode）的？

---

## 第 3 步：严谨数学公式与推导

### 1. PagedAttention 物理显存地址映射

设当前序列长度为 $L$ 个词元，固定物理块容量为 $B$（例如 $B = 4$）。逻辑词元索引 $t \in [0, L-1]$ 映射到 GPU 物理显存的具体公式为：

$$
\text{逻辑块索引 } i = \lfloor t / B \rfloor, \quad \text{块内槽位偏移 } o = t \pmod B
$$

$$
\text{物理块号 } p = \text{BlockTable}[i]
$$

$$
\mathbf{k}_t = \text{PhysicalMemory}[p][\text{Key}][o], \quad \mathbf{v}_t = \text{PhysicalMemory}[p][\text{Value}][o]
$$

### 2. 离散块上的 Scaled Dot-Product 注意力算子

对于当前最新生成词元所产生的查询向量 $\mathbf{q} \in \mathbb{R}^{1 \times d_k}$：

$$
S_t = \frac{\mathbf{q} \cdot \mathbf{k}_t}{\sqrt{d_k}} = \frac{\mathbf{q} \cdot \text{PhysicalMemory}[\text{BlockTable}[\lfloor t/B \rfloor]][\text{Key}][t \pmod B]}{\sqrt{d_k}}
$$

$$
\alpha_t = \frac{\exp(S_t - \max_j S_j)}{\sum_{\tau=0}^{L-1} \exp(S_\tau - \max_j S_j)}
$$

$$
\mathbf{o} = \sum_{t=0}^{L-1} \alpha_t \, \mathbf{v}_t
$$

### 3. 静态批处理气泡浪费率量化公式

若一个批次内包含 $N$ 个并发请求，在静态批处理下该批次必须持续运行直至最长请求 $L_{\max} = \max_i L_i$ 结束。理论总计算槽位为 $N \cdot L_{\max}$，有效生成的总词元数为 $\sum_{i=1}^N L_i$。气泡浪费率 $\eta_{\text{bubble}}$ 为：

$$
\eta_{\text{bubble}} = 1 - \frac{\sum_{i=1}^N L_i}{N \cdot \max_{1 \le j \le N} L_j}
$$

在连续批处理中，请求一旦生成 $\langle \text{eos} \rangle$ 即刻下线并腾出显存与计算位，使 $\eta_{\text{bubble}} \to 0$。

---

## 第 4 步：历史渊源与技术演进

2022 年，加州大学伯克利分校的 Yu 等人提出 **Orca**，首次证明将调度粒度下沉至“单步迭代级（Iteration-level）”可使大模型服务吞吐提升多达 36 倍。但 Orca 仍采用预分配连续显存策略，依然受到显存碎片的制约。

2023 年，Kwon 等人发表 **PagedAttention** 并开源了著名的 **vLLM** 框架。其灵感来源于 1962 年英国 Atlas 计算机操作系统中诞生的虚拟内存分页机制。vLLM 彻底将逻辑序列与物理显存解耦，将大模型显存浪费率从传统方案的 $60\% \sim 80\%$ 骤降至 $4\%$ 以下，使得单张显卡所能承载的并发用户数翻了数倍。

---

## 第 5 步：手算极简数值示例与代码实战

我们用 4 个物理块（每块容量 $B = 2$ 个词元）追踪两道请求的内存演进：

<figure>
<pre>
[初始物理内存状态 &bull; 4 个物理块，每块容量 2]
块 0: [ 空闲 ] [ 空闲 ]
块 1: [ 空闲 ] [ 空闲 ]
块 2: [ 空闲 ] [ 空闲 ]
块 3: [ 空闲 ] [ 空闲 ]

请求 1 抵达：提示词 = ["cat", "sat"] (需要 2 个槽位)
  - 分配物理块 0 -> 块表 BlockTable[Req 1] = [0]
  - 物理块 0 内容: ["cat", "sat"]

请求 2 抵达：提示词 = ["the", "big", "dog"] (需要 3 个槽位)
  - 分配物理块 1 (装填: "the", "big")
  - 分配物理块 2 (装填: "dog", 暂空)
  - 块表 BlockTable[Req 2] = [1, 2]

迭代 1 解码执行：
  - 请求 1 解码生成: "on" -> 块 0 已满！动态申请块 3 -> BlockTable[Req 1] = [0, 3]
  - 请求 2 解码生成: "barks" -> 填入块 2 剩余空槽！无需新增物理块
</pre>
<figcaption><strong>图 25b.2：</strong> 物理块仅在容量不足时按需分配，绝无大段连续内存预占浪费。</figcaption>
</figure>

### 实战文件结构

```
25b-lab-streaming-engine/
├── streaming_engine.py          # 完整参考推理服务内核 (~200 行)
├── streaming_engine_exercise.py # 引导式动手练习脚本 (含 TODO 与完整单元测试)
├── index.md                     # 英文版实战详解
└── index.zh.md                  # 中文版实战详解
```

### 运行参考引擎

```bash
python3 25b-lab-streaming-engine/streaming_engine.py
```

终端执行日志：
```
=====================================================================
Lab 05: Streaming KV & Continuous Batching Engine Simulation
=====================================================================

--- [Iteration 01] ---
  [ADMIT & PREFILL] Req 'Req-A': prompt ['the', 'cat'] -> blocks [0]
  [ADMIT & PREFILL] Req 'Req-B': prompt ['dog', 'sat', 'on'] -> blocks [1]
  [ADMIT & PREFILL] Req 'Req-C': prompt ['sun'] -> blocks [2]
  [DECODE STEP] Req 'Req-A' -> 'sat' (Blocks: [0])
  [DECODE STEP] Req 'Req-B' -> 'dog' (Blocks: [1])
  [DECODE STEP] Req 'Req-C' -> 'dog' (Blocks: [2])
  [MEMORY STATUS] Active Blocks: 3/12 (Free: 9)

--- [Iteration 02] ---
  [FINISH & EVICT] Req 'Req-A' produced '<eos>'. Blocks freed! Total output: ['sat']
  [DECODE STEP] Req 'Req-B' -> 'sun' (Blocks: [1, 3])
  [DECODE STEP] Req 'Req-C' -> 'sun' (Blocks: [2])
  [MEMORY STATUS] Active Blocks: 3/12 (Free: 9)

--- [Iteration 03] ---
  [ADMIT & PREFILL] Req 'Req-D': prompt ['the', 'cat', 'slept'] -> blocks [4]
  [DECODE STEP] Req 'Req-B' -> 'dog' (Blocks: [1, 3])
  [FINISH & EVICT] Req 'Req-C' produced '<eos>'. Blocks freed! Total output: ['dog', 'sun']
  [DECODE STEP] Req 'Req-D' -> 'the' (Blocks: [4])
  [MEMORY STATUS] Active Blocks: 3/12 (Free: 9)

=====================================================================
Final Performance Audit & Bubble Waste Analysis
=====================================================================
Total Requests Completed: 4
  * Req-A: Prompt=2 tok, Gen=1 tok, Latency=1 iters
  * Req-C: Prompt=1 tok, Gen=2 tok, Latency=2 iters
  * Req-D: Prompt=3 tok, Gen=1 tok, Latency=1 iters
  * Req-B: Prompt=3 tok, Gen=5 tok, Latency=5 iters

Static Batching Theoretical Slot Usage:     20 slot-steps
Actual Useful Compute Tokens:               9 tokens
Static Batching Bubble Waste:               55.0%
Continuous Batching Bubble Waste:           0.0% (Slots instantly recycled!)
=====================================================================
```

### 运行引导式单元测试

```bash
python3 25b-lab-streaming-engine/streaming_engine_exercise.py
```

---

## 第 6 步：核心精髓总结

大规模服务大模型本质上是**操作系统级内存管理与硬件调度协同的工程艺术**：通过将连续张量粉碎为离散物理块（**PagedAttention**），并将调度下沉到每个前向传播的微小切片（**连续批处理**），我们从物理底层根治了显存碎片，夺回了被静态填充气泡吞噬的高达 60% 的算力。
