# 第 16 章：迷雾中下山（梯度下降与反向传播）


## 第 1 步：3 岁孩子也能懂的直觉（迷雾下山与接力传话筒） {: #step-1 }

!!! note "3岁小孩的直觉: 蒙眼登山客与下山传话向导队"
    想象黄昏时分，你独自站在一座陡峭、云雾缭绕的高山顶上。

    你的目标是安全走到山谷最底部的温暖小木屋（在数学上，那里的误差损失是零）。但山上的浓雾实在太重了，能见度只有眼前两厘米，你连脚趾头都看不清：

    1. **用登山靴感知脚底的地面倾斜**：
       - 尽管你完全看不见远方的小木屋在哪个方位；
       - 但是，只要你静下心来，凭借登山靴鞋底与岩石地面的触感，你立刻就能感知到脚底下的土地正在向哪个方向倾斜！
       - 如果地面明显向你的左下方倾斜，你就小心翼翼地向左下方迈出一小步。
       - 这一小步让你的海拔高度切实降低了一点点。只要连续踩准倾斜方向迈出几百步，你总能顺着山体斜坡，稳稳当当降落到谷底！
       - 这个在脚底下感知到的地面倾斜角度与方向，正是数学上的**梯度（<dfn id="def-gradient-zh">Gradient, 记作 $\nabla \mathcal{L}$</dfn>）**！

    2. **山间向导的接力传话筒**：
       - 现在，整座下山的险峻山路上，从谷底的小木屋一直排到山顶，站着一整队尽职尽责的传话向导。
       - 最底部的侦察员看到了登山客刚才偏离了多少距离，在纸条上写下评语：<samp>“往左偏了整整 2 米！”</samp>，然后立刻转过身，把纸条逆向递给身后上一级的向导。
       - 上一级的向导看完纸条，飞快算出自己负责的路段该怎么微调路标，并在纸条背面补上最新修正意见，继续逆流向上传递。
       - 眨眼之间，整座山上的每一个向导，都清清楚楚地知道了该把自己的路标向左挪几毫米，以便下一次把登山客引导得更靠近小木屋！

    在大语言模型中，这种顺着倾斜下山的旅程就叫做**梯度下降（Gradient Descent）**，而这支自下而上逆向传递错误修正纸条的向导队，就是震撼整个现代人工智能的**反向传播算法（<dfn id="def-backprop-zh">Backpropagation</dfn>）**。

    哪怕像 LLaMA 这样的大模型体内装有 700 亿个可调节旋钮（参数权重），反向传播也能在一次极其迅速的逆向计算流中，把全部 70,000,000,000 个旋钮的下山倾斜度瞬间全量算清！

<figure>
<pre>
前向传播（做出猜测与预测）：
  [ 输入词元 ] ──► [ 第 1 层 ] ──► [ 第 2 层 ] ──► [ 预测输出 ]
                                                       │
                                                       ▼
                                                 [ 误差损失 ]
                                                       │
反向传播（逆向传递误差纸条）：                           ▼
  [ 微调 W_1 ] ◄── [ 微调 W_2 ] ◄── [ "你刚才高估了 +2.0！" ]
  （每一层接收下游传来的误差纸条，算出本层权重的梯度，
   然后把更新后的梯度信息继续逆流传递给上一层）
</pre>
<figcaption><strong>图 16.1：</strong> 前向传播逐层向前合成预测；反向传播逐层逆向回传误差梯度。</figcaption>
</figure>

---

## 第 2 步：承前启后的关键过渡 {: #step-2 }

!!! question "计算连接问题: 计算机如何在一瞬间算准 700 亿个参数的偏导数？"
    在第 15 章中，我们推导出了网络最顶端的奇迹对数几率梯度：

    $$
    \frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}
    $$

    但这仅仅告诉了我们最顶层 Logits 输出应该如何微调。

    在这些 Logits 的深处，层层叠叠掩埋着数十层 Transformer 结构：反向词嵌入矩阵 $\mathbf{W}_U$、前馈升降维矩阵 $\mathbf{W}_{\text{down}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{gate}}$、注意力头投影矩阵 $\mathbf{W}_O, \mathbf{W}_V, \mathbf{W}_K, \mathbf{W}_Q$、RMSNorm 增益参数 $\boldsymbol{\gamma}$，以及最底层的词嵌入查找表 $\mathbf{E}$。

    如果我们试图用朴素的数值微积分方法，对全部 $P = 70,000,000,000$ 个参数逐个求偏导（比如先给其中一个权重加上微小的 $\epsilon = 0.0001$，重新运行一次全网前向推理，观察损失变动情况；然后再给第二个权重加扰动……），那么每计算单个训练词元，就需要执行 **700 亿次全网前向传播**！算完一句话可能要花上几百年。

    “多元微积分链式法则究竟运用了怎样的拓扑数学原理，把这原本需要 700 亿次漫长迭代的绝境，奇迹般地压缩为单次反向回传即可全量解出的高效算法？”

---

## 第 3 步：严谨数学推导与公式 {: #step-3 }

### 1. 计算图与多元链式法则

在数学上，神经网络被形式化为一个有向无环图（\lt dfn id="def-comp-graph-zh">计算图，Computational Graph</dfn>）。
假设变量 $\mathbf{x}$ 经过基础变换生成中间变量 $\mathbf{y} = f(\mathbf{x})$，随后经由后续网络生成最终标量损失 $\mathcal{L} = g(\mathbf{y})$。

根据多元微积分的**链式法则（Chain Rule）**：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
$$

其中：
- $\frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^{1 \times d_y}$ 是下游节点逆流回传给当前节点的误差梯度（即向导手中的“误差纸条”）。
- $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{d_y \times d_x}$ 是局部操作 $f$ 自身的**雅可比矩阵（Jacobian Matrix）**。
- $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times d_x}$ 是当前节点完成链式乘法后，准备继续向上游祖先节点回传的更新梯度。

---

### 2. 核心积木：线性矩阵乘法层的反向梯度

大语言模型超过 95% 的浮点运算算力（FLOPs）都消耗在线性矩阵乘法中：$\mathbf{y} = \mathbf{x}\mathbf{W}$。

设：
- $\mathbf{x} \in \mathbb{R}^{B \times d_{\text{in}}}$ 为输入激活矩阵（批大小为 $B$）。
- $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ 为当前层的可学习权重矩阵。
- $\mathbf{y} = \mathbf{x}\mathbf{W} \in \mathbb{R}^{B \times d_{\text{out}}}$ 为前向输出激活矩阵。
- $\mathbf{G}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^{B \times d_{\text{out}}}$ 为下游逆流回传的输出梯度。

在反向传播经过该线性节点时，只需执行两次独立的矩阵乘法：

#### A. 对权重矩阵 $\mathbf{W}$ 的梯度（用于参数更新）：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{x}^\top \mathbf{G}_y \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}
$$

#### B. 对输入激活 $\mathbf{x}$ 的梯度（继续逆向回传给上一层）：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{G}_y \mathbf{W}^\top \in \mathbb{R}^{B \times d_{\text{in}}}
$$

\lt figure>
\lt pre>
线性矩阵层的反向传播数据流向图：

前向传播：  x  [ B x d_in ] ──► ( * W ) ──► y  [ B x d_out ]
                                    │
                                    ▼
反向传播：  dL/dx = G_y * W^T  ◄── ( G_y ) ◄── dL/dy = G_y
                                    │
                                    ▼
                           dL/dW = x^T * G_y
</pre>
\lt figcaption>\lt strong>图 16.2：</strong> 线性层将回传的梯度 $\mathbf{G}_y$ 一分为二：与 $\mathbf{x}^\top$ 相乘获得权重的更新梯度，与 $\mathbf{W}^\top$ 相乘获得回传给上一层的激活梯度。</figcaption>
</figure>

注意这里的维度对称美学：
- 更新权重 $\mathbf{W}$（$d_{\text{in}} \times d_{\text{out}}$）时，转置输入 $\mathbf{x}^\top$（$d_{\text{in}} \times B$）与梯度 $\mathbf{G}_y$（$B \times d_{\text{out}}$）点积。
- 回传输入 $\mathbf{x}$（$B \times d_{\text{in}}$）时，梯度 $\mathbf{G}_y$（$B \times d_{\text{out}}$）与转置权重 $\mathbf{W}^\top$（$d_{\text{out}} \times d_{\text{in}}$）点积。

---

### 3. 梯度下降参数更新法则

一旦算出了权重梯度矩阵 $\nabla_{\mathbf{W}} \mathcal{L}$，模型旋钮便沿着负梯度方向更新步进：

$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla_{\mathbf{W}} \mathcal{L}_t
$$

其中 $\eta > 0$ 是**学习率（Learning Rate）**，决定了我们在迷雾山体上每一步下迈的尺度。

---

### 4. 为什么反向模式比前向模式快 700 亿倍？

为什么微积分求导必须“从后往前”推，而不能“从前往后”推？

考察一个拥有 $P$ 个参数、最终输出单个标量损失 $\mathcal{L} \in \mathbb{R}$ 的大模型：
- **前向自动微分（切空间前推）**：从输入参数向前传播导数。因为每个输入参数都需要单独执行一次完整遍历，求解全部 $P$ 个参数的导数需要 **$\mathcal{O}(P)$ 次计算遍数**（整整 700 亿遍！）。
- **反向自动微分（伴随空间回传，即反向传播）**：从单一标量损失 $\mathcal{L}$ 出发，逆向回传梯度。因为标量输出**只有唯一 1 个数值**，反向扫过一遍计算图，就能顺带获得全部 $P$ 个参数的偏导数，复杂度仅为 **$\mathcal{O}(1)$ 次反向遍数**！

$$
\frac{\text{反向模式计算代价}}{\text{前向模式计算代价}} = \frac{1}{P} \approx \frac{1}{70,000,000,000}
$$

没有反向模式自动微分，现代万亿参数大语言模型的预训练在物理世界上根本不可能实现。

---

## 第 4 步：历史源流与思考演进（Linnainmaa、Rumelhart 与 Hinton） {: #step-4 }

\lt dl>
  \lt dt>\lt time datetime="1970">1970</time> &mdash; \lt strong>塞波·林纳因马（Seppo Linnainmaa）</strong></dt>
  \lt dd>在赫尔辛基大学的硕士论文中首次提出了反向模式自动微分的通用算法，证明了嵌套代数函数的复合导数计算时间，与原始函数的前向计算时间严格成同阶正比。</dd>

  \lt dt>\lt time datetime="1986">1986</time> &mdash; \lt strong>戴维·鲁梅尔哈特、杰弗里·辛顿 与 罗纳德·威廉姆斯</strong></dt>
  \lt dd>在顶级科学期刊 \lt em>Nature</em> 上发表了划时代论文 \lt cite>《Learning representations by back-propagating errors》</cite>。他们首次证明反向传播能够训练多层神经网络自发形成复杂的内部特征表征，一举击碎了明斯基当年针对感知机 XOR 问题的悲观论断，终结了第一次 AI 寒冬。</dd>

  \lt dt>\lt time datetime="2017">2017</time> &mdash; \lt strong>PyTorch 与动态自动求导机制</strong></dt>
  \lt dd>由 Adam Paszke 等人开创的基于磁带记录（Tape-based Autograd）的动态计算图机制，让研究人员无需手写任何一行求导代码，即可对前沿 Transformer 的注意力掩码、因果循环和多分支动态控制流执行无缝自动微分。</dd>
</dl>

---

## 第 5 步：手把手超简单数字积木（两层网络的完整反向推导手算） {: #step-5 }

为了让你彻底看清反向传播在底层是如何逐层传递纸条并更新权重的，我们用最简单的小数字，纯手算一套完整的两层标量前向与反向传播。

### 1. 微型架构与参数设定
- 输入标量：$x = 2.0$
- 第 1 层权重：$w_1 = 3.0$
- 隐藏层激活值：$h = x \cdot w_1$
- 第 2 层权重：$w_2 = 2.0$
- 模型最终预测：$\hat{y} = h \cdot w_2$
- 训练集真实目标值：$y = 10.0$
- 损失函数（均方误差的一半）：$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$
- 学习率设定：$\eta = 0.01$

---

### 2. 前向传播：做出预测并计算损失

\lt fieldset>
\lt legend>\lt strong>计算流程清单</strong></legend>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>步骤 A：</strong> 前向计算隐藏层 $h = x \cdot w_1$。</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>步骤 B：</strong> 前向计算模型输出 $\hat{y} = h \cdot w_2$。</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>步骤 C：</strong> 计算当前误差损失 $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$。</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>步骤 D：</strong> 反向求出 $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ 及第 2 层权重梯度 $\frac{\partial \mathcal{L}}{\partial w_2}$。</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>步骤 E：</strong> 误差梯度逆流传给 $h$，求出第 1 层权重梯度 $\frac{\partial \mathcal{L}}{\partial w_1}$。</p>
\lt p>\lt input type="checkbox" checked disabled> \lt strong>步骤 F：</strong> 执行梯度下降参数更新，并验证新损失降低幅度。</p>
</fieldset>

#### 步骤 A：第 1 层前向
$$
h = x \cdot w_1 = 2.0 \times 3.0 = 6.0
$$

#### 步骤 B：第 2 层前向
$$
\hat{y} = h \cdot w_2 = 6.0 \times 2.0 = 12.0
$$

#### 步骤 C：损失计算
真实答案是 $10.0$，而模型给出了 $12.0$：

$$
\mathcal{L} = \frac{1}{2}(12.0 - 10.0)^2 = \frac{1}{2}(2.0)^2 = 2.0000
$$

---

### 3. 反向传播：逆向传递误差纸条

#### 步骤 D：计算第 2 层的误差与权重梯度
首先对输出预测 $\hat{y}$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = 12.0 - 10.0 = \mathbf{+2.0}
$$

对第 2 层的权重 $w_2$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot h = 2.0 \times 6.0 = \mathbf{+12.0}
$$

将梯度逆向传导回隐藏层状态 $h$（写给上一层向导的评语）：

$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot w_2 = 2.0 \times 2.0 = \mathbf{+4.0}
$$

#### 步骤 E：计算第 1 层的权重梯度
利用回传的 $\frac{\partial \mathcal{L}}{\partial h} = 4.0$，计算第 1 层权重 $w_1$ 的偏导：

$$
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot x = 4.0 \times 2.0 = \mathbf{+8.0}
$$

各层权重梯度归纳：
- $\nabla_{w_2} \mathcal{L} = +12.0$（正梯度指示：立刻把 $w_2$ 调小！）
- $\nabla_{w_1} \mathcal{L} = +8.0$（正梯度指示：立刻把 $w_1$ 调小！）

---

### 4. 梯度步进更新与效果验证

#### 步骤 F：应用梯度下降更新公式
使用学习率 $\eta = 0.01$：

$$
w_{2,\text{新}} = w_2 - \eta \frac{\partial \mathcal{L}}{\partial w_2} = 2.0 - (0.01 \times 12.0) = 2.0 - 0.12 = \mathbf{1.88}
$$

$$
w_{1,\text{新}} = w_1 - \eta \frac{\partial \mathcal{L}}{\partial w_1} = 3.0 - (0.01 \times 8.0) = 3.0 - 0.08 = \mathbf{2.92}
$$

#### 步骤 G：使用新权重执行一次全新的前向预测
用微调后的新旋钮重新计算：
- $h_{\text{新}} = x \cdot w_{1,\text{新}} = 2.0 \times 2.92 = 5.84$
- $\hat{y}_{\text{新}} = h_{\text{新}} \cdot w_{2,\text{新}} = 5.84 \times 1.88 = 10.9792$
- 目标真实值：$y = 10.0$

计算全新的误差损失：

$$
\mathcal{L}_{\text{新}} = \frac{1}{2}(10.9792 - 10.0)^2 = \frac{1}{2}(0.9792)^2 \approx \mathbf{0.4794}
$$

<mark>看：仅仅经历了一次微小的梯度修正，模型的误差损失就从 2.0000 暴跌至 0.4794——单步误差直降了整整 76.0%！</mark>

---

## 第 6 步：核心精要（一句话记住核心奥秘） {: #step-6 }

!!! tip "核心要点: 反向传播的核心心法"
    **反向传播的本质，就是多元微积分链式法则在网络计算图上的反向模式自动微分。**

    它从唯一的标量损失出发，自顶向下逆流遍历网络，单次反向回传即可同时解出全部数千亿参数的下山斜率，将训练大语言模型从原本不可逾越的百年算力泥潭中彻底解放出来。
