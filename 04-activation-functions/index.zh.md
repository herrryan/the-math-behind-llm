# 第 04 章：单向门——激活函数（ReLU、GELU 与 SwiGLU）

<nav aria-label="目录导航">
  <p>
    <strong>目录导航：</strong> 
    <a href="#step-1">1. 直觉理解</a> &bull; 
    <a href="#step-2">2. 过渡问题</a> &bull; 
    <a href="#step-3">3. 数学公式</a> &bull; 
    <a href="#step-4">4. 公式溯源</a> &bull; 
    <a href="#step-5">5. 简易计算范例</a> &bull; 
    <a href="#step-6">6. 核心精髓</a>
  </p>
</nav>

---

<h2 id="step-1">第 1 步：三岁孩子都能懂的物理直觉</h2>

想象你正坐在游戏房的手工桌前，桌上放着一张平整的彩色卡纸。

如果手工规则只允许你把纸张在桌面上滑动、旋转、或者用双手沿边缘拉伸（这正是我们在第 03 章中学习的线性矩阵乘法），那么这张纸将永远保持绝对的平整。

<figure>
<pre>
       平整卡纸被反复拉伸                     划出一道锐利折痕（非线性）
   ┌───────────────────────┐                    ▲
   │                       │                   / \
   │   永远保持 100% 平整   │      折叠        /   \     变成一只立体
   │   （平的叠加依然是平的）│    ────────►    /     \    纸千纸鹤！
   │                       │                /       \
   └───────────────────────┘               /         \
</pre>
<figcaption><strong>图 4.1：</strong> 无论叠加多少次平整变换，形状永远是平的。只有引入一道锋利的折痕（激活函数），二维纸片才能折叠成立体的千纸鹤。</figcaption>
</figure>

无论你把一张平整的纸拉伸或旋转多少万次，你都**绝不可能**折出一只立体的千纸鹤、一把手风琴或一架纸飞机。

要将二维平纸变成丰富的三维立体结构，你必须创造一道**锋利的折痕** &mdash; 也就是在某条线上，平直的规则被瞬间打破。

### 叠加平板玻璃的困局

再想象组装一架天文望远镜：
- 如果你把 100 块平整的窗户玻璃整整齐齐地叠在一起，然后透过它们看天空。所有的星星依然是原来的大小。把 100 块平板玻璃叠在一起，在光学上等价于**一块更厚的单层平板玻璃**，它根本无法放大任何微小的天体。
- 要放大遥远的星光，你必须磨制出**凸透镜或凹透镜**。透镜之所以能弯折光线，正是因为它的表面是**弯曲**的，每一个位置的斜率都在动态变化。

### 游乐园的单向旋转闸门

现在再想象游乐园入口处的单向旋转闸机：
- 如果小朋友带着正向的动力向前走，闸机顺畅旋转，放他们轻松通过。
- 如果有人想要倒退往回走，内部的棘爪齿轮瞬间锁死，坚决将其阻截在零位刻度线（$0$）。

这个单向旋转门，就是**激活函数（Activation Function）**。

在人工神经网络中，矩阵乘法就是平整的玻璃板；而激活函数，就是赋予整个网络**空间弯曲能力**的透镜与单向阀门。它使模型能够执行逻辑判断、筛选有效信息、并孕育出深邃的逻辑推理能力。

### 咔哒响的硬开关 vs. 平滑无级调光旋钮

在 20 世纪 80 年代之前，早期人工智能先驱们把神经元想象成墙壁上那种粗糙的、一按即响的咔哒开关：
- 轻轻按开关：什么都不会发生。
- 用力推过某个临界阈值：*咔哒！* 灯光瞬间全亮（输出 $1$）。

设想你置身于一间漆黑的暗室中，试图通过微调 10,000 个隐藏的开关来调出最适宜的房间亮度。如果每一个开关都只有绝对亮（1）和绝对灭（0）两个状态，你将彻底陷入抓瞎绝境。当你把手搭在一个开关上时，由于开关在被触碰的大部分位置表面都是完全平坦的（斜率恒为零），你**根本无法知道**自己距离触发它究竟只差一毫米，还是差了整整一米！

现在，把这个硬性开关换成一个**平滑无级的调光旋钮**：
- 当你的手指轻轻拨动哪怕十分之一毫米，室内的光线就会随之泛起一丝极其微弱的明暗变化。
- 这正是**微积分中的导数（Derivative，即 $\frac{dy}{dx}$）**：它度量了“输入微弱扰动所引发的输出微弱变化”。

导数赋予了计算机**触觉反馈**：它清晰地告诉优化算法**旋钮应当往哪个方向拧**、以及**用力该多大多猛**，才能有效减少预测误差。没有导数，计算机就宛如在漆黑绝境中失去了全部触觉与痛觉，根本无法展开学习。

---

<h2 id="step-2">第 2 步：计算跨越的桥梁问题</h2>

在第 03 章中，我们见证了矩阵乘法作为向量空间变换核心引擎的威力：

$$
\mathbf{y} = \mathbf{x} \mathbf{W}
$$

现代深度学习最引人注目的架构设计，就是将数十层甚至上百层网络纵向堆叠：

$$
\mathbf{x} \to \text{第 1 层} \to \text{第 2 层} \to \dots \to \text{第 } L \text{ 层} \to \mathbf{y}
$$

然而，这引发了一个致命的数学疑问：

> *“如果第 1 层计算 $\mathbf{h}_1 = \mathbf{x} \mathbf{W}_1$，第 2 层计算 $\mathbf{y} = \mathbf{h}_1 \mathbf{W}_2$，究竟是什么力量阻止了整座拥有 100 层的庞大网络，退化塌陷为仅仅一次单薄的矩阵乘法？”*

这揭示了一个根本事实：一个合格的激活函数必须在数学上**同时解决两大生死攸关的硬性要求**：
1. **非线性表达能力（前向传播）**：必须弯折、扭曲平直的高维向量空间，阻止深层网络塌陷退化为单层矩阵。
2. **平滑可微性与良态导数（反向传播）**：必须提供一条清晰、充沛的数学导数通道（$\frac{d\sigma}{dz}$），使反向传播算法能够计算参数更新方向。

> *“为什么我们不能随心所欲地挑选任意一条弯曲的折线（比如锯齿阶梯或无规律随机函数）？为什么数学导数（$\sigma'(z)$）是深度神经网络得以训练繁衍的生命维系系统？”*

---

<h2 id="step-3">第 3 步：严谨数学公式与推导</h2>

---

### 1. 线性塌陷灾难（The Linear Collapse）

让我们用严格的线性代数证明，为什么没有激活函数的“深度神经网络”毫无实际意义。

假设我们构建了一个 $L$ 层的深度网络，且每一层都只包含纯粹的线性变换。设输入行向量为 $\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{in}}}$。每一层包含权重矩阵 $\mathbf{W}_l$ 与偏置向量 $\mathbf{b}_l$：

$$
\begin{aligned}
\mathbf{h}_1 &= \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1 \\
\mathbf{h}_2 &= \mathbf{h}_1\mathbf{W}_2 + \mathbf{b}_2 = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2) + (\mathbf{b}_1\mathbf{W}_2 + \mathbf{b}_2) \\
\mathbf{h}_3 &= \mathbf{h}_2\mathbf{W}_3 + \mathbf{b}_3 = \mathbf{x}(\mathbf{W}_1\mathbf{W}_2\mathbf{W}_3) + (\mathbf{b}_1\mathbf{W}_2\mathbf{W}_3 + \mathbf{b}_2\mathbf{W}_3 + \mathbf{b}_3)
\end{aligned}
$$

通过数学归纳法推导至全部 $L$ 层，整个网络的所有矩阵乘积可直接合并为一个复合矩阵：

$$
\mathbf{W}_{\text{comb}} = \prod_{l=1}^L \mathbf{W}_l = \mathbf{W}_1 \mathbf{W}_2 \dots \mathbf{W}_L \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}
$$

复合偏置向量同样可以直接求和合并：

$$
\mathbf{b}_{\text{comb}} = \sum_{l=1}^L \mathbf{b}_l \left(\prod_{j=l+1}^L \mathbf{W}_j\right) \in \mathbb{R}^{1 \times d_{\text{out}}}
$$

最终整个百层网络的总输出，退化为一次最基础的线性回归：

$$
\mathbf{y} = \mathbf{x}\mathbf{W}_{\text{comb}} + \mathbf{b}_{\text{comb}}
$$

<fieldset>
<legend><strong>线性塌陷定理（Linear Collapse Theorem）</strong></legend>
无论你堆叠多少个隐藏层、耗费多少亿参数、燃烧多少千瓦时的算力：<strong>连续纯线性变换的级联，在数学上完全等价于仅仅单个线性变换层</strong>。它只能在空间中画出平直的超平面，甚至连最简单的 <dfn id="def-xor-problem">XOR（异或）</dfn> 逻辑关卡都无法突破。
</fieldset>

为了打破这种线性塌陷，我们必须在每次线性矩阵乘法之后，**穿插植入**一个逐元素执行的非线性标量函数 $\sigma(\cdot)$：

$$
\mathbf{h}_l = \sigma(\mathbf{h}_{l-1}\mathbf{W}_l + \mathbf{b}_l)
$$

由于非线性函数的存在打破了结合律：$\sigma(\mathbf{A}\mathbf{B}) \ne \sigma(\mathbf{A})\sigma(\mathbf{B})$，网络层与层之间再也无法被合并压缩。每增加一个隐藏层，模型所能逼近的高维函数曲面复杂度便呈指数级倍增。

---

### 2. 为什么导数神圣不可侵犯：参数敏感度与反向传播机制

为什么深度学习研究者如此痴迷于激活函数的数学导数 $\sigma'(z)$？

因为神经网络**在前向传播过程中根本不发生任何学习行为**。前向传播仅仅是将输入向量层层变换，输出一个候选词的预测概率。真正的智力进化，完全发生于**反向传播（Backpropagation）**与**梯度下降（Gradient Descent）**阶段。

而这一阶段的核心任务只有一件事：**为网络中的每一个权重参数 $w$，精确计算损失函数（预测误差）$\mathcal{L}$ 对它的敏感程度（Sensitivity）**。

#### 什么是“敏感程度”？（泰勒一阶展开与杠杆转化率）
数学上，损失 $\mathcal{L}$ 对某个特定权重 $w$ 的敏感程度就是偏导数 $\frac{\partial \mathcal{L}}{\partial w}$。根据一阶泰勒展开式，当我们给权重施加一个微小扰动 $\Delta w$ 时，最终误差的变化量 $\Delta \mathcal{L}$ 满足：

$$
\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}}{\partial w} \cdot \Delta w
$$

这个偏导数本质上就是**参数调整量向误差变动量映射的“杠杆转换率”**，它同时提供了参数更新的两大核心情报：
1. **调整方向（符号指示）**：
   - 若 $\frac{\partial \mathcal{L}}{\partial w} > 0$：说明 $w$ 越往大调，模型误差越大；为了让误差减小（$\Delta \mathcal{L} < 0$），我们必须**调小**该参数（$\Delta w < 0$）。
   - 若 $\frac{\partial \mathcal{L}}{\partial w} < 0$：说明 $w$ 越大，模型误差反而越小；为了让误差减小，我们必须**调大**该参数（$\Delta w > 0$）。
   - 这正是最经典的负梯度更新规则：$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$（其中 $\eta > 0$ 为学习率）。
2. **调整幅度（绝对值大小）**：
   - 绝对值 $\left|\frac{\partial \mathcal{L}}{\partial w}\right|$ 越大，说明该参数是决定模型成败的“高杠杆命门”，稍有晃动就会带来误差巨幅变动；
   - 绝对值趋近于 $0$，说明无论怎么拨动这个参数，对最终预测几乎毫无影响。

#### 为什么非算敏感度不可？（责任归因困境与算力绝境）
想象一架拥有 700 亿个控制旋钮的超巨型客机驾驶舱（对应 70B 权重参数 $w_1, w_2, \dots, w_B$）。飞机突然偏离航线 500 米（最终终端只产生了一个标量误差：$\mathcal{L} = 500$）。

此时仪表盘上只跳动着这一个最终误差数字，你该如何调整旋钮？究竟是第 4,231 号旋钮拧偏了，还是第 12,890,442 号旋钮拧反了？这就是著名的<dfn id="def-credit-assignment"><strong>责任归因困境（Credit Assignment Problem）</strong></dfn>。

如果我们不借助微积分求解敏感度，现实中只有两种绝望的替代路径：
- **盲目试凑（蒙特卡洛随机搜索）**：在 700 亿维超高维空间中，最优解所在的超体积测度趋近于零。即便动用全宇宙的所有粒子进行并行随机抽样，到宇宙热寂也绝不可能试凑出一组能通顺交流的参数组合。
- **逐个试探（有限差分数值扰动）**：
  若采用物理测试法，即对每个参数单独施加微小扰动 $\epsilon$，通过两次前向传播相减测量敏感度：
  
  $$
  \frac{\partial \mathcal{L}}{\partial w_i} \approx \frac{\mathcal{L}(w_i + \epsilon) - \mathcal{L}(w_i)}{\epsilon}
  $$

  对于 700 亿个参数的模型，仅仅为了进行**单次参数更新**，就需要执行整整 **700 亿次模型前向传播**！即便顶级超级集群以每秒 10 次前向传播极速运转，完成一次梯度更新也需要耗费整整 **221 年**。

所以，我们必须找到一种能在瞬间同时算出全部 700 亿个参数敏感度的算法——这就是**反向传播（Backpropagation）**。

#### 究竟什么是反向传播（Backpropagation）？
许多教科书直接甩出复杂的偏导数符号，让人望而生畏。但剥开术语外衣，反向传播的物理本质和计算逻辑极其自然。

##### 1. 直觉比喻：汽车流水线与质量责任溯源
- **前向传播（Forward Pass）**：
  如同汽车制造流水线。原料钢板（输入词向量 $\mathbf{x}$）从最左边送入：
  - 1 号工位（第 1 层）冲压焊接底盘；
  - 2 号工位（第 2 层）装配发动机；
  - 3 号工位（第 3 层）安装车身车门……
  - 最终流水线末端驶出一辆成品车（预测词概率分布 $\hat{\mathbf{y}}$）。
  **整个过程信息严格从左向右单向流动。各工位只负责处理上游送来的零件，没有任何学习行为。**
- **损失计算（Loss Computation）**：
  质检员在终点用卡尺测量，发现车门缝隙偏大了 5 毫米（产生标量损失 $\mathcal{L} = 5\text{ mm}$）。
- **反向传播（Backward Pass）**：
  质检员不可能把车砸掉重来，而是**沿着流水线倒着走回车间，逐级向下追溯责任**：
  - 质检员对 3 号工位（车门组装）说：“整车误差 5 毫米，你的铰链螺栓有直接责任，回调你的螺栓！”
  - 3 号工位收到反馈后，检查发现：“我这边的 5 毫米里，有 3 毫米是因为 2 号工位送来的发动机底座倾斜被放大了！”于是他立刻拍 2 号工位的肩膀：“你的底盘偏了，把你的底座往回调！”
  - 2 号工位依此类推，继续向 1 号工位反向传达修正指标……
  **误差信号逆流而上，每个工位在收到下游反馈后，立刻就能算出自己手中的工具该拧松还是拧紧。**

##### 2. 构成反向传播的三大底层积木
要把上述故事搬进计算机，我们只需要三个极其简单的前置数学积木：

- **积木一：计算图（Computational Graph）**
  复杂的神经网络在计算机底层，会被拆解成由最简单的基本算子节点构成的有向图：加法节点（$+$）、乘法节点（$\times$）、激活函数节点（$\sigma$）。每个节点只做一步小学级别的运算。
- **积木二：局部导数（Local Derivative / 本地齿轮比）**
  每个独立算子只对自己的局部输入输出负责，完全不关心全局：
  - 例如乘法算子 $z = w \cdot x$：如果参数输入 $w$ 变动一丁点，局部输出 $z$ 会变动多少？答案就是输入信号的大小：$\frac{\partial z}{\partial w} = x$。
  - 例如激活函数算子 $a = \sigma(z)$：如果输入 $z$ 变动一丁点，局部输出 $a$ 会变动多少？答案就是激活函数本身的导数：$\frac{\partial a}{\partial z} = \sigma'(z)$。
  **前向计算执行的同时，每个节点就能顺手在显存中存好自己的局部导数。**
- **积木三：链式法则（The Chain Rule / 齿轮级联定律）**
  想象三个咬合在一起的齿轮 A、B、C：齿轮 A 带动齿轮 B，齿轮 B 带动齿轮 C。
  - 齿轮 A 转 1 圈，齿轮 B 会转 3 圈（齿轮比 $\frac{dB}{dA} = 3$）；
  - 齿轮 B 转 1 圈，齿轮 C 会转 2 圈（齿轮比 $\frac{dC}{dB} = 2$）；
  - 请问：当齿轮 A 转动 1 圈时，最终的齿轮 C 会转动几圈？
  
  答案不言而喻：$3 \times 2 = 6$ 圈！
  用微积分语言写下来就是：
  
  $$
  \frac{dC}{dA} = \frac{dC}{dB} \times \frac{dB}{dA}
  $$
  
  **链式法则本质：终点对起点的总敏感度，就是沿途每一级局部导数（齿轮比）的连乘积！**

##### 3. 为什么必须“反向”而不是“正向”推导？（动态规划对重复计算的终结）
既然链式法则就是把局部导数乘起来，为什么不能从输入端顺着往前乘，非要从损失端逆着往回推？

答案在于**网络维度的极端不对称性**：
- **起点有 700 亿个参数，但终点只有一个标量损失 $\mathcal{L}$。**
- **如果正向推导（Forward-Mode）**：
  从参数 $w_1$ 出发一路向后追踪到 $\mathcal{L}$，得到 $\frac{\partial \mathcal{L}}{\partial w_1}$；然后再从参数 $w_2$ 出发，把后面的网络又重新追踪一遍……
  由于 700 亿个参数共用同一个下游网络，网络后半段将被**完全重复遍历 700 亿次**！计算量直接爆表。
- **如果反向推导（Reverse-Mode / 反向传播）**：
  从唯一的终点损失 $\mathcal{L}$ 出发！终点对自己的敏感度显而易见是 $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1$。
  我们往回退一步，计算输出层的误差信号 $\delta = \frac{\partial \mathcal{L}}{\partial a}$；
  再退一步，用已经算好的 $\delta$ 乘以当前节点的局部导数，得到上一节点的误差信号；
  **每一层计算出的误差信号立即缓存在内存中，供更早的所有上游节点直接复用！全网每一个算子节点仅被逆向遍历了一次。**
  仅仅用一次反向遍历（耗时仅为前向的 2 倍左右），全网全部 700 亿个参数对损失的偏导数同时求出！

#### 单神经元内前向与反向的数据流微观全景
现在，我们把这套反向机制对准神经元内部的激活函数。

考虑一个单神经元结构，它计算加权线性组合 $z = \sum_k w_k x_k + b$，随后经过激活函数 $a = \sigma(z)$。在前向传播中，特征信号从左流向右；在反向传播中，误差信号从右流向左：

<figure>
<pre>
【前向计算通道：从左向右生成预测】
输入 x ───► [ 乘法节点 z = w·x ] ───► 中间量 z ───► [ 激活节点 a = σ(z) ] ───► 输出 a ───► ... ───► 损失 L
                       ▲                                      ▲
                       │                                      │
                   权重参数 w                             激活函数 σ

─────────────────────────────────────────────────────────────────────────────────────────────────

【反向传播通道：从右向左溯源责任】
敏感度 ∂L/∂w ◄── [ 乘上本地导数 x ] ◄── 误差信号 ∂L/∂z ◄── [ 乘上本地导数 σ'(z) ] ◄── 误差信号 ∂L/∂a ◄── ...
      │                                                                           ▲
      ▼                                                                           │
  更新参数 w                                                       下游网络传回的累积误差
</pre>
<figcaption><strong>图 4.2：</strong> 神经元内部前向计算与反向传播的镜像双通道。前向传递特征值，反向传递责任误差信号。</figcaption>
</figure>

根据微积分链式法则，损失 $\mathcal{L}$ 对权重 $w_k$ 的偏导数（敏感度）可严格展开为三项连乘：

$$
\frac{\partial \mathcal{L}}{\partial w_k} = \underbrace{\frac{\partial \mathcal{L}}{\partial a}}_{\text{下游传回的累积误差 } \delta} \cdot \underbrace{\frac{\partial a}{\partial z}}_{\mathbf{\sigma'(z)}} \cdot \underbrace{\frac{\partial z}{\partial w_k}}_{x_k}
$$

<figure>
<pre>
              下游反向传回的误差信号 (∂L/∂a)
                          │
                          ▼
                  ┌───────────────┐
                  │     σ'(z)     │  ◄─── 激活函数的导数扮演了物理阀门角色！
                  └───────┬───────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        若 σ'(z) = 0            若 σ'(z) ≈ 1
       梯度通道被彻底切断        误差信号畅通无阻
     ∂L/∂w = 0（网络冻结）      权重平稳更新进化
</pre>
<figcaption><strong>图 4.3：</strong> 激活函数的导数 $\sigma'(z)$ 是误差梯度回传的传动阀门。若导数为零，误差信号将在此瞬间中断，连接前端的权重完全丧失更新能力。</figcaption>
</figure>

观察链式乘积中间项 $\frac{\partial a}{\partial z} = \sigma'(z)$ 的枢纽地位：
- **导数就是物理水阀的开合度**：它直接决定了下游的误差信号是以多大比例传导给权重。
- **阶跃函数的毁灭性打击**：如果使用赫维赛德阶跃函数 $\Theta(z)$，除了跳跃点外其导数处处为 0（在跳跃点导数不存在）：

$$
\frac{\partial \mathcal{L}}{\partial w_k} = \frac{\partial \mathcal{L}}{\partial a} \cdot \mathbf{0} \cdot x_k = 0
$$

  误差梯度瞬间归零湮灭，所有权重全部锁死冻结，现代深度学习赖以生存的反向传播直接报废！

#### 反向传播被堵死，在现实中到底意味着什么？
很多初学者看到“导数归零、梯度消失”时，只觉得是一个抽象的数学符号推导。但在真实的大模型训练中，这意味着四场灾难性的毁灭后果：

1. **数学计算上的瘫痪：参数永久冻结在随机初始状态**
   神经网络更新权重的核心机制是梯度下降：$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$。一旦敏感度 $\frac{\partial \mathcal{L}}{\partial w} = 0$，参数更新量便恒等于零：

$$
w_{\text{新}} = w_{\text{旧}} - \eta \times 0 = w_{\text{旧}}
$$

   这意味着无论你喂给网络几万亿词元的训练语料，无论成千上万张显卡狂轰滥炸燃烧多少度电，这些参数在几百天里都不会发生哪怕一丁点的修正！它从刚开机初始化是什么随机样，训练结束就还是什么样。
2. **多层级联的雪崩式断链：上游所有层全被连带“饿死”**
   根据链式法则，上一层的反向误差为：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{l-1}} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{a}_l} \odot \sigma'(\mathbf{z}_l)\right) \mathbf{W}_l
$$

   只要第 $l$ 层的激活导数 $\sigma'(\mathbf{z}_l) = \mathbf{0}$，传向上游的误差信号就变成了绝对的零向量 $\mathbf{0}$。
   这就像供水管道在第 50 层被一道铁闸门死死焊死：不仅第 50 层收不到水，**位于它前面的第 49 层、第 48 层……直到第 1 层的所有上游权重，全部同时断水断粮**！整座大厦的上游完全沦为瘫痪的“数字僵尸”。
3. **表征学习彻底破产：模型“五官未开”，深层在乱码上搭积木**
   深度模型具备强大的语言理解能力，根本原因在于各层分工明确的**层次化表征学习**：
   - **底层（第 1 ~ 10 层）**：学习词根、标点、词性、基础语法规则等基本砖石；
   - **中层（第 11 ~ 40 层）**：学习句法结构、指代关系、实体关联；
   - **高层（第 41 ~ 96 层）**：学习复杂逻辑、长程推理、常识推理。
   如果反向误差信号无法下渗到浅层，浅层网络就永远停留在刚初始化时的随机噪声状态！模型的“眼睛与耳朵”从头到尾都是瞎的。高层网络只能被动地试图在一堆未经训练的乱码之上去拼凑高阶逻辑，最终整个模型吐出的只能是毫无意义的胡言乱语。
4. **虚假深度的算力欺骗：百层架构退化为极浅层网络**
   如果你花数千万美元搭建了一个 100 层的深层网络，但误差梯度在第 90 层就已彻底衰减熄灭，那么前面 89 层的作用与一个纯随机数发生器毫无区别。你实质上只是在用前 89 层的随机乱码作为输入，去训练一个仅仅 11 层的极浅层模型！这就是为什么在 20 世纪 90 年代，研究人员发现“把网络做深反而比浅层网络表现更差”的根本根源。
- **20 世纪 80 年代的算力奇迹**：在早期 CPU 算力极度匮乏的年代，浮点指数与除法运算极其昂贵。Sigmoid（$\sigma$）与 Tanh（$\tanh$）之所以被奉为圭臬，正是因为它们的导数可以**直接利用前向传播已经存入内存的激活值 $a$ 极速算出**，无需调用任何昂贵的超越函数：

$$
\sigma'(z) = a(1 - a), \quad \tanh'(z) = 1 - a^2
$$

  仅仅需要一次减法与一次乘法，计算机就能以极高效率完成整网的反向传播！

---

### 3. 第一时代：Sigmoid、Tanh 与梯度消失危机

早期神经网络研究者借鉴生物神经元的电生理响应特性，采用了平滑的 S 型激活曲线。

#### Logistic Sigmoid 函数
将实数域输入 $z \in (-\infty, \infty)$ 压缩映射至概率区间 $(0, 1)$：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

其导数具备极其优美的代数结构：

$$
\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))
$$

#### 双曲正切函数（Tanh）
零中心化的 S 型函数，将输入映射至 $(-1, 1)$ 区间：

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
$$

其导数为：

$$
\frac{d\tanh(z)}{dz} = 1 - \tanh^2(z)
$$

<figure>
<pre>
       激活输出值 σ(z)                             导数值 σ'(z)
 1.0 ┌───────────────────----┐         0.25 ┌─────────/\─────────┐  最大峰值仅为 0.25
     │                     / │              │        /  \        │  在 z = 0 处
 0.5 │........./‾‾‾‾‾........│              │       /    \       │
     │        /              │              │     /        \     │  当 |z| > 4 时
 0.0 └───----────────────────┘         0.00 └───/────────────\───┘  导数迅速衰减为 0
    -6  -4  -2   0   2   4   6             -6  -4  -2   0   2   4   6
</pre>
<figcaption><strong>图 4.4：</strong> 梯度消失危机：对于绝对值较大的正负输入，Sigmoid 导数剧烈跌落归零。在深层反向传播中跨层连乘，会导致误差信号彻底熄灭。</figcaption>
</figure>

#### 为什么 Sigmoid 和 Tanh 在深度网络中全面崩溃？
请注意观察 Sigmoid 导数的极值特性：当输入处于原点 $z = 0$ 时，导数取得最大理论峰值 $\sigma'(0) = 0.5 \times (1 - 0.5) = 0.25$。只要输入偏离原点，导数值便急剧萎缩至低于 $0.25$。

在深度反向传播计算中，根据微积分链式法则，损失函数的梯度需要沿着层级向后逐层反向连乘：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \prod_{l=1}^L \left(\mathbf{W}_l^\top \cdot \operatorname{diag}(\sigma'(\mathbf{z}_l))\right)
$$

如果每一层都将反向传导的误差梯度缩减至原本的至多 $0.25$ 倍，那么在仅仅穿越 $L = 10$ 层之后：

$$
(0.25)^{10} \approx 0.00000095
$$

梯度信号衰减了上百万倍，几近归零！靠近输入端的浅层参数根本接收不到有效的更新指导，整个深层网络陷入停滞。这就是著名的<dfn id="def-vanishing-gradient"><strong>梯度消失问题（Vanishing Gradient Problem）</strong></dfn>。

---

### 4. 第二时代：ReLU 革命（线性整流单元）

2010 年至 2012 年间，深度学习先驱意识到：盲目模仿生物神经元的 S 型曲线反而成为了深度网络的桎梏。他们抛弃了复杂的自然指数计算，换用了最朴素的单向阈值截断：<dfn id="def-relu"><strong>线性整流单元（ReLU, Rectified Linear Unit）</strong></dfn>。

$$
\operatorname{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{若 } z > 0 \\ 0 & \text{若 } z \le 0 \end{cases}
$$

其导数在正半轴恒定为 1：

$$
\frac{d\operatorname{ReLU}(z)}{dz} = \begin{cases} 1 & \text{若 } z > 0 \\ 0 & \text{若 } z < 0 \end{cases}
$$

*(注：在 $z = 0$ 拐点处数学上不可导，但工程实现在代码中统一定义其分段次梯度为 $0$ 或 $1$。)*

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 4.1：</strong> 为什么 ReLU 彻底改写了深度学习的历史轨迹</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left" width="20%">关键特性</th>
      <th scope="col" align="left" width="40%">传统 Sigmoid / Tanh</th>
      <th scope="col" align="left" width="40%">现代化 ReLU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>正半轴梯度</strong></th>
      <td>随输入绝对值增大指数衰减趋近于 0</td>
      <td><strong>恒为 $1.0$</strong> &mdash; 误差梯度可以在 100 层间无损穿透</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>计算开销</strong></th>
      <td>昂贵的浮点数指数运算（$e^{-z}$）与除法</td>
      <td><strong>单条 GPU 汇编比较指令</strong>（<code>max(0, x)</code>）极速完成</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>表征稀疏性</strong></th>
      <td>稠密：所有神经元几乎都输出非零连续值</td>
      <td><strong>真稀疏表征</strong>：约 50% 负向神经元精准归零，显著降低冗余</td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>潜在隐患</strong></th>
      <td>全网络梯度消失，彻底锁死深层结构</td>
      <td><dfn id="def-dying-relu"><strong>神经元死亡（Dying ReLU）</strong></dfn>：参数一旦跌入负死区将永久失去梯度</td>
    </tr>
  </tbody>
</table>

#### “神经元死亡（Dying ReLU）”缺陷
如果网络在训练过程中遭遇过大的学习率冲击，导致某神经元的权重更新后在所有训练样本上均满足 $\mathbf{x}\mathbf{W} + b < 0$，那么该神经元的输出将恒为 $0$，其反向传播梯度也永久为 $0$。该神经元在数学上便永久“死亡”，再也无法自我苏醒。

---

### 5. 现代 Transformer 时代：GELU（GPT-2、GPT-3 与 BERT）

在打造大型语言模型与 Transformer 骨架时，研究人员提出了新的思考：*我们能否既保留 ReLU 梯度不消失的线性优势，又抹去 $z = 0$ 处过于生硬死板的折角突变？*

2016 年，Dan Hendrycks 与 Kevin Gimpel 提出了<dfn id="def-gelu"><strong>高斯误差线性单元（GELU, Gaussian Error Linear Unit）</strong></dfn>。

GELU 的核心思想不再是确定性地按正负号“一刀切”，而是根据标准正态分布随机变量落在当前输入以下的**概率累积值**，对输入 $z$ 进行平滑加权放行：

$$
\operatorname{GELU}(z) = z \cdot \Phi(z) = z \cdot P(X \le z), \quad \text{其中 } X \sim \mathcal{N}(0, 1)
$$

其中 $\Phi(z)$ 是标准高斯分布的累积分布函数（<abbr title="Cumulative Distribution Function">CDF</abbr>）：

$$
\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-\frac{t^2}{2}} \, dt = \frac{1}{2} \left[1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right]
$$

<figure>
<pre>
   ReLU（在原点处存在生硬折角）               GELU（平滑过渡并在负轴形成平缓浅谷）
 2.0 ┌                     /       2.0 ┌                     /
     │                    /            │                    /
 1.0 │                   /         1.0 │                   /
     │                  /              │                  /
 0.0 └─────────--------┌───        0.0 └─────────-.....-─/───
    -3   -2   -1   0   1   2          -3   -2   -1   0   1   2
             严格截断为 0                    在 z = -0.75 处平滑下探至 -0.17 浅谷
</pre>
<figcaption><strong>图 4.5：</strong> ReLU 与 GELU 的形态对比。注意 GELU 在负半轴的平滑过渡：微弱的负向特征不会被暴力切断，而是被温柔地保留微量梯度。</figcaption>
</figure>

#### GELU 的快速 Tanh 近似公式
由于高斯误差函数 $\operatorname{erf}(\cdot)$ 在硬件底层直接求积开销较大，初代 GPT-2、GPT-3 与 BERT 普遍采纳了 Hendrycks 提出的快速数值近似公式：

$$
\operatorname{GELU}(z) \approx 0.5 z \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(z + 0.044715 z^3\right)\right)\right)
$$

GELU 的三大数学优势：
1. **渐近等价**：当 $z \to +\infty$ 时，$\Phi(z) \to 1$，故 $\operatorname{GELU}(z) \to z$（与 ReLU 一样无梯度衰减）。
2. **负向平滑抑制**：当 $z \to -\infty$ 时，$\Phi(z) \to 0$，故 $\operatorname{GELU}(z) \to 0$。
3. **负轴梯度保留通道**：在轻微负值区域（$z \approx -0.7517$ 附近），$\operatorname{GELU}(z)$ 会自然下探出一个极小值谷底（约 $-0.170$），允许微弱的负反馈梯度安全通过，彻底攻克了 ReLU 的“神经元永久死亡”难题。

---

### 6. 当今前沿标准：SwiGLU（LLaMA-3、Mistral、Gemma、DeepSeek）

2020 年，原 Google Brain 架构科学家 Noam Shazeer 发表了一篇极具影响力的里程碑论文：*《GLU 变体全方位提升 Transformer（GLU Variants Improve Transformer）》*。

Shazeer 提出了根本性的构想：*为什么神经元的激活只能是一个固定的单输入标量函数？为什么不能让一条矩阵分支去充当动态阀门，实时调节另一条矩阵分支的信号强弱？*

#### 门控线性单元（GLU）的演化
Dauphin 等人在 2017 年率先提出了<dfn id="def-glu"><strong>门控线性单元（GLU, Gated Linear Unit）</strong></dfn>。它将输入向量经由两个独立的权重矩阵投影为两条并行支路：

$$
\operatorname{GLU}(\mathbf{x}, \mathbf{W}, \mathbf{V}) = (\mathbf{x}\mathbf{W}) \odot \sigma(\mathbf{x}\mathbf{V})
$$

其中：
- $\mathbf{x}\mathbf{W}$ 为**候选值分支**（承载特征的候选信息信号）。
- $\sigma(\mathbf{x}\mathbf{V})$ 为**门控分支**（通过 Sigmoid 生成在 $[0, 1]$ 之间浮动的连续门控系数）。
- $\odot$ 代表逐元素阿达马积（Hadamard Product）。

#### Swish / SiLU 激活函数
在构建 SwiGLU 之前，Shazeer 将原本生硬饱和的 Sigmoid 门控替换成了更流畅的 <dfn id="def-swish"><strong>Swish</strong></dfn>（又称 <abbr title="Sigmoid Linear Unit">SiLU</abbr>，由 Ramachandran 等人于 2017 年提出）：

$$
\operatorname{Swish}_1(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

#### SwiGLU 完整数学形式
将 Swish 激活与双线性门控单元深度融合，便诞生了统治现代开源大语言模型的 <dfn id="def-swiglu"><strong>SwiGLU</strong></dfn>：

$$
\operatorname{SwiGLU}(\mathbf{x}) = \operatorname{Swish}_1(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})
$$

在现代大语言模型的前馈全连接层（<abbr title="Feed-Forward Network">FFN</abbr>）中，完整的 SwiGLU 模块首先将隐藏向量投影升维至超宽的中间维度 $d_{\text{ffn}}$，完成双分支交互后，再通过降维矩阵投影回模型主维度 $d_{\text{model}}$：

$$
\operatorname{FFN}_{\text{SwiGLU}}(\mathbf{x}) = \left(\operatorname{Swish}_1(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})\right)\mathbf{W}_{\text{down}}
$$

<figure>
<pre>
                    输入 Token 隐藏向量 x  [1 × d_model]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       W_gate [d_model × d_ffn]        W_up [d_model × d_ffn]
               │                               │
               ▼                               │
            Swish(·)                           │
               │                               │
               └───────────────┬───────────────┘
                               ▼
                      逐元素哈达玛积 ⊙
                               │
                               ▼
                     中间特征向量 h  [1 × d_ffn]
                               │
                               ▼
                      W_down [d_ffn × d_model]
                               │
                               ▼
                     最终输出向量 y  [1 × d_model]
</pre>
<figcaption><strong>图 4.6：</strong> LLaMA-3、Mistral、Gemma 和 DeepSeek 普遍采用的现代 SwiGLU 前馈网络拓扑。两条并行矩阵分别计算门控与候选值，经过双线性乘积调制后输出。</figcaption>
</figure>

#### SwiGLU 的参数量对齐与缩放秘诀
传统 GPT 风格的前馈网络仅仅包含**两**个矩阵：
- 升维矩阵 $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$
- 降维矩阵 $\mathbf{W}_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$
- 总参数量为：$2 \times 4 d_{\text{model}}^2 = 8 d_{\text{model}}^2$。

但 SwiGLU 拥有**三**个矩阵（$\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$）。如果仍设定 $d_{\text{ffn}} = 4d_{\text{model}}$，参数量将暴涨 50%（$3 \times 4 = 12 d_{\text{model}}^2$）。

为了确保模型总参数量与算力开销严格与传统模型对齐，Shazeer 将中间隐藏维度缩放为 $\frac{8}{3}d_{\text{model}}$：

$$
d_{\text{ffn}} \approx \left\lfloor \frac{8}{3} d_{\text{model}} \right\rfloor = \left\lfloor \frac{2}{3} \times 4 d_{\text{model}} \right\rfloor
$$

在现代架构（如 LLaMA-3）中，这个数值还会进一步向上取整至 256 或 1024 的整倍数，以便完全契合英伟达 GPU Tensor Core 的内存硬件对齐规范。

---

### 7. 严谨数学变量与维度对照表

<details>
<summary><strong>点击展开：完整数学符号与维度速查手册</strong></summary>

<dl>
  <dt><strong>$\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{model}}}$</strong></dt>
  <dd>单层输入中单个 Token 对应的激活行向量。</dd>

  <dt><strong>$d_{\text{model}}$</strong></dt>
  <dd>模型的核心隐藏特征维度（例如 LLaMA-3 8B 中为 $4096$，70B 中为 $8192$）。</dd>

  <dt><strong>$d_{\text{ffn}}$</strong></dt>
  <dd>前馈网络扩展的中间维度，通常取 $\approx \frac{8}{3}d_{\text{model}}$（在 LLaMA-3 8B 中精准设定为 $14336$）。</dd>

  <dt><strong>$\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$</strong></dt>
  <dd>门控投影权重矩阵，负责动态调节各通道的信息流通量。</dd>

  <dt><strong>$\mathbf{W}_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$</strong></dt>
  <dd>升维候选投影权重矩阵，负责产出未过滤的候选特征向量。</dd>

  <dt><strong>$\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$</strong></dt>
  <dd>降维输出权重矩阵，将完成门控交互的向量投影回主模型维度。</dd>

  <dt><strong>$\odot$</strong></dt>
  <dd>逐元素阿达马积（Hadamard Product）：$(\mathbf{a} \odot \mathbf{b})_i = a_i \cdot b_i$。</dd>

  <dt><strong>$\Phi(z)$</strong></dt>
  <dd>标准高斯累积分布函数：$\Phi(z) = P(X \le z)$，其中随机变量 $X \sim \mathcal{N}(0, 1)$。</dd>
</dl>

</details>

---

<h2 id="step-4">第 4 步：历史渊源与技术演进</h2>

激活函数的演化史，正是深度学习探寻一种既能**极度强大地表达非线性**、又**处处平滑可导**、且在百层反向传播中**数值极其稳定**的数学实体的奋斗史。

<dl>
  <dt><time datetime="1943">1943年</time> &mdash; <strong>Warren McCulloch 与 Walter Pitts</strong>：阶跃生物神经元</dt>
  <dd>
    诞生了历史上第一个神经元数学模型，基于单位阶跃阈值函数（Heaviside step function）：

$$
f(z) = \begin{cases} 1 & \text{若 } z \ge \theta \\ 0 & \text{若 } z < \theta \end{cases}
$$

    <strong>为什么受挫：</strong> 阶跃函数除了跳跃点外，其余所有位置的导数恒为零（$f'(z) = 0$）。当梯度处处为零时，基于微积分的梯度下降算法根本无法启动！
  </dd>

  <dt><time datetime="1986">1986年</time> &mdash; <strong>Rumelhart、Hinton 与 Williams</strong>：平滑 Sigmoid 破冰</dt>
  <dd>
    为了让反向传播算法能够运转，先驱们将阶跃函数软化为处处可导的 Logistic 曲线 $\sigma(z) = \frac{1}{1 + e^{-z}}$。非零梯度的诞生，终于让人类能够成功训练多层神经网络。<cite>《Learning representations by back-propagating errors》, Nature 1986</cite>。
  </dd>

  <dt><time datetime="1991">1991年</time> &mdash; <strong>Sepp Hochreiter</strong>：梯度消失的数学病理确诊</dt>
  <dd>
    Hochreiter 在其德国学术毕业论文中，从数学理论上严谨证明了 S 型函数连乘将导致误差梯度呈指数级衰退枯竭，指出了神经网络无法做深的根本死因。
  </dd>

  <dt><time datetime="2010">2010年</time>&ndash;<time datetime="2012">2012年</time> &mdash; <strong>Nair、Hinton 与 Krizhevsky</strong>：ReLU 革命</dt>
  <dd>
    Vinod Nair 与 Geoffrey Hinton 在玻尔兹曼机中引入 ReLU（2010），随后 Alex Krizhevsky 在历史性的 AlexNet 中用其摧枯拉朽般拿下 ImageNet 冠军（2012）。正半轴恒为 1.0 的导数不仅让训练提速 6 倍，更彻底解除了深度的封印。
  </dd>

  <dt><time datetime="2016">2016年</time> &mdash; <strong>Dan Hendrycks 与 Kevin Gimpel</strong>：GELU 与初代 LLM 黄金时代</dt>
  <dd>
    Hendrycks 将随机正则化思想与激活函数熔铸一体，发明了 GELU。OpenAI 在训练初代 GPT-1、GPT-2、GPT-3 以及 Google 训练 BERT 时全盘采纳，GELU 成为大模型爆发第一浪潮的核心基石。<cite>《Gaussian Error Linear Units (GELUs)》, arXiv:1606.08415</cite>。
  </dd>

  <dt><time datetime="2020">2020年</time> &mdash; <strong>Noam Shazeer</strong>：SwiGLU 登顶现代王座</dt>
  <dd>
    Noam Shazeer 证明双线性乘积门控在语言建模困惑度与推理评测上全方位超越单输入激活。如今，几乎所有世界主流大模型 &mdash; Meta 的 <strong>LLaMA-3</strong>、法国 <strong>Mistral</strong>、Google <strong>Gemma</strong>、中国开源标杆 <strong>DeepSeek</strong> 与 <strong>Qwen</strong> &mdash; 均一致以 SwiGLU 为核心架构。<cite>《GLU Variants Improve Transformer》, arXiv:2002.05202</cite>。
  </dd>
</dl>

---

<h2 id="step-5">第 5 步：可手算验证的简易计算范例</h2>

现在，让我们用最简便的袖珍数字，在草稿纸上一步一步亲手完成完整的算术推导。

---

### 第 A 部分：线性塌陷的实际数值验证

设输入行向量为 $\mathbf{x} = \begin{bmatrix} 2 & 1 \end{bmatrix}$。

设网络第 1 层与第 2 层为纯粹的线性变换矩阵：

$$
\mathbf{W}_1 = \begin{bmatrix} 1 & 2 \\ 0 & 3 \end{bmatrix}, \quad \mathbf{W}_2 = \begin{bmatrix} -1 & 0 \\ 2 & 1 \end{bmatrix}
$$

#### 检验方式 1：逐层前向传播推导

**1. 第 1 层输出：**

$$
\mathbf{h}_1 = \mathbf{x}\mathbf{W}_1 = \begin{bmatrix} 2(1) + 1(0) & 2(2) + 1(3) \end{bmatrix} = \begin{bmatrix} 2 & 7 \end{bmatrix}
$$

**2. 第 2 层输出：**

$$
\mathbf{y} = \mathbf{h}_1\mathbf{W}_2 = \begin{bmatrix} 2(-1) + 7(2) & 2(0) + 7(1) \end{bmatrix} = \begin{bmatrix} -2 + 14 & 0 + 7 \end{bmatrix} = \begin{bmatrix} 12 & 7 \end{bmatrix}
$$

#### 检验方式 2：将两个矩阵提前合并为单一复合矩阵

**1. 预先计算复合矩阵 $\mathbf{W}_{\text{comb}} = \mathbf{W}_1 \mathbf{W}_2$：**

$$
\mathbf{W}_{\text{comb}} = \begin{bmatrix} 1(-1) + 2(2) & 1(0) + 2(1) \\ 0(-1) + 3(2) & 0(0) + 3(1) \end{bmatrix} = \begin{bmatrix} 3 & 2 \\ 6 & 3 \end{bmatrix}
$$

**2. 单步直达前向运算：**

$$
\mathbf{y} = \mathbf{x}\mathbf{W}_{\text{comb}} = \begin{bmatrix} 2(3) + 1(6) & 2(2) + 1(3) \end{bmatrix} = \begin{bmatrix} 6 + 6 & 4 + 3 \end{bmatrix} = \begin{bmatrix} 12 & 7 \end{bmatrix}
$$

两种计算方式得出了分毫不差的相同向量 $\begin{bmatrix} 12 & 7 \end{bmatrix}$。事实证明：两个纯线性层毫无悬念地彻底退化为了一层！

---

### 第 B 部分：ReLU、GELU 与 Swish 输出实测对比

设经过前向线性投影后的待激活向量为：

$$
\mathbf{z} = \begin{bmatrix} 2.0 & 0.0 & -1.5 \end{bmatrix}
$$

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 4.2：</strong> 输入 $\mathbf{z} = [2.0, 0.0, -1.5]$ 在各激活函数下的手算输出对比</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th scope="col" align="left">输入分量 $z$</th>
      <th scope="col" align="center">$\operatorname{ReLU}(z)$</th>
      <th scope="col" align="center">$\operatorname{GELU}(z)$</th>
      <th scope="col" align="center">$\operatorname{Swish}_1(z)$</th>
      <th scope="col" align="left">激活能量动态计量表</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row" align="left"><strong>$z_1 = +2.0$</strong></th>
      <td align="center">$\max(0, 2.0) = \mathbf{2.000}$</td>
      <td align="center">$2.0 \times \Phi(2.0) \approx \mathbf{1.954}$</td>
      <td align="center">$2.0 \times \sigma(2.0) \approx \mathbf{1.762}$</td>
      <td><meter min="-0.5" max="2.0" low="0.0" high="1.5" optimum="1.8" value="1.954">1.954</meter></td>
    </tr>
    <tr bgcolor="#fcfcfc">
      <th scope="row" align="left"><strong>$z_2 = 0.0$</strong></th>
      <td align="center">$\max(0, 0.0) = \mathbf{0.000}$</td>
      <td align="center">$0.0 \times \Phi(0.0) = \mathbf{0.000}$</td>
      <td align="center">$0.0 \times \sigma(0.0) = \mathbf{0.000}$</td>
      <td><meter min="-0.5" max="2.0" low="0.0" high="1.5" optimum="1.8" value="0.0">0.000</meter></td>
    </tr>
    <tr>
      <th scope="row" align="left"><strong>$z_3 = -1.5$</strong></th>
      <td align="center">$\max(0, -1.5) = \mathbf{0.000}$ <del>（硬性截断）</del></td>
      <td align="center">$-1.5 \times \Phi(-1.5) \approx \mathbf{-0.100}$ <ins>（软性保留）</ins></td>
      <td align="center">$-1.5 \times \sigma(-1.5) \approx \mathbf{-0.274}$ <ins>（软性保留）</ins></td>
      <td><meter min="-0.5" max="2.0" low="0.0" high="1.5" optimum="1.8" value="-0.100">-0.100</meter></td>
    </tr>
  </tbody>
</table>

清晰可见：ReLU 将负分量 $z_3 = -1.5$ 粗暴地归零；而 GELU 与 Swish 则放行了微弱的负向涓流（分别为 $-0.100$ 与 $-0.274$），从而让该特征在反向传播中依旧保留了苏醒的学习能力！

#### 真实反向传导实测：为什么 GELU 能防止神经元死亡？
假设在反向传播中，从深层网络传回该神经元输出端的累积误差信号为 $\delta = \frac{\partial \mathcal{L}}{\partial a} = 1.0$。我们来检验负输入分量 $z_3 = -1.5$ 处的反向传导状况：

1. **ReLU 的反向反馈**：
   由于 $z_3 = -1.5 < 0$，根据导数定义 $\operatorname{ReLU}'(-1.5) = 0$：

$$
\frac{\partial \mathcal{L}}{\partial z_3} = \delta \cdot \operatorname{ReLU}'(-1.5) = 1.0 \times 0 = 0.0
$$

   传导阀门被完全焊死！上游所有相连权重的梯度全被归零（$\frac{\partial \mathcal{L}}{\partial w} = 0$），该神经元陷入永久死亡。
2. **GELU 的反向反馈**：
   在 $z_3 = -1.5$ 处，通过导数公式可得局部导数 $\operatorname{GELU}'(-1.5) \approx -0.045$：

$$
\frac{\partial \mathcal{L}}{\partial z_3} = \delta \cdot \operatorname{GELU}'(-1.5) = 1.0 \times (-0.045) = -0.045
$$

   阀门并未锁死！一个虽微弱但非零的修正信号成功渗透传回，使上游权重在梯度下降中得以微调（$w \leftarrow w - \eta \cdot (-0.045 x)$），为神经元苏醒保留了生命火种。

---

### 第 C 部分：完整 SwiGLU 模块前向计算全流程

让我们手动跑通一次现代大模型中原汁原味的 SwiGLU 前馈层计算：
- 输入向量：$\mathbf{x} = \begin{bmatrix} 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{1 \times 2}$（主维度 $d_{\text{model}} = 2$）。
- 中间升维维度：$d_{\text{ffn}} = 2$。

权重矩阵配置如下：

$$
\mathbf{W}_{\text{gate}} = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}, \quad \mathbf{W}_{\text{up}} = \begin{bmatrix} 2 & 1 \\ 0 & -1 \end{bmatrix}, \quad \mathbf{W}_{\text{down}} = \begin{bmatrix} 1 & 2 \\ 1 & 0 \end{bmatrix}
$$

<fieldset>
<legend><strong>执行清单：SwiGLU 前向传播分步计算核对</strong></legend>

<p><input type="checkbox" checked disabled> <strong>第 1 步：计算门控分支投影</strong><br>
将输入特征投影到门控调节空间：</p>

$$
\mathbf{g} = \mathbf{x}\mathbf{W}_{\text{gate}} = \begin{bmatrix} 1(1) + 2(-1) & 1(0) + 2(1) \end{bmatrix} = \begin{bmatrix} -1.0 & 2.0 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>第 2 步：使用 $\operatorname{Swish}_1$ 激活门控系数</strong><br>
对门控分量分别应用连续平滑放行曲线：</p>
<ul>
  <li>分量 1：$g_1 = -1.0 \implies \sigma(-1.0) \approx 0.2689 \implies \operatorname{Swish}(-1.0) = -1.0 \times 0.2689 = \mathbf{-0.269}$</li>
  <li>分量 2：$g_2 = 2.0 \implies \sigma(2.0) \approx 0.8808 \implies \operatorname{Swish}(2.0) = 2.0 \times 0.8808 = \mathbf{1.762}$</li>
</ul>

$$
\operatorname{Swish}(\mathbf{g}) \approx \begin{bmatrix} -0.269 & 1.762 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>第 3 步：计算候选值分支投影（Up-Projection）</strong><br>
提取准备被筛选的原始特征信息：</p>

$$
\mathbf{u} = \mathbf{x}\mathbf{W}_{\text{up}} = \begin{bmatrix} 1(2) + 2(0) & 1(1) + 2(-1) \end{bmatrix} = \begin{bmatrix} 2.0 & -1.0 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>第 4 步：门控动态调制（逐元素阿达马积 $\odot$）</strong><br>
用门控系数乘上候选信息，实施动态过滤与放大：</p>

$$
\mathbf{h} = \operatorname{Swish}(\mathbf{g}) \odot \mathbf{u} = \begin{bmatrix} -0.269 \times 2.0 & 1.762 \times (-1.0) \end{bmatrix} = \begin{bmatrix} -0.538 & -1.762 \end{bmatrix}
$$

<p><input type="checkbox" checked disabled> <strong>第 5 步：降维投影回模型主空间（Down-Projection）</strong><br>
将调制后的高维特征映射回 $d_{\text{model}}$ 维度：</p>

$$
\mathbf{y} = \mathbf{h}\mathbf{W}_{\text{down}} = \begin{bmatrix} -0.538(1) + (-1.762)(1) & -0.538(2) + (-1.762)(0) \end{bmatrix} = \begin{bmatrix} -2.300 & -1.076 \end{bmatrix}
$$

</fieldset>

SwiGLU 前馈模块的最终输出结果为：$\mathbf{y} = \begin{bmatrix} -2.300 & -1.076 \end{bmatrix}$。

观察其中的智能逻辑：第 1 个特征由于门控值为负值而被大幅收窄压制（仅留下 $-0.269$），而第 2 个特征却获得了大幅度的开放通行权（放行倍率为 $1.762$）。模型自主决定了哪些信息应当放大，哪些应当退火静默！

---

<h2 id="step-6">第 6 步：全章核心精髓与记忆锚点</h2>

<fieldset>
<legend><strong>全景总结黄金法则</strong></legend>
线性矩阵乘法只能对平直空间进行旋转和缩放；<strong>非线性激活函数是深度神经网络能够折叠高维空间、雕琢复杂决策曲面的唯一源泉</strong>。现代大语言模型采用的 <strong>SwiGLU</strong> 机制，更让被动的空间变换升华为了主动的双线性动态门控，让每一个词元都能随心所欲地驾驭与过滤上下文思想。
</fieldset>

---

<nav aria-label="章节导航">
  <p>
    <a href="../03-matrix-multiplication/index.html">&larr; 第 03 章：神奇拉伸盒（矩阵乘法）</a> &bull;
    <a href="../index.html">课程主页</a> &bull;
    <a href="../04b-lab-micro-brain/index.html">阶段实战工坊 01：80 行纯 Python 训练首个自研大脑（Bengio 2003） &rarr;</a>
  </p>
</nav>
