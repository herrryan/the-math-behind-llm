# 第 29 章：语言建模的马尔可夫决策过程与策略梯度（REINFORCE 与方差缩减）

---

## 步骤 1：3 岁小孩直觉（蒙眼射箭手与耳语教练）

想象你站在一片白茫茫的大雾草地上，眼睛上蒙着厚厚的黑布，手里握着一把玩具弓箭：

1. **蒙眼射箭手（语言模型自身）**：
   - 你根本看不见远处的靶子在哪里。
   - 你只能凭着手感把弓拉满，把箭射向迷雾中。
   - 当大模型生成文本时，它就像是在迷雾中接连射出一支支箭 &mdash; 根据自己内心的感觉（模型内部的词表概率分布）依次挑选下一个词元。

2. **教练的大喇叭（外部奖励评分）**：
   - 当弓箭啪的一声落入草地，远处隐藏的裁判会拿起大喇叭喊出一个数字：
     - *“正中红心！100 分！”*
     - 或者 *“脱靶偏了十米！0 分！”*
   - 请注意最关键的一点：裁判**绝不会**走过来教你如何微调肌肉动作，他不会说*“把你的左手肘抬高两厘米”*。他给你的只有一个冷冰冰的最终总分。

3. **记分员的历史小本子（平均基线 Baseline）**：
   - 如果裁判喊出 *“50 分！”*，这一箭到底算好还是算坏？如果你不知道自己平时的平均水平，你根本无从判断！
   - 如果你平时的历史平均成绩只有 10 分，那么 50 分简直是惊天奇迹！你必须赶紧记住刚才射箭时手臂的酸麻感与角度。
   - 但如果你平时随手一射都是 90 分，那么 50 分就是严重失误！你必须马上避免刚才的肌肉发力方式。
   - 通过**从当前得分中减去历史平均分**，你就能确保：只有当某一箭*真正超越了日常预期*时，才去正向强化身体的肌肉记忆。

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                     强化学习策略梯度在语言模型中的循环                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   输入提示 [x] ──► [ 模型策略 π_θ ] ──► 生成词元序列 [y_1, ..., y_T]   │
│                          ▲                               │             │
│                          │                               ▼             │
│                     参数梯度更新                   任务环境 /          │
│                  Δθ ∝ ∇ log π_θ · (R - b)          奖励函数评估        │
│                          ▲                               │             │
│                          │                               ▼             │
│                     超预期优势 ◄───────── 标量奖励值 R(τ)              │
│                   A = R(τ) - Baseline b                                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 29.1：</strong> 强化学习策略梯度数据流：语言模型作为策略生成动作序列，获得轨迹奖励，根据超出基线的优势信号更新模型参数。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

在第 16 章中，我们学习了反向传播算法：通过多元微积分链式法则直接对连续损失函数求导更新权重：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{W}}
$$

然而，在将大语言模型投入真实任务强化学习（如写 Python 代码、做多步数学推导、回答复杂指令）时，反向传播链条遭遇了物理阻断：
1. **采样过程不可导（离散性鸿沟）**：模型将 logits 转化为离散词元时，使用的是离散多项分布采样（$y_t \sim \operatorname{Categorical}(\mathbf{p}_t)$）或 $\operatorname{argmax}$ 贪心选取。离散选择的导数根本不存在：$\frac{\partial \text{token}}{\partial \mathbf{W}}$ 在数学上无意义！
2. **外部环境是黑盒判别器**：一个 Python 编译器或人类评审员在评价输出时，只能给出离散评判：要么单元测试全部通过（$R=1$），要么报错崩溃（$R=0$）。我们根本无法将梯度反向穿透一个外部编译器。

这就是承前启后的核心难题：
$$\text{当动作是离散符号且奖励函数是不可导的黑盒时，我们如何在数学上精确求取期望收益关于模型权重的梯度 } \nabla_{\boldsymbol{\theta}} \mathbb{E}[R] \text{？}$$

---

## 步骤 3：严谨数学公式与推导

### 1. 将语言自回归生成建模为马尔可夫决策过程（MDP）

我们将自回归文本序列生成严格建模为离散时间、有限步长马尔可夫决策过程 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$：

- **状态空间（$\mathcal{S}$）**：在生成第 $t$ 步时，状态 $s_t$ 由原始 Prompt $x$ 与截至当前生成的所有历史词元组成：
  $$
  s_t = (x, y_1, y_2, \dots, y_{t-1}) = (x, y_{\lt t}) \in \mathcal{S}
  $$
- **动作空间（$\mathcal{A}$）**：动作 $a_t$ 即为从离散词表 $\mathcal{V}$ 中挑选中下一个词元：
  $$
  a_t = y_t \in \mathcal{V} \quad (|\mathcal{V}| \approx 32{,}000 \sim 128{,}000)
  $$
- **状态转移函数（$\mathcal{P}$）**：在语言生成中，状态转移是确定性的字符串追加：
  $$
  s_{t+1} = [s_t, a_t] = (x, y_1, \dots, y_t)
  $$
  其确定性转移概率为 $\mathcal{P}(s_{t+1} \mid s_t, a_t) = 1$。
- **策略网络（$\pi_{\boldsymbol{\theta}}$）**：参数为 $\boldsymbol{\theta} \in \mathbb{R}^D$ 的语言模型在给定状态 $s_t$ 下输出词表上的多项式分布：
  $$
  \pi_{\boldsymbol{\theta}}(a_t \mid s_t) = \operatorname{softmax}\left(\mathbf{z}_t\right)_{a_t} = \frac{\exp\left(z_{t, a_t}\right)}{\sum_{v \in \mathcal{V}} \exp\left(z_{t, v}\right)}
  $$
- **完整轨迹（$\tau$）与累积收益（$R(\tau)$）**：生成一段完整文本构成一条轨迹 $\tau = (s_1, a_1, \dots, s_T, a_T)$。在文本终结（输出结束符 EOS）时，环境给出一个标量总奖励 $R(\tau) \in \mathbb{R}$。

在参数 $\boldsymbol{\theta}$ 控制下，生成特定完整轨迹 $\tau$ 的联合概率为：

$$
P(\tau; \boldsymbol{\theta}) = P(s_1) \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t) = \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

---

### 2. 期望目标与对数导数技巧（Score Function Trick）

强化学习的总体优化目标是最大化在所有可能生成轨迹上的期望收益：

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[R(\tau)] = \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

我们对模型权重参数 $\boldsymbol{\theta}$ 求梯度：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau) = \sum_{\tau} \nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

请注意：这个式子不能直接用蒙特卡洛随机采样来估计，因为 $\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})$ 本身不是概率分布！

为了把它重新变回数学期望形式，我们引入最为核心的**对数导数技巧（Log-Derivative Trick / 分数函数估计器）**。
由微积分基本公式 $\frac{d}{dx} \ln f(x) = \frac{f'(x)}{f(x)}$ 可知：

$$
\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

将该等式代回梯度公式中：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{\tau} P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \\
&= \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right]
\end{aligned}
$$

展开轨迹对数概率 $\log P(\tau; \boldsymbol{\theta})$：

$$
\log P(\tau; \boldsymbol{\theta}) = \log \left( \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right) = \sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

由于求导与求和可以交换顺序，连乘积神奇地化解为了各个单步词元对数概率梯度的累加：

$$
\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) = \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

由此我们得到了著名的**策略梯度定理（REINFORCE 形式）**：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) R(\tau) \right]
$$

<fieldset markdown="1">
<legend><strong>为什么这个公式是一座数学奇迹？</strong></legend>

仔细观察这个公式：原本对完全不可导的黑盒奖励期望 $\mathbb{E}[R]$ 求梯度，被完美转化成了对模型自己生成的词元计算 $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}$ 并乘以标量奖励值 $R(\tau)$！

项 $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$ 与第 15 章交叉熵损失函数关于模型参数的梯度形式完全一致。唯一区别是：监督微调是无条件把概率推向固定的人类参考答案，而强化学习则是把梯度方向按照环境给出的奖励大小进行动态缩放！
</fieldset>

---

### 3. 基线消除方差定理（Baseline Subtraction）

朴素 REINFORCE 估计器存在致命缺陷：**采样方差极大**。
如果所有采样的奖励值都是正数（例如得分都在 50 到 100 分之间），那么*每一个采样的输出其概率都会被正向推高*，导致糟糕的输出仅仅是因为得到了 50 分也被模型盲目鼓励。

为了解决这个问题，我们在公式中引入一个与当前动作无关的**状态基线** $b(s_t)$：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( R(\tau) - b(s_t) \right) \right]
$$

#### 严格数学证明：减去基线带来零偏差（Zero Bias）
我们需要证明：引入 $b(s_t)$ 绝不会改变梯度的数学期望。
考察在时间步 $t$ 对动作取期望的内部项：

$$
\begin{aligned}
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] &= \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)} b(s_t) \\
&= b(s_t) \sum_{a_t \in \mathcal{V}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \\
&= b(s_t) \nabla_{\boldsymbol{\theta}} \left( \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right)
\end{aligned}
$$

由于概率在整个词表上的累加和恒等于 1：

$$
\sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \equiv 1 \implies \nabla_{\boldsymbol{\theta}}(1) = \mathbf{0}
$$

因此：

$$
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] = b(s_t) \cdot \mathbf{0} = \mathbf{0}
$$

减去基线在数学上**严格保持无偏**，但它将原本均为正数的奖励中心化，大幅削减了在数万维词表上采样的方差 $\operatorname{Var}(\hat{\mathbf{g}})$！

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="1992">1992年</time> &mdash; <strong>Ronald J. Williams（REINFORCE 诞生）</strong></dt>
  <dd>Williams 提出著名的 <em>REINFORCE</em> 算法（<cite>《Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning》</cite>），首次为人工神经网络推导了分数函数梯度估计与基线消除方差原理。</dd>

  <dt><time datetime="1999">1999年</time> &mdash; <strong>Richard Sutton、David McAllester、Satinder Singh、Yishay Mansour</strong></dt>
  <dd>形式化证明了通用<em>策略梯度定理（Policy Gradient Theorem）</em>，证明在任意未知环境状态转移分布下，均无需计算状态分布对权重的导数 $\nabla_{\boldsymbol{\theta}} d^{\pi}(s)$，即可准确计算策略梯度。</dd>

  <dt><time datetime="2016">2016–2020年</time> &mdash; <strong>大模型强化学习的早期危机</strong></dt>
  <dd>在机器翻译等任务中，直接用 REINFORCE 优化 BLEU 指标经常遭遇训练崩溃、输出退化和模式坍缩。模型极易学会输出全标点符号或刷分套路。这直接倒逼学术界引入密集 Actor-Critic 评论家机制（第 30 章）以及现代基于群组统计的归一化方法（第 31 章）。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

让我们用极小规模的词表与参数，完整手算一遍策略梯度更新。

### 极简玩具设定
- 词表：$\mathcal{V} = \{A, B, C\}$（大小 $|\mathcal{V}| = 3$）。
- 单步动作：$a \in \{A, B, C\}$。
- 当前模型的未归一化 logits：
  $$
  \mathbf{z} = [1.0, \; 0.0, \; -1.0]^\top
  $$
- 学习率 $\eta = 0.1$。

---

### 第一阶段：前向推断与 Softmax 概率分布

计算指数项：
- $e^{z_A} = e^{1.0} \approx 2.718$
- $e^{z_B} = e^{0.0} = 1.000$
- $e^{z_C} = e^{-1.0} \approx 0.368$
- 分母和：$\sum e^{z_i} \approx 2.718 + 1.000 + 0.368 = 4.086$

各项概率 $\pi_{\boldsymbol{\theta}}(a)$：
$$
\pi(A) = \frac{2.718}{4.086} \approx 0.665 \quad (<meter min="0" max="1" value="0.665"></meter>)
$$
$$
\pi(B) = \frac{1.000}{4.086} \approx 0.245 \quad (<meter min="0" max="1" value="0.245"></meter>)
$$
$$
\pi(C) = \frac{0.368}{4.086} \approx 0.090 \quad (<meter min="0" max="1" value="0.090"></meter>)
$$

---

### 第二阶段：采样动作与环境反馈

假设模型的随机采样机制抽中了词元 <kbd>"B"</kbd>（$a = B$）：
- 外部环境评估词元 <kbd>"B"</kbd> 并返回得分：
  $$
  R = 2.0
  $$
- 此时历史记录的平均基准线为：
  $$
  b = 0.5
  $$
- 计算**超预期优势（Advantage）**：
  $$
  A = R - b = 2.0 - 0.5 = +1.5
  $$
  由于 $A > 0$，说明词元 <kbd>"B"</kbd> 远超历史平均水平！我们希望大幅提升选它的概率。

---

### 第三阶段：对数概率梯度精确计算

对于多项式 Softmax 分布，当选中动作 $a$ 时，对数概率关于 logit $z_i$ 的导数为：

$$
\frac{\partial \log \pi(a)}{\partial z_i} = \mathbb{I}(i = a) - \pi(i)
$$

对于选中的 $a = B$：
- 针对词元 $A$：$\frac{\partial \log \pi(B)}{\partial z_A} = 0 - \pi(A) = -0.665$
- 针对词元 $B$：$\frac{\partial \log \pi(B)}{\partial z_B} = 1 - \pi(B) = 1 - 0.245 = +0.755$
- 针对词元 $C$：$\frac{\partial \log \pi(B)}{\partial z_C} = 0 - \pi(C) = -0.090$

验证梯度分量和：$-0.665 + 0.755 - 0.090 = 0.000$（严格零和！）。

---

### 第四阶段：策略梯度参数更新

带基线修正的策略梯度向量为：

$$
\mathbf{g} = \nabla_{\mathbf{z}} \log \pi(B) \cdot (R - b) = \begin{bmatrix} -0.665 \\ +0.755 \\ -0.090 \end{bmatrix} \times 1.5 = \begin{bmatrix} -0.9975 \\ +1.1325 \\ -0.1350 \end{bmatrix}
$$

以步长 $\eta = 0.1$ 执行梯度上升（增加收益）：

$$
\mathbf{z}_{\text{新}} = \mathbf{z} + \eta \mathbf{g} = \begin{bmatrix} 1.0 \\ 0.0 \\ -1.0 \end{bmatrix} + 0.1 \begin{bmatrix} -0.9975 \\ +1.1325 \\ -0.1350 \end{bmatrix} = \begin{bmatrix} 0.90025 \\ 0.11325 \\ -1.01350 \end{bmatrix}
$$

检验更新后模型在 $\mathbf{z}_{\text{新}}$ 下的新概率分布：
- $e^{0.90025} \approx 2.460$
- $e^{0.11325} \approx 1.120$
- $e^{-1.01350} \approx 0.363$
- 新分母和 $\approx 3.943$

新概率：
- $\pi_{\text{新}}(A) = \frac{2.460}{3.943} \approx 0.624$（从 $0.665$ 下降）
- $\pi_{\text{新}}(B) = \frac{1.120}{3.943} \approx \mathbf{0.284}$（<mark>从 $0.245$ 显著跃升！</mark>）
- $\pi_{\text{新}}(C) = \frac{0.363}{3.943} \approx 0.092$

动作 <kbd>"B"</kbd> 的概率瞬间增加了 $+3.9\%$，因为其超额回报（$+1.5$）赋予了极强的正向激励！

---

## 步骤 6：核心精髓总结

<fieldset markdown="1">
<legend><strong>核心精髓总结</strong></legend>

策略梯度定理通过巧妙的**对数导数技巧**跨越了离散采样的不可导鸿沟：我们完全无需穿透离散词元或外部评测环境求导，而是将模型自带的标准交叉熵梯度乘以外部标量奖励的超额回报。

从奖励中减去状态平均基线在数学上严格保持无偏性，却能将数万维离散词表上的发散方差压缩至可稳定收敛的物理区间。
</fieldset>
