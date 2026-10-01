# 第 29 章：语言建模的马尔可夫决策过程与策略梯度（REINFORCE 与方差缩减）

---

## 步骤 1：3 岁小孩直觉（人行道粉笔线、旋钮晃动与蒙眼射箭手）

让我们从数学最原初的起点出发：数石头与地面上的一道粉笔线：

1. **人行道上的粉笔线（数轴）**：
   - 拿出一支粉笔在地上画一条笔直的线，在你站立的地方画一个记号：**$0$**。
   - 往前走一步是 $+1$，走两步是 $+2$，一路延伸到 $+100$（奖励得分）；往后退一步是 $-1$，退两步是 $-2$（惩罚扣分）。
   - 在强化学习中，我们的终极目标极度简单朴实：**让机器的平均得分，在这条粉笔线上一步步往右走得更远。**

2. **带旋钮的木盒与装满弹珠的玻璃罐**：
   - 语言模型就是一个正面装满了几亿个微调旋钮（参数 $\boldsymbol{\theta}$）的机器，内部放着一个装有 10 万颗词元弹珠的玻璃罐。
   - 当你晃动一个旋钮，罐子里各种单词弹珠的配比就会发生改变。“梯度”（$\nabla$）无非就是在问：*如果我把这个旋钮往右轻轻拧一毫米，罐子里的中奖弹珠会多掉出来几颗？*

3. **手把手教的幼儿园老师 vs. 蒙眼射箭手**：
   - 在**监督学习（SFT）**中（第 00 章到第 18 章），老师一直站在你身后握着你的手，直接把箭头对准靶心。每一次出手，你都能看到唯一正确的人类参考答案。
   - 在**强化学习（RL）**中，老师走出了房间。你被蒙上黑布带到一片大雾草地上，手里拿着弓箭，完全看不见靶子。
   - 你只能凭着手感把弓拉满，把箭射向大雾中，然后松开手指。大模型生成文本就像在迷雾中接连射箭 &mdash; 依次从罐中摸出一颗颗弹珠。

4. **教练的大喇叭（外部黑盒奖励）**：
   - 当弓箭啪的一声落地，远处隐藏的裁判会拿起大喇叭喊出一个总分：*“正中红心！100 分！”* 或 *“脱靶偏了十米！0 分！”*
   - 裁判**绝不会**走过来教你如何微调肌肉（他不会说*“把左手肘抬高两厘米”*）。他给你的只有一个冷冰冰的最终总分。
   - 你必须完全依靠自己去琢磨：刚才到底哪几个微小的肌肉颤动促成了这记高分？

5. **记分员的历史小本子（平均基线 Baseline）**：
   - 如果裁判喊出 *“50 分！”*，这一箭到底算好算坏？如果不知道平时水平，你根本无从判断！
   - 如果你平时的历史平均成绩只有 10 分，那么 50 分简直是惊天奇迹（在数轴上往前跨越了 $+40$ 步）！你必须赶紧记住刚才的肌肉发力。
   - 但如果你平时随手一射都是 90 分，那么 50 分就是严重失误（在数轴上后退了 $-40$ 步）！
   - 通过**从当前得分中减去历史平均分**，你就能把所有尝试以 $0$ 为中心对齐：差的尝试变成向后退的负向调整，优秀的尝试变成向前推的正向强化。

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

在标准的监督微调（<abbr title="Supervised Fine-Tuning">SFT</abbr>）中，每个训练样本都附带人类专家提供的唯一参考答案 $y^*$，损失函数是标准的交叉熵：

$$
\mathcal{L}_{\text{SFT}}(\boldsymbol{\theta}) = -\log \pi_{\boldsymbol{\theta}}(y^* \mid x)
$$

然而，当让大模型去自主解决复杂任务（例如写一段 100 行的 Python 代码、求解多步数学几何证明、进行开放式逻辑推理）时，微积分的反向传播链条遭遇了三重物理阻断：

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 29.1：</strong> 阻止标准反向传播直接应用于自主文本生成的三大数学壁垒。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="22%">数学壁垒</th>
      <th align="left" width="38%">微积分断裂的根本原因</th>
      <th align="left" width="40%">大语言模型具体体现</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. 离散采样不可导</strong></td>
      <td>从概率分布中选取词元（$y_t \sim \operatorname{Categorical}(\mathbf{p}_t)$）是阶跃式的离散操作。离散选择对参数的导数 $\frac{\partial \text{token}}{\partial \mathbf{W}}$ <strong>处处不存在（或处处为 0）</strong>。</td>
      <td>模型选中了词元编号 <kbd>4821</kbd>（"def"）而不是 <kbd>102</kbd>（"class"），这是一种跳变，不是光滑曲面。</td>
    </tr>
    <tr>
      <td><strong>2. 外部环境是黑盒判别器</strong></td>
      <td>评估代码或解题正确性的外部环境（Python 解释器、单元测试、数学验算器或人类评分员）不是神经网络。你根本无法对代码中的 `if/else` 或终端报错信息反向传导梯度向量！</td>
      <td>终端运行 `pytest` 给出全部通过（$R=1$）或崩溃报错（$R=0$），终端命令无法向模型回传反向导数。</td>
    </tr>
    <tr>
      <td><strong>3. 缺乏唯一标准答案</strong></td>
      <td>一道复杂的编程或数学题可能有成千上万种完全正确的解答路径。如果强行用 SFT 去拟合某个人类写的步骤，模型反而会惩罚自己发现的更简短、更新颖的解法。</td>
      <td>可以用代数法、几何法、反证法。SFT 强行要求逐字模仿，限制了模型的自主解题潜能！</td>
    </tr>
  </tbody>
</table>

<br>

这就是承前启后的核心难题：
$$\text{当动作是离散符号且奖励函数是不可导的黑盒时，我们如何在数学上精确求取期望收益关于模型权重的梯度 } \nabla_{\boldsymbol{\theta}} \mathbb{E}[R] \text{？}$$

---

## 步骤 3：严谨数学公式与推导

### 1. 从纸带与玻璃弹珠谈起：文本自回归的极简本质（破除马尔可夫神话）

在传统教科书中，很多作者喜欢用一堆枯燥抽象的学术定义震慑初学者：*“设自回归文本生成为一个离散时间有限步长的马尔可夫决策过程四元组 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$……”*

让我们彻底丢掉这套空洞的学术行话，回归书桌前正在发生的真实物理过程：

1. **不断延长的纸带（状态 $s_t$）**：
   - 想象一台打字机正在吐出一条长长的纸带。
   - 一开始，纸带上印着用户输入的提示词 $x$（例如：<samp>“请问 2 + 2 等于几？”</samp>）。
   - 模型每产生一个词，就在纸带末尾打印出来。
   - 在第 $t$ 步时，所谓的状态 $s_t$ 根本不需要什么高深的拓扑空间，它**就是此刻纸带上印出的全部文字**：
     $$
     s_t = (x, y_1, y_2, \dots, y_{t-1}) = (x, y_{\lt t})
     $$

2. **装满词元的玻璃罐（动作 $a_t$）**：
   - 机器内部有一个巨大的玻璃罐，里面装有 $|\mathcal{V}|$ 颗弹珠（词表大小，约 $32{,}000 \sim 128{,}000$ 个词）。
   - 所谓的“采取动作” $a_t$，无非就是**伸手从罐子里摸出一颗弹珠 $y_t$**，印在纸带末尾。

3. **破除“马尔可夫性”的神秘感（为什么不需要记住整个过去？）**：
   - 物理学中所谓的*马尔可夫性*，用大白话讲就是：**下一步发生什么，只取决于此刻眼前的状态，与过去是怎么一步步走过来的无关**。
   - 很多同学会困惑：“大模型生成下一个词明明极度依赖上文，怎么可能是马尔可夫的？”
   - 请低头看看你桌上的纸带：**过去的每一个词，此刻不都已经清清楚楚地印在纸带上了吗！**
   - 因为纸带 $s_t$ 已经原原本本地记录了完整的历史，模型只需要看着眼前的纸带即可，根本不需要额外设计任何时空回溯机制。条件概率完全由 $s_t$ 决定，自回归生成极其纯粹地满足马尔可夫性。

4. **一整句话的概率（相乘的弹珠）**：
   - 模型内部有数以亿计的旋钮 $\boldsymbol{\theta}$。给定纸带 $s_t$，机器通过 Softmax 计算罐中每颗弹珠的抽取概率：
     $$
     \pi_{\boldsymbol{\theta}}(y_t \mid s_t) = \frac{\exp(z_{t, y_t})}{\sum_{v \in \mathcal{V}} \exp(z_{t, v})}
     $$
   - 当模型写完一整句话 $\tau = (y_1, y_2, \dots, y_T)$，抽中这串弹珠的联合概率是多少？
   - 依据最朴素的计数与乘法原理，就是把每一步摸出那颗弹珠的概率连乘起来：
     $$
     P(\tau; \boldsymbol{\theta}) = \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(y_t \mid s_t)
     $$
   - 句子生成完毕后，外部裁判（Python 解释器、数学验算器或人类评估员）在数轴上给出唯一的总得分 $R(\tau) \in \mathbb{R}$。

---

### 2. 期望求导悖论与对数导数技巧（Score Function Trick）

强化学习的总体优化目标是最大化在所有可能生成轨迹上的期望收益：

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}[R(\tau)] = \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

<fieldset markdown="1">
<legend><strong>微积分的核心困境：参数 $\boldsymbol{\theta}$ 到底在哪里？</strong></legend>

仔细对比监督学习与强化学习在微积分结构上的根本区别：
- 在普通监督学习中：$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{x \sim \mathcal{D}} [f_{\boldsymbol{\theta}}(x)]$。数据集 $\mathcal{D}$ 是固定的静态语料，参数 $\boldsymbol{\theta}$ 位于函数内部。求导算子可以直接穿透期望符号：$\nabla_{\boldsymbol{\theta}} \mathbb{E}[f_{\boldsymbol{\theta}}] = \mathbb{E}[\nabla_{\boldsymbol{\theta}} f_{\boldsymbol{\theta}}]$。
- 但在强化学习中：奖励函数 $R(\tau)$ 是外部裁判（如 Python 单元测试）。**$R(\tau)$ 的计算逻辑里根本不包含模型权重 $\boldsymbol{\theta}$！**
- 相反，参数 $\boldsymbol{\theta}$ 隐藏在期望的*下标*中 &mdash; 它决定的是**采样概率分布 $P(\tau; \boldsymbol{\theta})$**！
</fieldset>

对目标函数求梯度：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \sum_{\tau} P(\tau; \boldsymbol{\theta}) R(\tau) = \sum_{\tau} \nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) R(\tau)
$$

现在我们面临极大的计算障碍：
1. **千亿文本的组合爆炸**：求和符号 $\sum_{\tau}$ 需要遍历所有可能的句子。哪怕生成仅 500 个词、词表大小为 100,000，可能的句子总数也有 $100{,}000^{500} = 10^{2500}$ 种，远超全宇宙原子总数（$10^{80}$）！没有任何计算机能直接遍历。
2. **必须转回期望形式**：在机器学习中，当无法遍历整个宇宙时，唯一的出路是把它写成期望形式 $\sum_u P(u) g(u) = \mathbb{E}_{u \sim P}[g(u)]$，然后利用 **蒙特卡洛随机采样**（在 GPU 上采样几段文本求平均）来无偏估计它。
3. **关键阻碍**：当前求和式里是 $\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})$，它**不是**一个合法的概率分布（导数有正有负，且加和不为 1）。我们无法从 $\nabla P$ 中抽样！

为了解决这个难题，我们通过三步代数变换完成著名的**对数导数技巧（Log-Derivative Trick / 分数函数估计器）**：

#### 步骤 A：同乘以并同除以 $P(\tau; \boldsymbol{\theta})$
在概率大于 0 的前提下，做恒等变形：

$$
\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \cdot \frac{\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})}{P(\tau; \boldsymbol{\theta})}
$$

#### 步骤 B：利用自然对数的导数法则（相对百分比晃动）

回忆微积分基本求导公式 $\frac{d}{dx} \ln f(x) = \frac{f'(x)}{f(x)}$。反向运用该恒等式：

$$
\frac{\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})}{P(\tau; \boldsymbol{\theta})} \equiv \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

<fieldset>
<legend><strong>费曼的秘密：为什么公式里会凭空冒出对数？（相对百分比增长）</strong></legend>
<p>很多初学者会困惑：<em>强化学习推导里为什么会突然跳出一个 $\log$？</em></p>
<p>请看词表中两个极其典型的词：</p>
<ul>
  <li><strong>高频常见词 A（<kbd>“的”</kbd>）</strong>：当前概率为 $P = 0.500$（1000 颗弹珠里占 500 颗）。如果旋钮让它增加了 1 颗弹珠（$\Delta P = +0.001$），概率变成 $0.501$。这只是微不足道的 <strong>$+0.2\%$</strong> 相对变化 &mdash; 几乎可以忽略不计的噪声。</li>
  <li><strong>低频推理关键词 B（<kbd>“勾股定理”</kbd>）</strong>：当前概率仅为 $P = 0.001$（1000 颗弹珠里只有 1 颗）。如果旋钮同样让它增加 1 颗弹珠（$\Delta P = +0.001$），概率直接翻倍成 $0.002$！这是惊人的 <strong>$+100\%$</strong> 暴增 &mdash; 模型迎来了关键的逻辑突破！</li>
</ul>
<p>在粉笔数轴上，这两个词获得的绝对增量完全一样（都是 $+0.001$）。但在真实的推理世界中，让一个罕见推理词概率翻倍，其意义远超给常见虚词增加一丝噪音。什么数学工具能天然衡量这种“相对百分比影响”？</p>
<p>正是<strong>相对百分比变化率</strong>：</p>
$$
\frac{\text{概率的变化量}}{\text{原本的概率}} = \frac{\nabla_{\boldsymbol{\theta}} P}{P} \equiv \nabla_{\boldsymbol{\theta}} \log P
$$
<p><strong>自然对数绝不是数学家为了显得高深而人为拼凑的，它本身就是“相对百分比增速”在微积分中的天然物理化身！</strong></p>
</fieldset>

由此得到核心恒等式：

$$
\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta}) = P(\tau; \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

#### 步骤 C：重构期望形式
将恒等式代回梯度式中：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \sum_{\tau} P(\tau; \boldsymbol{\theta}) \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right] \\
&= \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) R(\tau) \right]
\end{aligned}
$$

因为 $P(\tau; \boldsymbol{\theta})$ 重新回到了最外层，式子成功变回了一个标准的数学期望！现在我们完全可以在显卡上用自己的大模型生成若干候选文本，直接计算均值来估计梯度！

进一步展开整条轨迹的对数概率：

$$
\log P(\tau; \boldsymbol{\theta}) = \log \left( \prod_{t=1}^T \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right) = \sum_{t=1}^T \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

求导与求和交换顺序：

$$
\nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta}) = \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t)
$$

由此我们得到了著名的**策略梯度定理（REINFORCE 形式）**：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) R(\tau) \right]
$$

<fieldset markdown="1">
<legend><strong>大一统视角：策略梯度就是自产文本上的加权监督微调！</strong></legend>

把策略梯度的参数更新公式与第 15 章的监督微调（SFT）放在一起对比：

<table border="1" cellpadding="6" cellspacing="0" width="100%">
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="28%">学习范式</th>
      <th align="left" width="42%">参数梯度更新公式</th>
      <th align="left" width="30%">物理行为本质</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>监督微调（SFT）</strong></td>
      <td>$\Delta \boldsymbol{\theta} \propto + \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(y_t^* \mid s_t)$</td>
      <td>以固定的权重 $+1$，把模型参数拉向人类专家给出的标准答案 $y^*$。</td>
    </tr>
    <tr>
      <td><strong>策略梯度（RL）</strong></td>
      <td>$\Delta \boldsymbol{\theta} \propto + \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \cdot \mathbf{R(\tau)}$</td>
      <td>把模型参数拉向<em>模型自己生成的动作 $a_t$</em>，但拉力的强度由奖励值 $R(\tau)$ 动态决定！</td>
    </tr>
  </tbody>
</table>

如果 $R(\tau) = +10$，模型就会以常规 SFT 10 倍的力度强化刚才自己说过的词！如果 $R(\tau) = 0$，什么都不做。如果 $R(\tau) < 0$，模型就会反向排斥那些词！

**强化学习并没有发明一套全新的梯度运算系统，它本质上就是拿模型自己探索出来的文本做加权交叉熵，而奖励值就是那个动态调节音量大小的旋钮！**
</fieldset>

---

### 3. 基线消除方差定理（Baseline Subtraction）

虽然原始 REINFORCE 在数学上推导严谨，但在实际工程中却极难收敛，根本原因在于其**灾难性的采样方差**。

<fieldset markdown="1">
<legend><strong>全正数奖励带来的灾难</strong></legend>

假设大模型在做数学题，奖励分值范围在 0 到 100 分之间。模型尝试了 3 种解法：
- 解法 1 得分平平：$R = 90$
- 解法 2 表现不错：$R = 95$
- 解法 3 完美解出：$R = 100$

如果没有基线，看看 REINFORCE 会做什么：它会用 $+90$、$+95$ 和 $+100$ 的权重同时更新网络参数！**这三种解法的概率全都被大幅推高了！** 即使是组内最差的解法 1，其不良解题习惯也会被强烈鼓励。模型根本无法鲜明地意识到解法 3 才是胜利者。

现在我们引入一个平均基线 $b = 95$（组内平均分）：
- 解法 1 得到的优势：$A_1 = 90 - 95 = \mathbf{-5}$（受到惩罚！）
- 解法 2 得到的优势：$A_2 = 95 - 95 = \mathbf{0}$（不奖不罚）
- 解法 3 得到的优势：$A_3 = 100 - 95 = \mathbf{+5}$（受到正向奖励！）

学习信号瞬间变得黑白分明、高度居中！优秀的做法被强化，落后的做法被抑制，梯度的震荡方差急剧下降。
</fieldset>

为此，我们在公式中引入一个只依赖于状态 $s_t$、与具体动作 $a_t$ 无关的**状态基线** $b(s_t)$：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( R(\tau) - b(s_t) \right) \right]
$$

#### 零偏差（Zero Bias）严格逐步数学证明：
我们必须证明：从奖励中减去 $b(s_t)$ 绝不会歪曲真实的梯度方向。
考察在时间步 $t$ 对动作取期望的基线项：

$$
\begin{aligned}
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] &= \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( \frac{\nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)} \right) b(s_t) && \text{［按期望定义展开并代入对数导数］} \\
&= b(s_t) \sum_{a_t \in \mathcal{V}} \nabla_{\boldsymbol{\theta}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) && \text{［消去分子分母中的 } \pi \text{ 并提纯 } b(s_t)\text{］} \\
&= b(s_t) \nabla_{\boldsymbol{\theta}} \left( \sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \right) && \text{［有限项和的求导等于求导的和］}
\end{aligned}
$$

观察括号内部的项：$\sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t)$。
由于 Softmax 输出的概率分布在整个词表上的累加和恒等于 1：

$$
\sum_{a_t \in \mathcal{V}} \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \equiv 1.0
$$

常数的导数严格为零：$\nabla_{\boldsymbol{\theta}}(1.0) = \mathbf{0}$。因此：

$$
\mathbb{E}_{a_t \sim \pi_{\boldsymbol{\theta}}} \left[ \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) b(s_t) \right] = b(s_t) \cdot \mathbf{0} \equiv \mathbf{0}
$$

减去任意与当前动作无关的基线 $b(s_t)$ 对梯度期望的影响**严格为零偏差**！数学期望 100% 保持精准，但梯度的采样方差却得到了数量级级别的缩减！

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
