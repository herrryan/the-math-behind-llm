# 第 29 章：语言建模的马尔可夫决策过程与策略梯度（REINFORCE 与方差缩减）

---

## 步骤 1：3 岁小孩直觉（蒙眼射箭手与耳语教练）

想象你在学习射箭，但你之前从来没摸过弓箭：

1. **手把手教的幼儿园老师 vs. 蒙眼射箭手**：
   - 在**监督学习（SFT）**中（就像第 00 章到第 18 章所学），老师一直站在你身后，双手紧紧握着你的手，手把手把你瞄准的箭头直接对准靶心正中。每一次出手，你都能看到唯一正确的人类参考答案。
   - 而在**强化学习（RL）**中，老师走出了房间。你被蒙上一块厚厚的黑布，带到了一片白茫茫的迷雾草地上。你手里握着弓和箭，但完全看不见靶子在哪里。
   - 你只能凭着手感把弓拉满，把箭射向大雾中，然后松开手指。
   - 当大模型生成文本时，它就像是在迷雾中接连射出一支支箭 &mdash; 根据自己当前的内心概率，依次挑选下一个词元。

2. **教练的大喇叭（外部奖励评分）**：
   - 当弓箭啪的一声落入草地，远处隐藏的裁判会拿起大喇叭喊出一个数字：
     - *“正中红心！100 分！”*
     - 或者 *“脱靶偏了十米！0 分！”*
   - 请注意最关键的一点：裁判**绝不会**走过来教你如何微调肌肉动作，他不会说*“把你的左手肘抬高两厘米”*或*“把弓弦拉得更紧一些”*。他给你的只有一个冷冰冰的最终总分。
   - 你必须完全依靠自己去琢磨：刚才到底哪几个细微的肌肉颤动促成了这记高分？

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

### 1. 将语言自回归生成建模为马尔可夫决策过程（MDP）

为了在数学上严谨研究文本生成，我们将大模型的自回归生成形式化为离散时间、有限步长的<dfn id="def-mdp">马尔可夫决策过程（MDP）</dfn>，记为四元组 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$：

<fieldset markdown="1">
<legend><strong>为什么文本生成可以被视作 MDP？（马尔可夫性的本质）</strong></legend>

物理学与概率论中的*马尔可夫性质（Markov Property）*指的是：**未来只取决于现在的状态，而与过去的到达路径无关**：$P(s_{t+1} \mid s_t, s_{t-1}, \dots, s_0) = P(s_{t+1} \mid s_t)$。

很多初学者会疑惑：语言模型预测下一个词明明极度依赖前面的上文，怎么可能是马尔可夫过程？
关键在于**状态的定义**！我们将时间步 $t$ 的状态 $s_t$ 直接定义为*原始提示词加上目前为止生成的所有历史词元*：$s_t = (x, y_1, y_2, \dots, y_{t-1})$。因为 $s_t$ 已经完全打包了截至目前的所有上下文，所以在给定 $s_t$ 的前提下，下一个词元不再依赖任何外部历史！自回归生成严格满足马尔可夫性。
</fieldset>

- **状态空间（$\mathcal{S}$）**：在生成第 $t$ 步时，状态 $s_t$ 由原始 Prompt $x$ 与截至当前生成的所有历史词元拼接而成：
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
  其确定性转移概率为 $\mathcal{P}(s_{t+1} \mid s_t, a_t) \equiv 1$。
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

#### 步骤 B：利用自然对数的导数法则
回忆微积分基本求导公式 $\frac{d}{dx} \ln f(x) = \frac{f'(x)}{f(x)}$。反向运用该恒等式：

$$
\frac{\nabla_{\boldsymbol{\theta}} P(\tau; \boldsymbol{\theta})}{P(\tau; \boldsymbol{\theta})} \equiv \nabla_{\boldsymbol{\theta}} \log P(\tau; \boldsymbol{\theta})
$$

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
