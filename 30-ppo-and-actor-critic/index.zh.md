# 第 30 章：Actor-Critic 架构与近端策略优化（PPO）

---

## 步骤 1：3 岁小孩直觉（舞台演员、前排导演与弹性安全绳）

想象百老汇剧场里的一场特殊彩排，舞台上下有三个人和一条安全吊带：

1. **舞台演员（生成策略网络 Actor $\pi_{\boldsymbol{\theta}}$）**：
   - 演员站在舞台聚光灯下，面对空无一人的观众席即兴念台词。
   - 他脑子里充满各种点子 &mdash; 随时尝试不同的用词、语调和停顿。

2. **前排导演（价值评估网络 Critic $V_{\boldsymbol{\phi}}$）**：
   - 坐在观众席第三排正中央的是经验丰富的导演，手里拿着记分板。
   - 演员每念完一句话，导演就会在纸上飞快写下一个即时的预期总分：
     - *“就凭现在的表演状态，我预计今晚剧目最终能拿 8 分（满分 10 分）。”*
   - 当演员把整场戏演完，观众进场打分：
     - 如果观众打出了 9 分，说明整场表现比导演预期的还要好 $+1$ 分！这多出来的惊喜就是**超额优势（Advantage）**。
     - 如果观众只打了 5 分，说明表现比预期差了 $-3$ 分，演员当场挨批评。

3. **腰间的弹性安全绳（PPO 概率裁剪机制）**：
   - 如果演员灵机一动讲了个笑话，全场突然爆发出雷鸣般的掌声，会发生什么？
   - 如果没有安全绳，演员可能会兴奋过度，第二天完全抛弃剧本，花两个小时在舞台上连续翻跟头大吼大叫！整场话剧瞬间报废。
   - 为了防止这种悲剧，我们在演员腰上系了一条**强力弹性安全绳**。
   - 安全绳规定：在单次彩排中，演员对任何一句台词的演法变动**最多不能超过 $20\%$**。哪怕某个改动赢得满堂彩，安全绳也会在物理上死死拉住他，绝不允许他一步迈入完全失控的未知深渊。

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                   大模型经典 4 模型 RLHF ACTOR-CRITIC 集群              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   输入提示 [x] ────────────────────┬────────────────────┐              │
│                                    │                    │              │
│                                    ▼                    ▼              │
│                           [ 参考模型 Ref ]      [ 演员策略 Actor ]      │
│                            (π_ref - 冻结)        (π_θ - 训练更新)      │
│                                    │                    │              │
│                                    │                    ▼              │
│                                    │             生成词元 y_t          │
│                                    │                    │              │
│                                    ▼                    ▼              │
│                             KL 散度惩罚约束:                            │
│                             D_KL(π_θ || π_ref)                         │
│                                    │                                   │
│                                    ▼                                   │
│   [ 奖励模型 RM ] ────────► 综合单步奖励 r_t ──► [ 评论家 Critic ]     │
│     (R_ψ - 冻结)              (任务分 + KL)        (V_φ - 训练更新)     │
│                                                         │              │
│                                                         ▼              │
│                               PPO 裁剪更新 ◄────── 广义优势估计        │
│                               min(r·A, clip·A)      (GAE 优势值)       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 30.1：</strong> 经典 4 模型 RLHF 训练闭环：演员生成词元，参考模型防止漂移，奖励模型评判质量，评论家计算 GAE 优势以驱动 PPO 裁剪更新。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

在第 29 章中，我们推导出了经典的 REINFORCE 策略梯度：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau} \left[ \sum_{t=1}^T \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a_t \mid s_t) \left( R(\tau) - b(s_t) \right) \right]
$$

尽管数学原理十分优美，但直接在大语言模型上运行纯粹的 REINFORCE 会遭遇两大灾难性的工程瓶颈：

1. **信用分配滞后瓶颈（第 995 个词元的拼写失误）**：
   - 在长达 1,000 个词元的长篇推导或 Python 代码生成中，模型只有在输出完最终的 `<|endoftext|>` 结束符后，才能拿到唯一的标量总奖励 $R(\tau)$。
   - 假设从第 1 到第 994 个词元展现了极其精妙的数学推理，但在第 995 个词元，模型粗心漏掉了一个闭括号 `)`。Python 解释器报错崩溃，环境返回总分 $R = 0$。
   - 原始 REINFORCE 对该轨迹中的每一个词元一视同仁：它会用完全相同的负向惩罚把前 994 个神来之笔与第 995 个失误词元一同打压！
   - 我们急需一个能够在句子中间步骤评估*局部阶段性进展*的细粒度裁判，即使后续发生失误，也能精准鼓励前面的正确逻辑。

2. **步长悬崖崩溃（策略坍塌）**：
   - 在深度神经网络中，由于 Softmax 函数具有天然的指数放大效应（$\exp(z_i)$），模型权重空间极其微小的更新 $\Delta \boldsymbol{\theta}$ 都可能在输出概率空间引发剧烈震荡。
   - 一次未加约束的梯度更新，可能让某个词元的输出概率在一夜之间从 $0.01$ 暴增到 $0.99$。
   - 一旦语言模型跌入崩溃悬崖，它就会陷入不可逆的退化循环：不断重复输出毫无意义的单词（例如无限循环输出 <samp>"the the the"</samp>），导致后续所有采样全部拿 0 分，梯度彻底消失或爆炸，模型永远无法恢复。

这就是承前启后的关键问题：
$$\text{我们如何利用可学习的价值网络实时估计词元级优势，并在数学上严格限制策略漂移幅度以保证单调稳定收敛？}$$

---

## 步骤 3：严谨数学公式与推导

### 1. 经典 RLHF 的 4 模型系统

在大语言模型经典强化学习对齐中，需要同时协同 4 个独立模型：

1. **演员策略（Actor $\pi_{\boldsymbol{\theta}}$）**：当前负责生成文本的大语言模型，其参数 $\boldsymbol{\theta} \in \mathbb{R}^D$ 持续接受梯度更新以迎合人类偏好与解题正确率。
2. **评论家（Critic / 价值模型 $V_{\boldsymbol{\phi}}$）**：估计从文本生成的任意中间阶段出发，预计最终能拿到的未来总回报期望值。
3. **参考模型（Reference Policy $\pi_{\text{ref}}$）**：初始监督微调（SFT）模型的完全冻结副本，充当防漂移基准锚点。
4. **奖励模型（Reward Model $R_{\boldsymbol{\psi}}$）**：在人类成对偏好数据 $(y_w \succ y_l)$ 上预先训练并冻结的打分模型（第 19 章）。

<fieldset markdown="1">
<legend><strong>代码中评论家网络（Critic）到底长什么样？</strong></legend>

很多初学者好奇：*评论家是一个结构截然不同的神经网络吗？*

不是！评论家物理上就是演员模型（Transformer）的克隆体！两者唯一的结构区别在于**最顶层的输出层**：
- 演员模型使用反嵌入投影矩阵 $\mathbf{W}_{\text{unembed}} \in \mathbb{R}^{d \times |\mathcal{V}|}$，为词表中的 100,000 个候选词元各输出一个打分 logit。
- 评论家模型则将这一层替换为了一个极其简单的单向量线性层 $\mathbf{w}_V \in \mathbb{R}^{d \times 1}$。

给定任意前缀状态 $s_t = (x, y_1, \dots, y_{t-1})$，评论家读取 Transformer 输出的最终隐藏状态向量 $\mathbf{h}_t \in \mathbb{R}^d$，并输出**一个单一标量数值**：

$$
V_{\boldsymbol{\phi}}(s_t) = \mathbf{w}_V^\top \mathbf{h}_t + b_V \in \mathbb{R}
$$

这个数字回答的核心问题是：*“从当前生成的这个词元位置出发，我预计整场对话最终平均能拿多少分？”*
</fieldset>

---

### 2. 带 KL 散度安全绳的词元级奖励

为了防止演员模型钻奖励模型的漏洞（即<dfn id="def-reward-hacking-zh">奖励黑客攻击 / 刷分套路</dfn>），并保护原有的语言通顺度，我们对偏离参考模型的每个词元施加 KL 散度惩罚：

$$
r_t = \begin{cases}
-\beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} \right) & \text{当处于中间词元 } t < T \\
R_{\boldsymbol{\psi}}(x, y) - \beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_T \mid s_T)}{\pi_{\text{ref}}(a_T \mid s_T)} \right) & \text{当处于终止词元 } t = T
\end{cases}
$$

其中 $\beta > 0$ 为 KL 惩罚系数（通常取 $\beta \approx 0.01 \sim 0.1$）。
- 如果演员模型保持在参考模型附近（$\pi_{\boldsymbol{\theta}} \approx \pi_{\text{ref}}$），比率约为 1，$\log(1) = 0$（零惩罚）。
- 如果演员试图钻空子，将某个奇怪用词的概率推得奇高（$\pi_{\boldsymbol{\theta}} \gg \pi_{\text{ref}}$），对数比值大于 0，就会被严厉扣分。
- 这就形成了一条**弹性安全绳**：模型可以自由探索更好的解法，但绝不能彻底丧失人类说话的语法规范。

---

### 3. 时间差分（TD）残差的本质解析

为了评估在状态 $s_t$ 下采纳词元 $a_t$ 到底比预期好多少，我们计算**时间差分（TD）残差** $\delta_t^V$：

$$
\delta_t^V = \underbrace{r_t + \gamma V_{\boldsymbol{\phi}}(s_{t+1})}_{\text{新现实 + 修正后的未来预期}} - \underbrace{V_{\boldsymbol{\phi}}(s_t)}_{\text{旧有的初始预期}}
$$

其中 $\gamma \in (0, 1]$ 为折现因子（在有限步长语言生成中通常设 $\gamma = 1.0$）。

<fieldset markdown="1">
<legend><strong>一个关于 TD 残差的直观数字故事</strong></legend>

假设给模型的输入提示词是：<code>"求解方程 2x + 6 = 14"</code>。

1. **初始状态 $s_0$**：评论家读取题目，预测 $V(s_0) = 0.50$（它判断模型有 50% 的初始解题成功率）。
2. **步骤 1（绝佳推导）**：模型生成词元 <kbd>"两边同时减 6 得：2x = 8"</kbd>。
   - 即时奖励 $r_1 = 0$。
   - 形成新状态 $s_1$。评论家评估当前式子，将胜率预期上调至：$V(s_1) = 0.95$（95% 的把握！）。
   - 计算 TD 惊喜程度：
     $$
     \delta_1 = r_1 + V(s_1) - V(s_0) = 0 + 0.95 - 0.50 = \mathbf{+0.45}
     $$
   - **结果**：仅仅这一个词元，就让成功概率暴涨了 $+0.45$！该词元得到强烈的正向奖赏！
3. **步骤 2（算术失误）**：模型接着生成词元 <kbd>"两边同时除以 2 得：x = 5"</kbd>（出现算术粗心失误！）。
   - 即时奖励 $r_2 = 0$。
   - 形成新状态 $s_2$。评论家敏锐察觉算术错误，胜率预期断崖式暴跌至：$V(s_2) = 0.05$（只剩 5% 希望！）。
   - 计算 TD 惊喜程度：
     $$
     \delta_2 = r_2 + V(s_2) - V(s_1) = 0 + 0.05 - 0.95 = \mathbf{-0.90}
     $$
   - **结果**：该失误词元当场遭到 $-0.90$ 的重罚！

<strong>看清 TD 残差的神奇威力</strong>：即使整场解题最终因算错而宣告失败（最终总分 $R=0$），模型**也不会**惩罚步骤 1！TD 残差精准表彰了步骤 1 的优秀推导（$+0.45$），并将责任精确锁定在发生错误的步骤 2（$-0.90$）。长文本信用分配难题迎刃而解！
</fieldset>

---

### 4. 广义优势估计（GAE）

1 步 TD 残差 $\delta_t^V$ 虽解决了信用分配，但它高度依赖评论家的预估值 $V(s_{t+1})$。若评论家网络本身预测不准，优势估计就会存在**偏差（Bias）**。
反之，若一直等到全文本结束再结算回报（蒙特卡洛回报），虽然绝对无偏，但在长序列上的**方差（Variance）极大**。

为了在两者之间取得最优平衡，<dfn id="def-gae-zh">广义优势估计（GAE-$\lambda$）</dfn>对未来的 TD 残差进行了指数加权求和：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V = \delta_t^V + (\gamma \lambda) \delta_{t+1}^V + (\gamma \lambda)^2 \delta_{t+2}^V + \dots
$$

其紧凑递推形式为：

$$
\hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}
$$

超参数 $\lambda \in [0, 1]$ 充当连续平衡旋钮：
- **$\lambda = 0$（纯 1 步 TD）**：$\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$。方差极小，但若评论家有偏差，梯度更新就会被误导。
- **$\lambda = 1$（纯蒙特卡洛全轨迹回报）**：$\hat{A}_t = \sum_{k=t}^T \gamma^{k-t} r_k - V(s_t)$。无评论家偏差，但长文本累积方差极高。
- **$\lambda = 0.95$（大模型工程黄金标准）**：当天的 TD 惊喜占 $100\%$，明天的占 $95\%$，后天的占 $(0.95)^2 \approx 90\%$，以平滑的几何衰减将局部精细归因与长期全局收益完美融合。

---

### 5. PPO 裁剪替代目标函数（Clipped Surrogate Objective）

定义当前策略 $\pi_{\boldsymbol{\theta}}$ 与采集数据时的旧策略 $\pi_{\boldsymbol{\theta}_{\text{旧}}}$ 在生成词元 $a_t$ 时的重要性采样概率比率：

$$
r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{旧}}}(a_t \mid s_t)} \quad (\text{初始状态下 } r_t(\boldsymbol{\theta}_{\text{旧}}) = 1.0)
$$

PPO 裁剪替代目标函数定义为：

$$
\mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta})\hat{A}_t, \; \operatorname{clip}\left(r_t(\boldsymbol{\theta}), \, 1-\epsilon, \, 1+\epsilon\right)\hat{A}_t \right) \right]
$$

其中 $\epsilon$ 为裁剪阈值（通常设 $\epsilon = 0.2$，将比率严格限制在 $[0.8, 1.2]$ 区间）。

<fieldset markdown="1">
<legend><strong>拆解四大象限：为什么目标函数中要取最小值（$\min$）？</strong></legend>

很多同学疑惑：*为什么公式要在未裁剪项与裁剪项之间取最小值？*

这个 $\min$ 操作构建了一个**悲观下界（Pessimistic Lower Bound）**，确保优化器在面对任何更新时都采取审慎防御姿态：

1. **情况 1：正优势（$\hat{A}_t > 0$，优秀动作）**
   - 我们希望增大这个动作的概率（$r_t > 1$）。
   - 如果 $r_t$ 增长过猛，超过了 $1 + \epsilon$（例如 $r_t = 1.4$，而上限是 $1.2$）：
   - 未裁剪项为 $1.4 \hat{A}_t$，裁剪项为 $1.2 \hat{A}_t$。
   - $\min(1.4 \hat{A}_t, 1.2 \hat{A}_t) = 1.2 \hat{A}_t$。目标函数被封顶截断！导数在此处变为**零**，防止模型过度贪婪地将参数全押在这一个词元上。
2. **情况 2：负优势（$\hat{A}_t < 0$，失误动作）**
   - 我们希望压低这个动作的概率（$r_t < 1$）。
   - *注意负号反转了大小关系！* 因为 $\hat{A}_t$ 是负数，乘以更小的数反而得到**更大的值**（例如 $0.1 \times (-10) = -1.0$，而 $0.8 \times (-10) = -8.0$）。
   - 如果模型过激地打压该词元，使 $r_t$ 暴跌低于 $1 - \epsilon$（例如 $r_t = 0.1$，而下限是 $0.8$）：
   - 未裁剪项为 $-1.0$，裁剪项为 $-8.0$。
   - $\min(-1.0, -8.0) = -8.0$。最小值选择了更悲观的裁剪项！因为 $-8.0$ 对参数 $\boldsymbol{\theta}$ 是水平常数线，其导数严格为**零**！
   - **结果**：一旦某个错误词元已经被充分打压（$r_t < 0.8$），PPO 绝不对其“穷追猛打”！它立即停止惩罚，防止概率下溢崩溃，保留未来的探索空间。
</fieldset>

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 30.1：</strong> PPO 裁剪在不同优势与比率区间下的完整数学响应行为。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left" width="20%">优势值区间</th>
      <th align="left" width="25%">比率 $r_t(\boldsymbol{\theta})$ 状态</th>
      <th align="left" width="25%">实际目标函数值</th>
      <th align="left" width="30%">梯度表现机制</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>正优势（$\hat{A}_t > 0$）</strong><br>该动作表现优于预期</td>
      <td>$r_t(\boldsymbol{\theta}) \le 1 + \epsilon$<br>处于安全步幅区间内</td>
      <td>$r_t(\boldsymbol{\theta}) \hat{A}_t$</td>
      <td><mark>正向有效梯度</mark>：继续提升该词元的生成概率。</td>
    </tr>
    <tr>
      <td><strong>正优势（$\hat{A}_t > 0$）</strong><br>该动作表现优于预期</td>
      <td>$r_t(\boldsymbol{\theta}) > 1 + \epsilon$<br>概率提升已达到上限</td>
      <td>$(1 + \epsilon) \hat{A}_t$</td>
      <td><strong>导数归零</strong>：强制截断梯度，防止过度优化。</td>
    </tr>
    <tr>
      <td><strong>负优势（$\hat{A}_t < 0$）</strong><br>该动作表现低于预期</td>
      <td>$r_t(\boldsymbol{\theta}) \ge 1 - \epsilon$<br>处于温和惩罚区间内</td>
      <td>$r_t(\boldsymbol{\theta}) \hat{A}_t$</td>
      <td><mark>负向惩罚梯度</mark>：压低该词元的生成概率。</td>
    </tr>
    <tr>
      <td><strong>负优势（$\hat{A}_t < 0$）</strong><br>该动作表现低于预期</td>
      <td>$r_t(\boldsymbol{\theta}) < 1 - \epsilon$<br>概率已被大幅打压</td>
      <td>$(1 - \epsilon) \hat{A}_t$</td>
      <td><strong>导数归零</strong>：停止追打已打压词元，防止概率下溢。</td>
    </tr>
  </tbody>
</table>

---

### 5. 评论家网络价值损失

评论家网络 $V_{\boldsymbol{\phi}}$ 与演员网络并行更新，采用均方误差（MSE）拟合目标回报：

$$
\mathcal{L}^V(\boldsymbol{\phi}) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left( V_{\boldsymbol{\phi}}(s_t) - V_t^{\text{目标}} \right)^2 \right]
$$

其中目标值通常设为 $V_t^{\text{目标}} = \hat{A}_t + V_{\boldsymbol{\phi}_{\text{旧}}}(s_t)$。

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2015">2015年</time> &mdash; <strong>John Schulman 等人提出 TRPO（信任域策略优化）</strong></dt>
  <dd>TRPO 利用费舍尔信息矩阵（Fisher Information Matrix）对策略更新前后的 KL 散度施加严格二阶约束：$\mathbb{E}[D_{\text{KL}}(\pi_{\text{旧}} \parallel \pi)] \le \delta$。虽然理论上完美单调不减，但需要使用共轭梯度法计算海量二阶黑塞矩阵（Hessian）的逆，在大语言模型上计算成本高到无法承受。</dd>

  <dt><time datetime="2017">2017年</time> &mdash; <strong>Schulman 等人提出 PPO（近端策略优化）</strong></dt>
  <dd>用一阶的<em>裁剪替代目标（Clipped Surrogate Objective）</em>彻底取代了高维二阶约束。PPO 仅使用标准一阶 Adam 优化器即可获得媲美甚至超越 TRPO 的训练稳定性，成为现代深度强化学习的事实标准。</dd>

  <dt><time datetime="2022">2022年</time> &mdash; <strong>OpenAI InstructGPT（Ouyang 等人）</strong></dt>
  <dd>成功将 PPO 扩展至 1750 亿参数大模型，首次向全球证明：经过 PPO RLHF 强化对齐的 1.3B 小模型，在真实人类偏好测试中能够击败未经对齐的 175B 原生 GPT-3 模型。</dd>

  <dt><time datetime="2023–2024">2023–2024年</time> &mdash; <strong>4 模型显存危机与架构反思</strong></dt>
  <dd>同时在 GPU 集群显存中部署 4 个完整的 70B 模型（演员、评论家、参考模型、奖励模型），光是加载权重就需要超过 560 GB 显存，外加优化器状态。这种极度沉重的系统负担直接催生了无需评论家网络的 DPO（第 19 章）与 GRPO（第 31 章）。</dd>
</dl>

---

## 步骤 5：手算极简数值示例

让我们用一组最简数值，手工走通一遍单个词元的 PPO 优势计算与裁剪逻辑。

### 极简玩具设定
- 单步生成的动作词元为 $a_t$。
- 旧策略概率：$\pi_{\text{旧}}(a_t \mid s_t) = 0.40$。
- 参考模型基准概率：$\pi_{\text{ref}}(a_t \mid s_t) = 0.50$。
- 奖励模型给出的终局打分：$R_{\boldsymbol{\psi}} = 1.0$。
- KL 惩罚系数：$\beta = 0.1$。
- PPO 裁剪窗口：$\epsilon = 0.2$（允许的比率区间为 $[0.80, 1.20]$）。
- 评论家预测基线：
  - $V(s_t) = 0.60$
  - 终结状态价值：$V(s_{t+1}) = 0.0$（回合结束）。
  - 折现因子：$\gamma = 1.0$。

---

### 第一阶段：单步综合奖励与优势估计

#### 1. 词元级 KL 惩罚
计算对数比值：
$$
\log \left( \frac{\pi_{\text{旧}}}{\pi_{\text{ref}}} \right) = \log\left(\frac{0.40}{0.50}\right) = \log(0.80) \approx -0.2231
$$

扣除 KL 漂移项后的净奖励为：
$$
r_t = R_{\boldsymbol{\psi}} - \beta \log\left(\frac{\pi_{\text{旧}}}{\pi_{\text{ref}}}\right) = 1.0 - 0.1 \times (-0.2231) = 1.0 + 0.0223 = \mathbf{1.0223}
$$
*（由于旧策略比参考模型略保守，不仅没有被扣分，还获得了微弱的合规奖励！）*

#### 2. TD 残差与优势值
因为是最后一个词元，$V(s_{t+1}) = 0$：
$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t) = 1.0223 + 0.0 - 0.60 = +\mathbf{0.4223}
$$

在单步回合中，GAE 优势值等于：
$$
\hat{A}_t = \delta_t^V = +\mathbf{0.4223}
$$
由于 $\hat{A}_t > 0$，说明生成该词元的表现大幅超越了评论家的及格线预估（$0.60$）。

---

### 第二阶段：PPO 裁剪目标函数验证

在策略参数更新过程中，考察以下两种不同幅度的优化情形：

#### 情形 A：策略适度提升概率（更新后 $\pi_{\boldsymbol{\theta}} = 0.46$）
1. 重要性比率：
   $$
   r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\text{旧}}} = \frac{0.46}{0.40} = \mathbf{1.15}
   $$
2. 检验 $r_t$ 是否落在裁剪安全窗口 $[0.80, 1.20]$ 之内？
   - 是！$1.15 \le 1.20$。
3. 计算未裁剪项与裁剪项：
   - 第一项：$r_t \hat{A}_t = 1.15 \times 0.4223 \approx \mathbf{0.4856}$
   - 第二项：$\operatorname{clip}(r_t, 0.8, 1.2) \hat{A}_t = 1.15 \times 0.4223 \approx \mathbf{0.4856}$
4. 目标函数值：
   $$
   \mathcal{L}^{\text{CLIP}} = \min(0.4856, 0.4856) = \mathbf{0.4856}
   $$
   此时导数 $\frac{\partial \mathcal{L}}{\partial \pi} = \frac{\hat{A}_t}{\pi_{\text{旧}}} = \frac{0.4223}{0.40} = +1.056 > 0$，模型平稳接收正向梯度鼓励。

---

#### 情形 B：策略激进提升概率（更新后 $\pi_{\boldsymbol{\theta}} = 0.52$）
1. 重要性比率：
   $$
   r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}}{\pi_{\text{旧}}} = \frac{0.52}{0.40} = \mathbf{1.30}
   $$
2. 检验 $r_t$ 是否落在 $[0.80, 1.20]$ 之内？
   - 否！$1.30 > 1.20$，比率已经击穿了 $1+\epsilon$ 的安全防线！
3. 计算两项：
   - 第一项：$r_t \hat{A}_t = 1.30 \times 0.4223 \approx \mathbf{0.5490}$
   - 第二项：$\operatorname{clip}(1.30, 0.8, 1.2) \hat{A}_t = 1.20 \times 0.4223 \approx \mathbf{0.5068}$
4. 目标函数值：
   $$
   \mathcal{L}^{\text{CLIP}} = \min(0.5490, 0.5068) = \mathbf{0.5068}
   $$
   <mark>目标函数被物理封顶在 $0.5068$！</mark>
   在比率超过 $1.20$ 之后，裁剪项相对于 $\pi_{\boldsymbol{\theta}}$ 是常数，其偏导数瞬间变为 **$0.0$**！优化器在此处坚决停止继续推高该词元，成功挽救模型于过拟合崩溃的悬崖边缘。

---

## 步骤 6：核心精髓总结

<fieldset markdown="1">
<legend><strong>核心精髓总结</strong></legend>

PPO 通过双重防护锁定了大语言模型强化学习的收敛稳定性：<strong>评论家网络</strong>利用广义优势估计（GAE）为海量词元提供低方差的细粒度超额回报；而<strong>裁剪替代目标函数</strong>则在概率分布的外围筑起了一道坚不可摧的数学防火墙 &mdash; 一旦策略漂移超出安全阈值 $\epsilon$，梯度立即归零，彻底根除了深层神经网络策略崩溃的历史死咒。
</fieldset>
