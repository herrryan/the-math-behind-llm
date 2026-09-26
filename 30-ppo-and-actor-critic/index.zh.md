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
1. **信用分配滞后瓶颈（Delayed Credit）**：在长达 1,000 个词元的长文本回复中，模型直到输出完毕才拿到一个总奖励 $R(\tau)$。如果第 20 个词元极具创造力，但在第 995 个词元发生幻觉编造了错误事实，最终奖励变成 0。REINFORCE 会将整个文本里的 1,000 个词元全部打入冷宫！我们急需一个能够在中间步骤评估部分进展的细粒度词元级奖励信号。
2. **步长悬崖崩溃（Policy Collapse）**：在深度神经网络中，权重空间极其微小的更新 $\Delta \boldsymbol{\theta}$ 都会在输出概率空间引发剧烈非线性震荡。如果单次梯度更新迈出的步幅过大，模型概率分布就会突变为胡言乱语。而一旦语言模型跌入崩溃悬崖，后续采样出的全是无意义乱码，模型将彻底丧失恢复能力。

这就是承前启后的关键问题：
$$\text{我们如何利用可学习的价值网络实时估计词元级优势，并在数学上严格限制策略漂移幅度以保证单调稳定收敛？}$$

---

## 步骤 3：严谨数学公式与推导

### 1. 经典 RLHF 的 4 模型系统

在大语言模型经典强化学习对齐中，需要同时协同 4 个独立模型：

1. **演员策略（Actor $\pi_{\boldsymbol{\theta}}$）**：当前负责生成文本的大语言模型，其参数 $\boldsymbol{\theta} \in \mathbb{R}^D$ 持续接受梯度更新以迎合人类偏好。
2. **评论家（Critic / 价值模型 $V_{\boldsymbol{\phi}}$）**：与语言模型具有相似主干架构，但顶层将词表投影层替换为标量回归头的价值网络，参数 $\boldsymbol{\phi}$ 负责拟合期望折现累计回报：$V_{\boldsymbol{\phi}}(s_t) \approx \mathbb{E}\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t \right]$。
3. **参考模型（Reference Policy $\pi_{\text{ref}}$）**：初始监督微调（SFT）模型的完全冻结副本，充当防漂移基准锚点。
4. **奖励模型（Reward Model $R_{\boldsymbol{\psi}}$）**：在人类成对偏好数据 $(y_w \succ y_l)$ 上预先训练并冻结的打分模型（第 19 章）。

---

### 2. 带 KL 散度惩罚的词元级奖励

为了防止演员模型钻奖励模型的漏洞（即<dfn id="def-reward-hacking-zh">奖励黑客攻击 / 刷分作弊</dfn>），并保留原有的泛化表达能力，我们对偏离参考模型的每个词元施加 KL 散度惩罚：

$$
r_t = \begin{cases}
-\beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} \right) & \text{当处于中间词元 } t < T \\
R_{\boldsymbol{\psi}}(x, y) - \beta \log \left( \frac{\pi_{\boldsymbol{\theta}}(a_T \mid s_T)}{\pi_{\text{ref}}(a_T \mid s_T)} \right) & \text{当处于终止词元 } t = T
\end{cases}
$$

其中 $\beta > 0$ 为 KL 惩罚系数。若演员大幅抬高某词元概率导致远超参考模型分布，则对数比值大于 0，从而扣除奖励分数。

---

### 3. 广义优势估计（GAE）

为了评估在状态 $s_t$ 下采纳动作 $a_t$ 到底比预期好多少，我们首先计算**时间差分（TD）残差** $\delta_t^V$：

$$
\delta_t^V = r_t + \gamma V_{\boldsymbol{\phi}}(s_{t+1}) - V_{\boldsymbol{\phi}}(s_t)
$$

其中 $\gamma \in (0, 1]$ 为折现因子（在有限步长语言生成中通常设 $\gamma = 1.0$）。
为了在偏差（较小的 $\lambda$）与方差（较大的 $\lambda$）之间求得最优数学平衡，我们采用<dfn id="def-gae-zh">广义优势估计（Generalized Advantage Estimation, GAE）</dfn>：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}^V
$$

其递推展开形式极为紧凑：

$$
\hat{A}_t = \delta_t^V + (\gamma \lambda) \hat{A}_{t+1}
$$

- 当 $\lambda = 0$ 时：$\hat{A}_t = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$（方差极低，但完全依赖评论家预测准确度，偏差较高）。
- 当 $\lambda = 1$ 时：$\hat{A}_t = \sum_{k=t}^T \gamma^{k-t} r_k - V(s_t)$（无偏的蒙特卡洛经验回报减去基线，方差较高）。
- 在现代大模型强化学习对齐中，标准参数通常选取 $\gamma = 1.0, \lambda = 0.95$。

---

### 4. PPO 裁剪替代目标函数（Clipped Surrogate Objective）

定义当前策略 $\pi_{\boldsymbol{\theta}}$ 与采集数据时的旧策略 $\pi_{\boldsymbol{\theta}_{\text{旧}}}$ 在生成词元 $a_t$ 时的概率比率：

$$
r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{旧}}}(a_t \mid s_t)} \quad (\text{初始状态下 } r_t(\boldsymbol{\theta}_{\text{旧}}) = 1.0)
$$

PPO 裁剪替代目标函数定义为：

$$
\mathcal{L}^{\text{CLIP}}(\boldsymbol{\theta}) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta})\hat{A}_t, \; \operatorname{clip}\left(r_t(\boldsymbol{\theta}), \, 1-\epsilon, \, 1+\epsilon\right)\hat{A}_t \right) \right]
$$

其中 $\epsilon$ 为裁剪阈值超参数（通常设 $\epsilon = 0.2$，将比率严格限制在 $[0.8, 1.2]$ 区间）。

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 30.1：</strong> PPO 裁剪在不同优势与比率区间下的数学响应行为。</caption>
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
      <td><strong>导数归零</strong>：阻止将概率过激打压至绝对零度。</td>
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
