# 第 31 章：群组相对策略优化（GRPO 与 DeepSeek-R1 慢思考推理）

---

## 步骤 1：3 岁小孩直觉（全班随堂测验）

想象一间教室里，老师给四个小朋友出了一道一模一样、很有挑战性的数学谜题：

1. **昂贵的老办法（配私人辅导老师的 PPO 机制）**：
   - 在第 30 章中，PPO 为每个学生配备了一名极其昂贵的专属家庭教师（评论家网络 Critic $V_{\boldsymbol{\phi}}$）。
   - 学生每在纸上写下一个字，家教就在旁边俯下身耳语：“嗯，这个逗号大概值 7 分……这个公式大概值 8 分……”
   - 这个家教体型巨大 &mdash; 一个人占了教室一半的座位（霸占了整整 50% 的 GPU 显存）！更糟糕的是，面对长达上万步的复杂数学推导，家教自己经常猜错分数，甚至给出错误的引导！

2. **DeepSeek-R1 的新办法（黑板群组随堂测验）**：
   - DeepSeek 提出了一个颠覆性的想法：**我们为什么不把这个昂贵的家教彻底开除？**
   - 老师直接把题目写在黑板上，让 **4 个小朋友**同时在黑板上独立作答：
     - **学生 1**：写出完整严谨的推导步骤，并且算出了正确答案 $\to$ **100 分！**
     - **学生 2**：字迹潦草但核心逻辑正确，也算出了正确答案 $\to$ **80 分！**
     - **学生 3**：推导了一半粗心算错 $\to$ **20 分！**
     - **学生 4**：完全胡乱涂鸦 $\to$ **0 分！**

3. **相对于全班平均分的打分机制**：
   - 这道题全班的平均成绩是：
     $$
     \text{平均分} = \frac{100 + 80 + 20 + 0}{4} = 50 \text{ 分}
     $$
   - 学生 1 高出平均分 $+50$ 分 $\to$ **正向优势！** 我们大幅赞赏并强化他的推导步骤。
   - 学生 2 高出平均分 $+30$ 分 $\to$ **正向优势！**
   - 学生 3 低于平均分 $-30$ 分 $\to$ **负向惩罚！** 我们适度抑制他的错误思路。
   - 学生 4 低于平均分 $-50$ 分 $\to$ **严重惩罚！**

完全不需要私人辅导老师！完全不需要庞大的价值评估网络！同批次生成的不同解法互为基线！

<figure>
<pre>
┌────────────────────────────────────────────────────────────────────────┐
│                   GRPO: 彻底摒弃评论家的群组相对优化                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   输入提示 [q] ─────────────► [ 演员策略 π_θ ]                         │
│                                    │                                   │
│            ┌───────────────────────┼───────────────────────┐           │
│            ▼                       ▼                       ▼           │
│        输出 o_1                输出 o_2                输出 o_G        │
│            │                       │                       │           │
│            ▼                       ▼                       ▼           │
│     [ 规则验证器 ]          [ 规则验证器 ]          [ 规则验证器 ]     │
│      答案正确 + 格式         答案正确 + 格式         答案正确 + 格式   │
│            │                       │                       │           │
│            ▼                       ▼                       ▼           │
│        标量奖励 R_1            标量奖励 R_2            标量奖励 R_G    │
│            └───────────────────────┬───────────────────────┘           │
│                                    ▼                                   │
│                    群组均值 μ_q 与标准差 σ_q                           │
│                                    │                                   │
│                                    ▼                                   │
│               归一化相对优势: A_i = (R_i - μ_q) / σ_q                  │
│                                    │                                   │
│                                    ▼                                   │
│                   PPO 裁剪式词元级策略更新                             │
│                  (零 Critic 网络！零显存占用！)                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
</pre>
<figcaption><strong>图 31.1：</strong> GRPO 算法架构：针对每个 Prompt 采样一组输出，通过规则引擎验证计算奖励，按群组统计量归一化优势，驱动策略裁剪更新，彻底消除了评论家模型。</figcaption>
</figure>

---

## 步骤 2：承前启后的关键过渡

在第 30 章中，近端策略优化（PPO）依赖评论家网络（Critic $V_{\boldsymbol{\phi}}$）来估计词元级优势 $\hat{A}_t$。
然而，当把强化学习推向高难度数学推导与代码生成任务（如 DeepSeek-R1 或 OpenAI o1/o3）时，评论家网络成为了系统的致命阿喀琉斯之踵：

1. **GPU 显存暴涨两倍（70B 孪生模型难题）**：
   - 评论家通常采用与生成策略同等参数量的主干网络（如 70B 参数）。
   - 在分布式并行训练中，Adam 优化器针对每个参数需要存储一阶动量与二阶方差两个 32 位浮点数状态，仅优化器状态本身就需要占用每个参数 $16$ 字节！
   - 让一个 70B 的 Critic 与 70B 的 Actor 同时驻留显存，直接让整个分布式集群的 GPU 规模翻倍，消耗数以千万计的算力成本。

2. **长思维链的价值评估幻觉陷阱**：
   - 在一串长达 15,000 个词元的复杂数学推导中，依靠另一个神经网络在中间第 4,210 个词元精准判断该推导是否最终通往正确答案几乎是不可能完成的任务。
   - 一旦评论家网络在中间步骤产生幻觉，错误地将一条充满希望的解题分支打上极低的负向估值，其充满噪声的价值梯度将彻底打乱策略模型，导致训练震荡发散。

3. **古德哈特定律与奖励黑客攻击**：
   - 著名社会学与统计学定律<dfn id="def-goodhart-zh">古德哈特定律（Goodhart's Law）</dfn>指出：*“当一个指标变成目标时，它就不再是一个好指标。”*
   - 在经典 RLHF 中，模型极易学会堆砌客套话和谄媚套路来骗取神经网络打分模型的高分，而根本没有真正解出问题。

这就是承前启后的关键问题：
$$\text{我们如何在数学上彻底剔除数十亿参数的评论家网络，同时利用确定性的规则验证器在群组采样中计算低方差、统计无偏的相对优势？}$$

---

## 步骤 3：严谨数学公式与推导

### 1. 群组采样（Group Rollout Sampling）

针对输入的每个提示问题 $q \sim \mathcal{P}(Q)$，旧策略模型 $\pi_{\boldsymbol{\theta}_{\text{旧}}}$ 独立采样生成 $G$ 个候选回答：

$$
\{o_1, o_2, \dots, o_G\} \sim \pi_{\boldsymbol{\theta}_{\text{旧}}}(q)
$$

其中 $G$ 为群组规模（在 DeepSeek-Math 与 DeepSeek-R1 中通常设为 $G \in [4, 16]$）。

---

### 2. 基于规则的可验证奖励（RLVR）

为了根除神经奖励模型容易被“套分”的弊端，**可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards, RLVR）**采用绝对客观、无死角的确定性判别函数：

$$
R_i = r_{\text{acc}}(o_i) + r_{\text{format}}(o_i)
$$

1. **准确度奖励（$r_{\text{acc}}$）**：检验最终提取出的数学答案或执行代码是否与标准答案绝对一致：
   $$
   r_{\text{acc}}(o_i) = \begin{cases} 1.0 & \text{当数学答案匹配或代码单元测试全过} \\ 0.0 & \text{否则} \end{cases}
   $$
2. **格式规范奖励（$r_{\text{format}}$）**：约束模型的思考结构，强制其将内省长思维链与最终答案严格分割：
   $$
   r_{\text{format}}(o_i) = \begin{cases} 1.0 & \text{当输出严格符合 } \texttt{<think>...</think><answer>...</answer>} \\ 0.0 & \text{否则} \end{cases}
   $$

<fieldset markdown="1">
<legend><strong>为什么 RLVR 能天然免疫古德哈特定律？</strong></legend>

在传统人类偏好奖励模型下，大模型发现只要语气谦逊、排版华丽，就能轻松获得 9.5 分的高分。

但在 RLVR 体系下，裁判是一个冷酷、客观的 Python 脚本（例如执行 `assert solve() == 42` 或正则表达式解析 LaTeX `\boxed{...}`）。任何客套话在单元测试面前都毫无意义。编译器不听奉承！答案做对就是 1 分，做错就是 0 分。模型作弊套分的路径被彻底切断。
</fieldset>

---

### 3. 群组相对优势归一化

对于问题 $q$，得到这 $G$ 个采样的奖励分数集合 $\{R_1, R_2, \dots, R_G\}$。
计算该群组的经验均值 $\mu_q$ 与经验标准差 $\sigma_q$：

$$
\mu_q = \frac{1}{G} \sum_{i=1}^G R_i, \quad \sigma_q = \sqrt{\frac{1}{G} \sum_{i=1}^G \left( R_i - \mu_q \right)^2 + \epsilon}
$$

其中 $\epsilon = 10^{-4}$ 用于防止除以零。
第 $i$ 个候选回答的<dfn id="def-grpo-advantage-zh">群组相对优势（Group-Relative Advantage）</dfn>定义为：

$$
A_i = \frac{R_i - \mu_q}{\sigma_q}
$$

<fieldset markdown="1">
<legend><strong>第一性原理拆解：为什么群组归一化能彻底消灭评论家网络？</strong></legend>

将此式直接关联回第 29 章的策略梯度基线定理：$\nabla J = \mathbb{E}[ \nabla \log \pi \cdot (R - b) ]$。

在 GRPO 中，<strong>群组样本均值 $\mu_q$ 正是该提示词下最纯粹的经验无偏基线 $b(q)$！</strong>

1. **天然零和性（Zero-Sum）**：观察未归一化优势在群组内部求和的结果：
   $$
   \sum_{i=1}^G (R_i - \mu_q) = \sum_{i=1}^G R_i - G \cdot \left( \frac{1}{G}\sum_{i=1}^G R_i \right) = \sum_{i=1}^G R_i - \sum_{i=1}^G R_i \equiv 0
   $$
   优势值严格以零为中心对称！在每一次采样中，必定只有高于当前平均水平的回答获得正向鼓励（$A_i > 0$），低于平均水平的回答受到抑制（$A_i < 0$）。
2. **自适应梯度屏蔽（Self-Tuning Gradient Attenuation）**：
   - 如果一道题目*太简单*，模型的 $G$ 个候选回答全部做对（全拿 1 分）：则 $\mu_q = 1$，分子 $R_i - \mu_q = 0 \implies A_i = 0$。
   - 如果一道题目*太难*，模型的 $G$ 个候选回答全部做错（全拿 0 分）：则 $\mu_q = 0$，分子 $R_i - \mu_q = 0 \implies A_i = 0$。
   - **惊艳的工程效果**：模型在完全掌握的题目和完全无能为力的题目上**自动产生零梯度更新**！模型绝不会在已经学会的知识上白白抖动参数，而是自发把 100% 的学习能力聚焦在处于自身能力边界前沿的挑战性题目上！
</fieldset>

---

### 4. GRPO 完整目标函数

GRPO 针对策略参数 $\boldsymbol{\theta}$ 的完整优化目标为：

$$
\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\boldsymbol{\theta}) = \mathbb{E}_{\substack{q \sim \mathcal{P}(Q) \\ \{o_i\}_{i=1}^G \sim \pi_{\boldsymbol{\theta}_{\text{旧}}}(q)}} \Bigg[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \bigg( &\min\left( \rho_{i,t}(\boldsymbol{\theta}) A_i, \; \operatorname{clip}\left(\rho_{i,t}(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon\right) A_i \right) \\
&- \beta D_{\text{KL}}\left(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}\right) \bigg) \Bigg]
\end{aligned}
$$

其中各项数学含义如下：
- $\rho_{i,t}(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}_{\text{旧}}}(o_{i,t} \mid q, o_{i,<t})}$ 为词元级重要性采样比率。
- $|o_i|$ 为第 $i$ 个回答的文本词元总长度。除以 $|o_i|$ 至关重要：它保证了简短回答与超长思维链在梯度更新中权值公平，防止冗长回答仅仅因为字数多就强行霸占梯度的统治权。
- $\epsilon$ 为 PPO 裁剪常数（通常 $\epsilon \approx 0.2$）。
- $\beta$ 为参考模型锚定系数。

---

### 5. Schulman 无偏非负 KL 估计器

在计算参考模型散度时，如果直接用简单的单点采样 $\log(\pi / \pi_{\text{ref}})$，可能在个别样本上出现负值导致梯度震荡。DeepSeek 采用了 John Schulman（2020）提出的低方差无偏非负估计形式：

$$
D_{\text{KL}}\left(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}\right) \approx \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})} - \log \left( \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\boldsymbol{\theta}}(o_{i,t} \mid q, o_{i,<t})} \right) - 1
$$

<fieldset markdown="1">
<legend><strong>第一性原理微积分证明：为什么该估计器处处非负？</strong></legend>

令比率变量为 $u = \frac{\pi_{\text{ref}}}{\pi_{\boldsymbol{\theta}}} > 0$。考察一元标量函数：

$$
g(u) = u - \log u - 1
$$

<ol>
  <li>当两策略分布完全吻合时（$u = 1$）：$g(1) = 1 - \log(1) - 1 = 1 - 0 - 1 = 0$。</li>
  <li>对 $u$ 求一阶导数：
    $$
    g'(u) = 1 - \frac{1}{u}
    $$
  </li>
  <li>分析导数正负与函数单调性：
    <ul>
      <li>当 $u > 1$ 时：$g'(u) > 0 \implies$ 函数严格单调递增。</li>
      <li>当 $u < 1$ 时：$g'(u) < 0 \implies$ 函数严格单调递减。</li>
    </ul>
  </li>
  <li>因此，$u = 1$ 是函数 $g(u)$ 在其定义域上的唯一<strong>全局最小值点</strong>，即：
    $$
    g(u) \ge 0 \quad (\forall u > 0)
    $$
  </li>
</ol>

这个初等微积分推导无可辩驳地证明了：该 KL 散度估计器在<strong>每一个单独词元上均严格非负（$\ge 0$）</strong>，彻底杜绝了参考模型散度在个别词元上转为负数、意外向模型提供伪奖励的系统隐患！
</fieldset>

---

## 步骤 4：历史渊源与技术演进

<dl>
  <dt><time datetime="2024-02">2024年2月</time> &mdash; <strong>Shao 等人（DeepSeek-Math）</strong></dt>
  <dd>首次提出 <em>群组相对策略优化（GRPO）</em>，成功去除了 PPO 的评论家网络，将训练所需显存削减一半，并在 GSM8K（88.2%）与竞赛级 MATH（51.7%）数据集上达到当时开源推理领域的最高水准。</dd>

  <dt><time datetime="2025-01">2025年1月</time> &mdash; <strong>DeepSeek-AI（DeepSeek-R1 与 R1-Zero）</strong></dt>
  <dd>直接在基座模型上应用纯粹的 GRPO 强化学习（完全跳过人类监督微调 SFT）。震惊全球地观测到了<strong>长思维链推理能力的自主涌现</strong>：模型在纯规则奖励的驱动下，自主学会了生成数千个词元的深度思考、自我质疑、回溯纠错，并自发出现了类似人类顿悟的“Aha! 让我重新检查一下……”内省推理行为。</dd>
</dl>

### 经典算法架构全景对比

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 31.1：</strong> 大模型主流后训练强化学习算法架构横向对比。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="left">对比维度</th>
      <th align="left">PPO（第 30 章）</th>
      <th align="left">DPO（第 19 章）</th>
      <th align="left">GRPO（DeepSeek-R1）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>显存常驻模型数</strong></td>
      <td>4 个（演员、评论家、参考、奖励）</td>
      <td><strong>2 个</strong>（演员、参考）</td>
      <td><strong>2 个</strong>（演员、参考）</td>
    </tr>
    <tr>
      <td><strong>评论家网络</strong></td>
      <td>完整的大型神经网络</td>
      <td>无（数学上消去）</td>
      <td><strong>无（由群组统计量动态替代）</strong></td>
    </tr>
    <tr>
      <td><strong>自主探索能力</strong></td>
      <td>在线动态采样</td>
      <td><mark>零（依赖静态离线偏好对）</mark></td>
      <td><strong>全在线群组自主探索</strong></td>
    </tr>
    <tr>
      <td><strong>奖励信号来源</strong></td>
      <td>神经网络奖励模型</td>
      <td>成对标注离线概率</td>
      <td><strong>客观规则引擎（RLVR）</strong></td>
    </tr>
    <tr>
      <td><strong>慢思考能力涌现</strong></td>
      <td>长链易受 Critic 误差干扰</td>
      <td>无法发现未见过的推导链</td>
      <td><strong>自发展开长思维链与自我纠错</strong></td>
    </tr>
  </tbody>
</table>

---

## 步骤 5：手算极简数值示例

让我们用一组极简数据，完整手算一遍 GRPO 的群组优势分配与更新。

### 极简玩具设定
- 输入数学问题 $q$：<kbd>"计算 2 + 3 * 4"</kbd>。
- 标准参考答案：$14$。
- 群组采样规模：$G = 4$。
- 评分规则：
  - 格式分：包含 `<think>...</think><answer>...</answer>` 给 $+0.5$ 分。
  - 准确分：最终答案为 $14$ 给 $+1.0$ 分。
  - 单项满分：$1.5$ 分。

---

### 第一阶段：评估群组输出并给出标量奖励

<table border="1" cellpadding="8" cellspacing="0" width="100%">
  <caption><strong>表 31.2：</strong> 4 个采样输出的得分明细。</caption>
  <thead>
    <tr bgcolor="#eae9e1">
      <th align="center">编号</th>
      <th align="left">模型生成的回答</th>
      <th align="center">格式分（$r_{\text{fmt}}$）</th>
      <th align="center">准确分（$r_{\text{acc}}$）</th>
      <th align="right">总得分（$R_i$）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">$o_1$</td>
      <td><samp>&lt;think&gt;3*4=12, 12+2=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;</samp></td>
      <td align="center">1.0 (+0.5)</td>
      <td align="center">1.0 (+1.0)</td>
      <td align="right"><strong>1.5</strong></td>
    </tr>
    <tr>
      <td align="center">$o_2$</td>
      <td><samp>&lt;think&gt;运算顺序：2+12=14&lt;/think&gt;&lt;answer&gt;14&lt;/answer&gt;</samp></td>
      <td align="center">1.0 (+0.5)</td>
      <td align="center">1.0 (+1.0)</td>
      <td align="right"><strong>1.5</strong></td>
    </tr>
    <tr>
      <td align="center">$o_3$</td>
      <td><samp>&lt;think&gt;2+3=5, 5*4=20&lt;/think&gt;&lt;answer&gt;20&lt;/answer&gt;</samp></td>
      <td align="center">1.0 (+0.5)</td>
      <td align="center">0.0 (+0.0)</td>
      <td align="right"><strong>0.5</strong></td>
    </tr>
    <tr>
      <td align="center">$o_4$</td>
      <td><samp>答案是 14。</samp></td>
      <td align="center">0.0 (+0.0)</td>
      <td align="center">1.0 (+1.0)</td>
      <td align="right"><strong>1.0</strong></td>
    </tr>
  </tbody>
</table>

得到当前批次的分数向量：$\mathbf{R} = [1.5, \; 1.5, \; 0.5, \; 1.0]$。

---

### 第二阶段：计算群组统计量与归一化优势

#### 1. 计算群组均值（$\mu_q$）
$$
\mu_q = \frac{1.5 + 1.5 + 0.5 + 1.0}{4} = \frac{4.5}{4} = \mathbf{1.125}
$$

#### 2. 计算群组方差与标准差（$\sigma_q$）
各样本与均值的偏差：
- $(1.5 - 1.125)^2 = (+0.375)^2 = 0.140625$
- $(1.5 - 1.125)^2 = (+0.375)^2 = 0.140625$
- $(0.5 - 1.125)^2 = (-0.625)^2 = 0.390625$
- $(1.0 - 1.125)^2 = (-0.125)^2 = 0.015625$

方差：
$$
\sigma_q^2 = \frac{0.140625 + 0.140625 + 0.390625 + 0.015625}{4} = \frac{0.6875}{4} = 0.171875
$$
标准差：
$$
\sigma_q = \sqrt{0.171875} \approx \mathbf{0.4146}
$$

#### 3. 计算各个回答的相对优势（$A_i = \frac{R_i - \mu_q}{\sigma_q}$）
- $A_1 = \frac{1.5 - 1.125}{0.4146} = \frac{+0.375}{0.4146} \approx \mathbf{+0.9045}$（<mark>高质量正向强化</mark>）
- $A_2 = \frac{1.5 - 1.125}{0.4146} = \frac{+0.375}{0.4146} \approx \mathbf{+0.9045}$（<mark>高质量正向强化</mark>）
- $A_3 = \frac{0.5 - 1.125}{0.4146} = \frac{-0.625}{0.4146} \approx \mathbf{-1.5077}$（<del>算错数学步骤严厉惩罚</del>）
- $A_4 = \frac{1.0 - 1.125}{0.4146} = \frac{-0.125}{0.4146} \approx \mathbf{-0.3015}$（未按格式输出标签温和扣分）

验证优势值总和：$+0.9045 + 0.9045 - 1.5077 - 0.3015 = -0.0002 \approx \mathbf{0}$！

---

### 第三阶段：词元级策略裁剪更新验证

考察回答 $o_1$ 中生成的关键推理词元 <kbd>"12"</kbd>：
- 采样时旧策略概率：$\pi_{\text{旧}}("12" \mid \dots) = 0.20$。
- 更新后当前策略概率：$\pi_{\boldsymbol{\theta}}("12" \mid \dots) = 0.23$。
- 重要性比率：
  $$
  \rho_{1, t} = \frac{0.23}{0.20} = \mathbf{1.15}
  $$
- 裁剪安全区间（$\epsilon = 0.2$）：$[0.80, 1.20]$。
- 由于 $1.15 \le 1.20$，未触发裁剪。
- 替代项值：
  $$
  \rho_{1, t} A_1 = 1.15 \times (+0.9045) = \mathbf{+1.0402}
  $$
- 该词元获得正向梯度更新，引导模型牢固掌握“在计算 $2+3 \times 4$ 时先算出 $12$”是高胜率推理路径！

---

## 步骤 6：核心精髓总结

<fieldset markdown="1">
<legend><strong>核心精髓总结</strong></legend>

群组相对策略优化（GRPO）通过同题多路采样的群组均值与方差完全替代了昂贵且脆弱的评论家网络，彻底击碎了长思维链强化学习的显存墙。

配合严格客观的可验证规则奖励（RLVR），GRPO 为大语言模型注入了自我进化的数学引擎 &mdash; 使其能够在无人类长推导标注的纯自主试错中，自发展开深度慢思考、反复自省检验与推理跃迁。
</fieldset>
