# 第三阶段：拥抱开源生态——主流大模型加载与轻量高效微调 (PEFT / LoRA / DPO)

当你亲手用 PyTorch 搭完 NanoGPT 后，你已经洞悉了自回归模型骨架中的每一根骨头。
但现实工业生产中，没有人会为了解决一个实际业务问题，去从头花费几百万美元预训练一个千亿参数模型。

第三阶段的核心任务是：**学会站在全人类开源成果的肩膀上，掌握现代大模型的标准化工业工具链，用消费级硬件完成模型的能力定制与价值观对齐**。

---

## 必须掌握的开源生态三件套

### 1. Hugging Face 工业标准库
- **`transformers`**：负责模型架构与预训练权重调度。掌握 `AutoTokenizer`、`AutoModelForCausalLM` 和 `pipeline`；
- **`datasets`**：负责大规模训练数据的零内存加载（Memory Mapping 流式读取）；
- **`accelerate`**：无需修改代码，一键实现单机多卡或混合精度（BF16/FP16）训练调度。

### 2. 显存精打细算：为什么全量微调不可行？
一个 7B 参数的模型，权重文件本身占用约 14 GB 显存（按 16 位浮点数计算）。但在进行传统的全量微调（Full Fine-Tuning）时，显存需要支撑：
- 静态模型权重：14 GB
- 反向传播梯度：14 GB
- AdamW 优化器状态（一阶动量 + 二阶方差）：28 GB
- 前向激活值（Activations）：十几 GB
**总显存需求飙升至 70-80 GB，必须依赖昂贵的 A100/H100 显卡！**

### 3. 救命解法：低秩自适应微调（LoRA, Low-Rank Adaptation）
LoRA 的数学核心极为优雅：**保持原始预训练权重 $W_0 \in \mathbb{R}^{d \times k}$ 完全冻结（不存优化器状态），仅在其旁边外挂两个极低维度的旁路矩阵 $A$ 和 $B$**：
$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)
$$
其中 $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}$，秩 $r$ 通常仅取 8 或 16。
- 训练参数量直接从 70 亿（7B）暴降至几百万（不到 0.1%）；
- 优化器显存从数十 GB 暴降至几百 MB；
- 加上 4-bit 量化（QLoRA），**一张 16GB 显存的普通显卡（如 RTX 4080 / 4060Ti）即可轻松微调 7B/8B 旗舰基座模型！**

---

## LoRA 极简微调实战模版

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 1. 加载开源基座模型 (以 Qwen2.5-7B 为例)
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 配置 LoRA 旁路矩阵
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                    # 低秩维度
    lora_alpha=32,                           # 缩放系数
    target_modules=["q_proj", "v_proj"],     # 仅注入注意力投影矩阵
    lora_dropout=0.05,
    bias="none"
)

# 3. 包装模型：99.8% 的参数被瞬间冻结
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例: trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.059%
```

---

## 现代对齐新范式：直接偏好优化（DPO）

在微调掌握特定领域知识后，模型可能会胡说八道或产生攻击性。如何对齐人类价值观？
- **传统 PPO 强化学习**：需要同时在显存中跑四个大模型（Actor、Critic、Reference、Reward），极度复杂且容易崩溃；
- **现代 DPO 革命**：彻底推翻四模型体系，证明了无需训练独立的 Reward 模型，只需利用人类标注的偏好对：
  - Prompt: 提问
  - Chosen: 人类更满意的优质回答
  - Rejected: 人类不满意的糟糕回答
  直接通过闭式解将偏好概率差融入二元交叉熵损失，用简单的有监督格式完成强化学习对齐！

---

## 推荐工业工具链

在实际工程项目中，你不需要自己从头手写数据拼接与梯度调度，推荐熟练掌握以下成熟开源工具：
1. **TRL (Transformer Reinforcement Learning)**：Hugging Face 官方出品的 SFT/DPO/PPO 全流程库；
2. **Unsloth**：针对开源模型手写 Triton 算子，将 LoRA 微调速度提升 2-5 倍，显存降低 70%；
3. **LLaMA-Factory**：集成了 WebUI 界面与百种开源模型的工业级快速微调平台。
