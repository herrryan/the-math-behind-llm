# 实验 E4：双卡张量并行（Tensor Parallelism）引擎手工构建实战

## 实验目标与挑战

Megatron-LM 的“列切 + 行切”张量并行，是支撑超千亿大模型跨卡切分训练与推理的绝对主力支柱。

**本实验的核心任务：** 纯手工使用 PyTorch 集合通信库（`torch.distributed`），在两张物理 GPU（或通过单机多进程模拟多卡）上，完整构建一个标准 Transformer MLP 层的 **张量并行引擎**。
1. 实现 **列并行线性层（ColumnParallelLinear）**：将权重矩阵按列拆分，处理输入并验证其在不需要跨卡通信的前提下各自输出局部切片；
2. 实现 **行并行线性层（RowParallelLinear）**：将权重矩阵按行拆分，接收切片输入并计算出部分和（Partial Sum）；
3. 注入 `all_reduce` 集合通信：将两张卡计算出的部分和无缝累加聚合；
4. 编写全量对齐测试用例：在单卡上运行标准的非切分 MLP，将双卡并行引擎的最终输出与单卡标准答案进行绝对数值比对，证明两者的浮点输出完全严格对齐（误差控制在 $10^{-6}$ 以内）。

---

## 动手实践代码框架

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.split_out_features = out_features // world_size
        self.weight = nn.Parameter(torch.empty(self.split_out_features, in_features))
        # 实际代码中需从完整权重中切取对应 rank 的切片
        
    def forward(self, x):
        # 列并行: 输入广播，各卡独立计算局部投影
        return torch.matmul(x, self.weight.t())

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.split_in_features = in_features // world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.split_in_features))
        
    def forward(self, x):
        # 行并行: 接收切片，计算部分和，并通过 All-Reduce 聚合
        partial_sum = torch.matmul(x, self.weight.t())
        dist.all_reduce(partial_sum, op=dist.ReduceOp.SUM)
        return partial_sum
```

---

## 思考与检验题
1. 如果在列并行后加入偏置项（Bias），系统会出现什么数学错误？为什么偏置项通常只能安全地加在行并行的输出端？
2. 试推导在反向传播过程中，列并行与行并行各自分别会触发什么样的集合通信（提示：前向与反向在数学上互为对偶变换）？
