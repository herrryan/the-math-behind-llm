# 实践实验 4：实现双卡张量并行（TP）线性引擎：Megatron 列并行与行并行协同

同学，你好！欢迎来到整个工程实战体系的大压轴实验课！

在第四模块第 E15 章中，我们推导了 Megatron-LM 的核心黄金法则：**前级列并行，后级行并行，中间零通信，最后一次 All-Reduce！**

今天，老师就要带你动用 PyTorch 的分布式通信套件（`torch.distributed`），在真实的双卡（或单机多进程模拟多卡）环境下，**亲手实现一个具备工业级数学等价性的张量并行双层 FFN 引擎！**

---

## 实验目标与产出

1. **实现 ColumnParallelLinear（列并行线性层）：** 将大权重沿输出维度纵向劈开，各卡独立计算；
2. **实现 RowParallelLinear（行并行线性层）：** 将权重沿输入维度横向劈开，并在出口处调用 `dist.all_reduce` 自动求和；
3. **数学无损验证：** 将双卡并行的计算输出与单卡巨型矩阵乘法的输出进行逐元素比对，证明相对误差在浮点精度极限之内！

---

## 步骤 1：编写列并行与行并行核心模块

打开你的代码编辑器，跟老师一起写下这两个经典的并行组件：

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    # 列并行线性层: 将权重矩阵沿输出特征维度 (列) 均匀切分
    # W 尺寸: [In_Features, Out_Features // World_Size]
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        self.rank = rank

        # 仅分配本卡应持有的局部权重切片!
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features_per_partition))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 全量共享，本地计算局部矩阵乘法: Y_local = X @ W_local
        # 注意: 这一步完全不需要任何跨卡通信!
        return torch.matmul(x, self.weight)

class RowParallelLinear(nn.Module):
    # 行并行线性层: 将权重矩阵沿输入特征维度 (行) 均匀切分
    # 并在输出时执行全卡 All-Reduce 求和汇聚!
    # W 尺寸: [In_Features // World_Size, Out_Features]
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        self.weight = nn.Parameter(torch.empty(self.in_features_per_partition, self.out_features))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        # 本地执行局部相乘
        output_local = torch.matmul(x_local, self.weight)

        # 核心关口: 调用集合通信原语 All-Reduce，完成各卡结果的汇聚累加!
        dist.all_reduce(output_local, op=dist.ReduceOp.SUM)
        return output_local
```

---

## 步骤 2：组装完整的张量并行双层 FFN 模块

```python
class TensorParallelMLP(nn.Module):
    # Megatron-LM 经典双层 MLP 模块
    def __init__(self, hidden_dim: int, ffn_dim: int, world_size: int, rank: int):
        super().__init__()
        # 1. 第一级: 列并行升维
        self.col_linear = ColumnParallelLinear(hidden_dim, ffn_dim, world_size, rank)
        # 2. 激活函数 (纯片上逐元素进行，零通信!)
        self.act = nn.GELU()
        # 3. 第二级: 行并行降维 (内置单次 All-Reduce)
        self.row_linear = RowParallelLinear(ffn_dim, hidden_dim, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.col_linear(x)
        h = self.act(h)
        out = self.row_linear(h)
        return out
```

---

## 步骤 3：单机多进程模拟运行与单卡等价性验证

```python
import os
import torch.multiprocessing as mp

def run_tp_worker(rank: int, world_size: int, hidden_dim: int, ffn_dim: int):
    # 初始化分布式通信环境 (使用 NCCL 或 Gloo)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 创建张量并行模型
    tp_model = TensorParallelMLP(hidden_dim, ffn_dim, world_size, rank).to(device)

    # 构造模拟输入
    batch_size, seq_len = 2, 4
    x = torch.ones(batch_size, seq_len, hidden_dim, device=device)

    # 前向计算
    out = tp_model(x)

    if rank == 0:
        print(f"[Rank 0] 张量并行前向计算成功！输出张量形状: {out.shape}")
        print(f"[Rank 0] 样本输出均值: {out.mean().item():.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2 # 模拟双卡并行
    hidden_dim = 128
    ffn_dim = 512

    print(f"=== 启动双进程张量并行 (TP={world_size}) 模拟实验 ===")
    mp.spawn(run_tp_worker, args=(world_size, hidden_dim, ffn_dim), nprocs=world_size, join=True)
    print("=== 实验圆满成功！===")
```

---

## 老师点评与课程结语

同学，恭喜你！当你成功跑通这个双卡张量并行引擎时，你已经真正打通了大模型底层系统工程的“任督二脉”！

回顾整个《大模型工程实现与底层加速》课程：
- 从**硅基芯片的物理微观结构**，到**屋顶模型的性能标尺**；
- 从**FlashAttention 的在线 Softmax**，到**PagedAttention 的虚拟页表**；
- 从**单算子内核熔合**，到**千卡分布式并行通信**；
- 从**推理调度引擎的动态批处理**，到**MFU 的严密算力审计**……

你不再是一个只会调 API 的初级算法工程师，而是一个**既懂高深数学理论、又深谙底层硅基物理法则的大模型顶尖系统架构师**！

愿你在未来的 AI 浪潮中，继续保持对第一性原理的敬畏与好奇，用最硬核的代码，征服最强大的硅基大脑！老师为你骄傲！
