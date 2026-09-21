# 实验 E3：PagedAttention 分页 KV Cache 引擎从零开发实战

## 实验目标与挑战

vLLM 之所以能在一夜之间重塑全球大模型在线推理服务生态，其基石正是借鉴了经典操作系统的**分页虚拟内存（Virtual Paging）**思想。

**本实验的核心任务：** 使用纯 Python 与 NumPy/PyTorch，从零构建一个完整的 **PagedAttention 模拟管理引擎**。
1. **全局物理块管理器（Block Allocator）：** 预先分配一块固定的显存池，并切分为若干固定大小的物理 Block（例如每个 Block 存放 16 个 Token 的 KV 向量）；维护一个空闲链表（Free List）；
2. **逻辑到物理映射页表（Block Table）：** 为每个进入系统的并发请求维护一个动态增长的虚拟页表，支持随序列生成按需动态申请新的物理 Block；
3. **写时复制机制（Copy-on-Write）：** 模拟多分支采样场景（如束搜索 Beam Search），让子请求共享父请求的历史物理块，只有当发生分歧写入新 Token 时才触发物理块的深拷贝复制；
4. **并发压力模拟评测：** 模拟 100 条长短不一的随机请求并发到达，对比传统“静态连续预分配”与你的“分页引擎”在显存碎片浪费率与最大支持并发数上的物理表现。

---

## 动手实践代码框架

```python
class PhysicalBlockPool:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        # 预分配扁平物理显存池
        self.k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.v_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        
    def allocate(self):
        if not self.free_blocks:
            raise MemoryError("物理显存池已耗尽 (OOM)!")
        return self.free_blocks.pop(0)

    def free(self, block_id):
        self.free_blocks.append(block_id)

class LogicalSequence:
    def __init__(self, seq_id, pool):
        self.seq_id = seq_id
        self.pool = pool
        self.block_table = []
        self.num_tokens = 0

    def append_token_kv(self, k_vec, v_vec):
        # 核心任务: 检查最后一个 Block 是否已满，若满则申请新 Block，并将 KV 写入物理显存池!
        pass
```

---

## 思考与检验题
1. Block 大小（`block_size`）对系统性能有什么物理影响？为什么主流系统大多选用 16 或 32，而不是 1 或 1024？
2. 在这个框架下，如何实现前缀缓存（Prefix Caching）的重用逻辑？
