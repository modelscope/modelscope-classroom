# Megatron

Megatron是NVIDIA开发的大规模语言模型训练框架，专为高效利用大规模GPU集群而设计。它实现了张量并行、流水线并行、序列并行等多种并行策略，是训练千亿参数级模型的重要工具。

## 概述与定位

Megatron最初由NVIDIA发布用于训练GPT系列模型，后与DeepSpeed结合形成了Megatron-DeepSpeed，在工业界大模型训练中广泛应用。

**核心特性**：
- 高效的张量并行实现
- 流水线并行与交错调度
- 序列并行减少激活显存
- 与DeepSpeed ZeRO的深度集成
- 针对NVIDIA GPU的深度优化

**适用场景**：
- 百亿至千亿参数模型预训练
- 大规模GPU集群（数十到数千张卡）
- 需要极致训练效率的场景

## 张量并行

Megatron的张量并行将模型层的权重切分到多个GPU上，主要针对Transformer的线性层：

### 列并行（Column Parallel）

将权重按列切分，每个GPU持有权重的一部分列：

$$\mathbf{Y} = \mathbf{X}\mathbf{W} = \mathbf{X}[\mathbf{W}_1, \mathbf{W}_2] = [\mathbf{X}\mathbf{W}_1, \mathbf{X}\mathbf{W}_2]$$

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank):
        super().__init__()
        self.output_size_per_partition = output_size // world_size
        self.weight = nn.Parameter(
            torch.empty(input_size, self.output_size_per_partition)
        )
        self.rank = rank
    
    def forward(self, x):
        # x: [batch, seq, input_size]
        # output: [batch, seq, output_size_per_partition]
        output = F.linear(x, self.weight.t())
        return output
```

### 行并行（Row Parallel）

将权重按行切分，输入需要按列切分，输出进行AllReduce聚合：

$$\mathbf{Y} = \mathbf{X}\mathbf{W} = [\mathbf{X}_1, \mathbf{X}_2]\begin{bmatrix}\mathbf{W}_1 \\ \mathbf{W}_2\end{bmatrix} = \mathbf{X}_1\mathbf{W}_1 + \mathbf{X}_2\mathbf{W}_2$$

```python
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, world_size, rank):
        super().__init__()
        self.input_size_per_partition = input_size // world_size
        self.weight = nn.Parameter(
            torch.empty(self.input_size_per_partition, output_size)
        )
    
    def forward(self, x):
        # x已经是切分后的输入
        output_parallel = F.linear(x, self.weight.t())
        # AllReduce聚合
        torch.distributed.all_reduce(output_parallel)
        return output_parallel
```

### Attention并行化

MLP层与Attention层的并行化策略：

```
MLP:
  [X] → ColumnParallel(up_proj) → Activation → RowParallel(down_proj) → [Y]
         无需通信                              AllReduce

Attention:
  [X] → ColumnParallel(QKV) → Attention计算 → RowParallel(out_proj) → [Y]
         无需通信            本地计算        AllReduce
```

通过精心设计，张量并行在每个Transformer层只需要2次AllReduce通信。

## 流水线并行

流水线并行将模型按层划分到不同的GPU（称为stage），采用微批次（micro-batch）流水线调度：

### GPipe调度

GPipe将一个batch切分为多个micro-batch，依次通过各stage：

```
Stage 0: [M0] [M1] [M2] [M3] [  ] [  ] [  ] [  ] [M3'] [M2'] [M1'] [M0']
Stage 1: [  ] [M0] [M1] [M2] [M3] [  ] [  ] [M3'] [M2'] [M1'] [M0'] [  ]
Stage 2: [  ] [  ] [M0] [M1] [M2] [M3] [M3'] [M2'] [M1'] [M0'] [  ] [  ]
Stage 3: [  ] [  ] [  ] [M0] [M1] [M2] [M2'] [M1'] [M0'] [  ] [  ] [  ]
                    |<-- Forward -->|<-- Backward -->|
```

### 1F1B调度

1F1B（One Forward One Backward）通过交错前向和反向减少气泡：

```
Stage 0: [F0] [F1] [F2] [F3] [B0] [F4] [B1] [F5] [B2] ...
Stage 1:      [F0] [F1] [F2] [B0] [F3] [B1] [F4] [B2] ...
```

### 交错调度

Megatron实现的交错流水线并行，将多个虚拟stage分配到同一GPU：

```python
# 每个GPU持有多个虚拟stage（非连续层）
# 例如：4个stage，每个GPU 2个虚拟stage
# GPU 0: [Layer 0, 1] + [Layer 8, 9]
# GPU 1: [Layer 2, 3] + [Layer 10, 11]
# GPU 2: [Layer 4, 5] + [Layer 12, 13]
# GPU 3: [Layer 6, 7] + [Layer 14, 15]
```

这种设计减少了流水线气泡。

## 序列并行

序列并行将序列维度切分到张量并行组的各GPU上，与张量并行配合：

```python
# 非张量并行区域（LayerNorm、Dropout）使用序列并行
# 输入: [batch, seq/tp_size, hidden] 在每个GPU上
# 操作: LayerNorm/Dropout 在切分的序列上独立进行
# 输出: [batch, seq/tp_size, hidden]

# 进入张量并行区域时，通过AllGather收集完整序列
# 退出张量并行区域时，通过ReduceScatter切分序列
```

序列并行显著降低了激活内存占用，特别是对于长序列场景。

## 使用方式

### 配置参数

```bash
python pretrain_gpt.py \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 500000 \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 0.00001 \
    --lr-warmup-fraction 0.01 \
    --fp16 \
    --data-path /path/to/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt
```

关键参数说明：

| 参数 | 说明 |
|-----|------|
| `tensor-model-parallel-size` | 张量并行度 |
| `pipeline-model-parallel-size` | 流水线并行度 |
| `micro-batch-size` | 每个micro-batch的大小 |
| `global-batch-size` | 全局batch大小 |
| `sequence-parallel` | 启用序列并行 |
| `recompute-activations` | 激活重计算 |

### 数据并行

总GPU数 = 张量并行 × 流水线并行 × 数据并行

```bash
# 64个GPU，TP=8，PP=4
# 数据并行度 = 64 / (8 * 4) = 2
```

### 与DeepSpeed集成

Megatron-DeepSpeed结合了两者优势：

```bash
deepspeed --num_gpus 64 pretrain_gpt.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    ...
```

DeepSpeed配置：

```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32,
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 500
    }
}
```

## 性能优化技巧

### 通信优化

```python
# 重叠计算与通信
# AllReduce与下一层计算重叠
output = F.linear(x, weight)
handle = dist.all_reduce(output, async_op=True)
# 执行其他计算...
handle.wait()
```

### 激活检查点

```python
# 在内存与计算间权衡
from megatron.core.tensor_parallel import checkpoint

def transformer_layer(x, layer):
    # 前向时丢弃中间激活，反向时重新计算
    return checkpoint(layer, x)
```

### Flash Attention集成

```bash
--use-flash-attn  # 启用Flash Attention
```

## 模型架构支持

Megatron支持多种模型架构：

- **GPT系列**：GPT-2、GPT-3风格的解码器模型
- **BERT系列**：双向编码器模型
- **T5系列**：编码器-解码器模型
- **LLaMA风格**：通过Megatron-LLaMA扩展

社区扩展如Megatron-LM、Megatron-Core提供了更多模型支持。

## 与其他框架的对比

| 特性 | Megatron | DeepSpeed | FSDP |
|-----|----------|-----------|------|
| 张量并行 | 原生支持 | ZeRO-TP | 有限 |
| 流水线并行 | 高效实现 | 支持 | 有限 |
| 序列并行 | 支持 | 部分 | 不支持 |
| 易用性 | 中等 | 较好 | 好 |
| 适用规模 | 超大 | 大 | 中大 |

Megatron在超大规模训练中展现出卓越的效率，但学习曲线相对陡峭。对于百亿参数以下的模型，FSDP或DeepSpeed可能是更便捷的选择；对于追求极致效率的千亿级训练，Megatron仍是首选方案。
