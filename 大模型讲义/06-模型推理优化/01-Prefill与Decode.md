# 6.1 Prefill 与 Decode

LLM 的自回归生成可以分解为两个阶段：**Prefill**（预填充）处理输入 prompt，**Decode**（解码）逐个生成输出 token。这两个阶段的计算特性截然不同，理解它们是优化推理的基础。

## 6.1.1 自回归生成的基本流程

### Transformer 推理

给定输入序列 $x_1, x_2, \ldots, x_n$，Transformer 的推理过程是：

1. **Embedding**：将 token 转换为向量
2. **逐层计算**：通过 $L$ 层 Transformer block
3. **输出**：最后一层的隐藏状态通过 LM head 得到 logits

对于自回归语言模型，关键在于：生成第 $t+1$ 个 token 时，需要访问前 $t$ 个 token 的信息。这通过**因果注意力**（Causal Attention）实现——每个位置只能看到自己和之前的位置。

### 朴素实现的问题

最朴素的实现：每生成一个新 token，重新计算整个序列的注意力。

设序列长度为 $n$，生成 $m$ 个 token，总计算量为：

$$\sum_{t=n}^{n+m-1} O(t^2 \cdot d) = O((n+m)^2 \cdot m \cdot d)$$

对于长 prompt 或长输出，这个开销难以接受。

## 6.1.2 Prefill 阶段

### 定义

**Prefill**（预填充）阶段处理用户输入的 prompt。所有 prompt token 已知，可以一次性并行处理。

设 prompt 长度为 $n$，Prefill 需要：

1. 计算所有 $n$ 个 token 的 embedding
2. 逐层计算注意力和 FFN
3. 为 Decode 阶段准备 KV Cache

### 计算特点

**并行性好**：所有 token 可以同时计算，类似训练的前向传播。

**计算密集**：主要瓶颈是矩阵乘法（计算 Q、K、V 和注意力），GPU 利用率高。

**一次性开销**：每个请求只需一次 Prefill。

### 计算量分析

Prefill 的主要计算：

**注意力计算**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

- $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 的计算：$O(n \cdot d^2)$
- $\mathbf{Q}\mathbf{K}^\top$：$O(n^2 \cdot d)$
- $\text{softmax} \cdot \mathbf{V}$：$O(n^2 \cdot d)$

**FFN 计算**：

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$$

- $O(n \cdot d \cdot d_{ff})$，其中 $d_{ff}$ 通常是 $4d$

总计算量：$O(L \cdot (n^2 \cdot d + n \cdot d^2))$

对于长 prompt（$n$ 大），注意力的 $O(n^2)$ 成为瓶颈。

### TTFT 优化

**TTFT**（Time to First Token）是用户感知的首要延迟，主要由 Prefill 决定。优化方向：

1. **Flash Attention**：减少显存访问，加速注意力计算
2. **Tensor 并行**：多卡分担计算
3. **量化**：减少计算精度
4. **Prefix Caching**：缓存公共前缀的 KV

## 6.1.3 Decode 阶段

### 定义

**Decode**（解码）阶段逐个生成输出 token。每一步：

1. 将上一步生成的 token 输入模型
2. 计算该 token 的隐藏状态
3. 通过 LM head 得到下一个 token 的概率分布
4. 采样得到下一个 token
5. 重复直到生成结束符或达到最大长度

### 计算特点

**串行依赖**：每个 token 依赖前一个，无法并行生成（不考虑投机解码）。

**内存密集**：每步只处理一个 token，计算量小，但需要加载全部模型参数。

**重复开销**：生成 $m$ 个 token 需要 $m$ 次前向传播。

### 算术强度分析

**算术强度**（Arithmetic Intensity）= 计算量 / 数据传输量

设模型参数量为 $P$，每步 Decode：

- 计算量：$O(d^2 \cdot L)$（一个 token 的矩阵乘法）
- 数据传输量：$O(P)$（加载全部参数）

对于大模型，$P$ 很大，算术强度低，GPU 大部分时间在等待数据。这就是 Decode 阶段**内存带宽受限**（Memory-bound）的根本原因。

### 量化示例

以 LLaMA-70B 为例：

- 参数量：70B
- FP16 显存：140GB
- A100 显存带宽：2TB/s
- 加载模型一次：140GB / 2TB/s = 70ms

这意味着每个 token 至少需要 70ms（不计算开销），即最多 ~14 tokens/s。

实际上通过 KV Cache 和批处理可以改善，但内存带宽仍是核心瓶颈。

## 6.1.4 Prefill 与 Decode 的对比

| 维度 | Prefill | Decode |
|------|---------|--------|
| 处理内容 | Prompt（已知） | 输出（逐个生成） |
| 并行性 | 高（所有 token 并行） | 低（串行依赖） |
| 计算特点 | Compute-bound | Memory-bound |
| 主要瓶颈 | 注意力的 $O(n^2)$ | 内存带宽 |
| 批处理效果 | 好 | 有限 |

### 系统设计启示

这种差异对系统设计有深刻影响：

**分离调度**：Prefill 和 Decode 可以分开调度，甚至用不同的硬件。

**批处理策略**：Prefill 受益于大批量，Decode 需要其他优化（如 Continuous Batching）。

**资源分配**：Prefill 需要更多计算资源，Decode 需要更多内存带宽。

## 6.1.5 Chunked Prefill

### 长 Prompt 的问题

当 prompt 很长（如 100K token）时，Prefill 的 $O(n^2)$ 注意力计算和显存占用成为问题：

1. **显存溢出**：注意力矩阵 $n \times n$ 可能超出显存
2. **延迟尖峰**：长 Prefill 阻塞其他请求
3. **批处理困难**：不同长度的 prompt 难以高效批处理

### Chunked Prefill

**Chunked Prefill** 将长 prompt 分块处理：

1. 将 prompt 分成大小为 $C$ 的 chunk
2. 逐 chunk 进行 Prefill，累积 KV Cache
3. 最后一个 chunk 后进入 Decode

优势：
- 显存占用可控（$O(C^2)$ 而非 $O(n^2)$）
- 长 prompt 不会阻塞其他请求
- 可以与 Decode 请求混合调度

代价：
- 总计算量略增（chunk 边界的处理）
- 实现复杂度增加

## 6.1.6 Prefill-Decode 分离架构

### 动机

Prefill 和 Decode 对硬件的需求不同：

- Prefill：高算力，适合 GPU
- Decode：高带宽，适合多卡并行或定制硬件

将两者分离可以独立优化和扩展。

### 分离部署

**物理分离**：
- Prefill 服务器：高算力 GPU（如 H100）
- Decode 服务器：多卡并行或专用硬件
- 通过网络传递 KV Cache

**逻辑分离**：
- 同一 GPU 上，Prefill 和 Decode 请求分开调度
- Prefill 使用大 batch，Decode 使用 Continuous Batching

### Disaggregated Serving

**Disaggregated Serving** 是一种极端的分离架构：

1. Prefill 节点专门处理 prompt
2. KV Cache 通过高速网络（如 NVLink、InfiniBand）传递
3. Decode 节点专门生成输出

这种架构在大规模部署中可以显著提高资源利用率，但增加了系统复杂度。
