# 6.2 KV Cache

**KV Cache** 是 LLM 推理中最基础也最重要的优化技术。它通过缓存历史 token 的 Key 和 Value，避免 Decode 阶段的重复计算，将生成每个 token 的复杂度从 $O(n)$ 降为 $O(1)$（相对于序列长度）。

## 6.2.1 为什么需要 KV Cache

### 注意力计算回顾

自注意力的计算：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

对于因果语言模型，位置 $t$ 的输出只依赖位置 $1, 2, \ldots, t$ 的信息。

### 朴素实现的冗余

考虑生成第 $t+1$ 个 token。朴素实现需要：

1. 计算位置 $1$ 到 $t+1$ 的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$
2. 计算完整的注意力矩阵 $(t+1) \times (t+1)$
3. 得到位置 $t+1$ 的输出

但位置 $1$ 到 $t$ 的 $\mathbf{K}, \mathbf{V}$ 在生成前 $t$ 个 token 时已经计算过。重复计算是巨大的浪费。

### KV Cache 的思想

**KV Cache** 的核心思想：缓存所有历史位置的 $\mathbf{K}, \mathbf{V}$，每步只计算新 token 的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$。

生成第 $t+1$ 个 token 时：

1. 只计算位置 $t+1$ 的 $\mathbf{q}_{t+1}, \mathbf{k}_{t+1}, \mathbf{v}_{t+1}$
2. 将 $\mathbf{k}_{t+1}, \mathbf{v}_{t+1}$ 追加到 Cache
3. 用 $\mathbf{q}_{t+1}$ 与缓存的 $\mathbf{K}_{1:t+1}, \mathbf{V}_{1:t+1}$ 计算注意力

## 6.2.2 KV Cache 的实现

### 数据结构

对于每一层，维护两个张量：

$$\mathbf{K}_{\text{cache}} \in \mathbb{R}^{B \times H \times L_{\max} \times d_k}$$
$$\mathbf{V}_{\text{cache}} \in \mathbb{R}^{B \times H \times L_{\max} \times d_v}$$

其中：
- $B$：批大小
- $H$：注意力头数
- $L_{\max}$：最大序列长度
- $d_k, d_v$：Key/Value 维度

### Prefill 阶段

Prefill 时，一次性计算所有 prompt token 的 $\mathbf{K}, \mathbf{V}$ 并存入 Cache：

```python
# 计算 Q, K, V
Q = x @ W_Q  # [B, n, d]
K = x @ W_K  # [B, n, d]
V = x @ W_V  # [B, n, d]

# 存入 Cache
kv_cache.K[:, :, :n, :] = K
kv_cache.V[:, :, :n, :] = V

# 计算注意力
attn_output = attention(Q, K, V)
```

### Decode 阶段

Decode 时，每步只计算一个 token：

```python
# 计算新 token 的 Q, K, V
q = x_new @ W_Q  # [B, 1, d]
k = x_new @ W_K  # [B, 1, d]
v = x_new @ W_V  # [B, 1, d]

# 追加到 Cache
kv_cache.K[:, :, t, :] = k
kv_cache.V[:, :, t, :] = v

# 用完整的 K, V 计算注意力
K_full = kv_cache.K[:, :, :t+1, :]  # [B, H, t+1, d]
V_full = kv_cache.V[:, :, :t+1, :]  # [B, H, t+1, d]
attn_output = attention(q, K_full, V_full)  # [B, 1, d]
```

### 计算量对比

| 阶段 | 无 KV Cache | 有 KV Cache |
|------|------------|-------------|
| Prefill（$n$ tokens） | $O(n^2 \cdot d)$ | $O(n^2 \cdot d)$ |
| Decode（每个 token） | $O(t^2 \cdot d)$ | $O(t \cdot d)$ |
| Decode（$m$ tokens 总计） | $O(m \cdot n^2 \cdot d)$ | $O(m \cdot n \cdot d)$ |

KV Cache 将 Decode 的复杂度从二次降为线性。

## 6.2.3 显存占用分析

### KV Cache 大小

设模型有 $L$ 层，隐藏维度 $d$，序列长度 $n$，批大小 $B$。

每层 KV Cache：$2 \times n \times d$ 个元素（K 和 V）

总 KV Cache：$2 \times L \times B \times n \times d$ 个元素

以 LLaMA-70B 为例（$L=80$，$d=8192$，$n=4096$，$B=1$，FP16）：

$$2 \times 80 \times 1 \times 4096 \times 8192 \times 2 \text{ bytes} = 10.7 \text{ GB}$$

**单个请求的 KV Cache 就占用 10.7GB**！这是 LLM 推理显存紧张的主要原因。

### 与模型参数的对比

LLaMA-70B 参数量：140GB（FP16）

KV Cache（4K 上下文）：10.7GB / 请求

批大小 8 时，KV Cache 总计：85.6GB，已接近模型参数量。

### GQA 对 KV Cache 的影响

**GQA**（Grouped Query Attention）让多个 Query 头共享一组 KV 头，减少 KV Cache 大小。

设 Query 头数为 $H_Q$，KV 头数为 $H_{KV}$：

$$\text{KV Cache} = 2 \times L \times B \times n \times H_{KV} \times d_k$$

LLaMA-2 70B 使用 GQA（$H_Q=64$，$H_{KV}=8$），KV Cache 缩小为 MHA 的 $1/8$。

## 6.2.4 KV Cache 优化技术

### 量化

将 KV Cache 从 FP16 量化到 INT8 或 INT4：

- INT8：显存减半
- INT4：显存减少 75%

量化可能影响生成质量，需要在精度和效率之间权衡。

### 压缩

**Key/Value 压缩**：用低秩分解或学习到的压缩函数减少 KV 维度。

**稀疏化**：只保留重要位置的 KV（如 H2O、StreamingLLM）。

### Sliding Window

**滑动窗口注意力**：只保留最近 $W$ 个位置的 KV Cache。

$$\mathbf{K}_{\text{cache}} = \mathbf{K}_{t-W+1:t}$$

Mistral 等模型原生支持滑动窗口，适合长序列场景。

### Prefix Caching

对于共享前缀的多个请求（如 few-shot 或系统提示），可以缓存前缀的 KV：

1. 首次计算公共前缀的 KV
2. 后续请求直接复用，跳过前缀的 Prefill

这在多轮对话和批量处理中非常有效。

## 6.2.5 Multi-Query 与 Grouped-Query Attention

### MHA 的冗余

标准 **MHA**（Multi-Head Attention）每个头有独立的 $\mathbf{W}_K, \mathbf{W}_V$，产生独立的 KV Cache。

但研究发现，不同头的 KV 有很高的相似性，存在冗余。

### MQA

**MQA**（Multi-Query Attention）所有 Query 头共享一组 KV：

$$H_{KV} = 1$$

KV Cache 大小降为 MHA 的 $1/H$，显著减少显存。

代价：生成质量可能略有下降。

### GQA

**GQA**（Grouped-Query Attention）是 MHA 和 MQA 的折中：

$$1 < H_{KV} < H_Q$$

将 Query 头分组，每组共享一组 KV。

例如 $H_Q = 64$，$H_{KV} = 8$：每 8 个 Query 头共享一组 KV。

GQA 在质量和效率之间取得了很好的平衡，被 LLaMA-2、Mistral 等主流模型采用。

### 转换

已有的 MHA 模型可以转换为 GQA/MQA：

1. 对每组 Query 头，将对应的 $\mathbf{W}_K, \mathbf{W}_V$ 取平均
2. 在小数据集上微调恢复质量

## 6.2.6 KV Cache 的内存管理

### 预分配 vs 动态分配

**预分配**：按最大长度 $L_{\max}$ 预分配 KV Cache。

- 优点：简单，无碎片
- 缺点：浪费（大多数请求不会达到最大长度）

**动态分配**：按实际长度分配。

- 优点：节省显存
- 缺点：内存碎片，分配开销

### 碎片问题

动态分配会导致内存碎片：

```
|--KV1--|    |--KV2--|      |--KV3--|
        ^空隙^        ^空隙^
```

短请求结束释放后，留下的空隙可能无法容纳新的长请求。

### Paged Attention

**Paged Attention**（下一节详述）借鉴操作系统的分页内存管理，将 KV Cache 分成固定大小的"页"，动态分配和回收。这是解决 KV Cache 内存管理的核心技术。
