# 3.6 Embedding 与权重共享

分词将文本转换为整数序列，但神经网络处理的是连续向量。**嵌入层**（Embedding Layer）负责将离散的 token ID 映射为稠密的向量表示。本节讨论嵌入的数学原理、学习方式，以及现代大模型中广泛使用的权重共享技术。

## 3.6.1 词嵌入的基本概念

### 从 One-Hot 到稠密表示

最简单的 token 表示是 **One-Hot 编码**：设词表大小为 $V$，token $i$ 表示为长度 $V$ 的向量，第 $i$ 位为 1，其余为 0。

$$\mathbf{e}_i = [0, \ldots, 0, \underbrace{1}_{i}, 0, \ldots, 0]^\top \in \mathbb{R}^V$$

One-Hot 的问题：

1. **维度灾难**：$V$ 通常在 30,000-150,000，向量极其稀疏
2. **语义无关**：任意两个词的 One-Hot 向量正交，$\mathbf{e}_i^\top \mathbf{e}_j = 0$（$i \neq j$），无法反映语义相似性

### 嵌入矩阵

嵌入层本质上是一个查表操作。设嵌入维度为 $d$，嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{V \times d}$，其中第 $i$ 行 $\mathbf{E}[i]$ 是 token $i$ 的嵌入向量：

$$\text{Embedding}(i) = \mathbf{E}[i] \in \mathbb{R}^d$$

等价地，可以写成矩阵乘法：

$$\text{Embedding}(i) = \mathbf{e}_i^\top \mathbf{E}$$

但实际实现不会真的构造 One-Hot 向量再做矩阵乘法，而是直接索引（查表），效率更高。

### 嵌入空间的几何性质

好的嵌入具有**语义相似性反映为向量相似性**的特点：

- 语义相近的词在嵌入空间中距离近
- 语义关系可以表示为向量运算

著名的例子是 Word2Vec 中的类比关系：

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

这表明嵌入空间捕获了"性别"这一语义维度。

## 3.6.2 嵌入的学习

### 随机初始化与端到端学习

在 Transformer 中，嵌入矩阵通常随机初始化，然后与模型一起端到端训练。常见初始化方式：

**正态分布**：

$$\mathbf{E}[i, j] \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.02 \text{ 或 } 1/\sqrt{d}$$

**均匀分布**：

$$\mathbf{E}[i, j] \sim \text{Uniform}(-a, a), \quad a = \sqrt{3/d}$$

初始化的目标是让嵌入向量有合理的范数，避免训练初期的梯度问题。

### 预训练词向量

在 Transformer 之前，预训练词向量（如 Word2Vec、GloVe）是常见做法：

1. 在大规模语料上无监督训练词向量
2. 用训练好的词向量初始化嵌入矩阵
3. 在下游任务上微调（或冻结）

现代大语言模型通常不使用预训练词向量，而是从头训练嵌入。原因：

1. Transformer 足够强大，能从大规模语料中学到更好的表示
2. 子词分词与词级预训练向量不兼容
3. 预训练语料与下游任务可能存在领域差异

## 3.6.3 输出层与 Softmax

### 语言模型的输出

语言模型的输出层将隐藏状态映射到词表上的概率分布：

$$P(w | \text{context}) = \text{softmax}(\mathbf{W}_o \mathbf{h} + \mathbf{b}_o)$$

其中 $\mathbf{h} \in \mathbb{R}^d$ 是最后一层的隐藏状态，$\mathbf{W}_o \in \mathbb{R}^{V \times d}$ 是输出权重矩阵。

输出层的参数量为 $V \times d$，与嵌入层相同。

### Softmax 的计算

$$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^V \exp(z_j)}$$

对于大词表（$V = 100,000$），计算 softmax 需要对 100,000 个元素求指数和归一化，计算量可观。这也是训练大词表模型的瓶颈之一。

### 数值稳定性

直接计算 $\exp(z_i)$ 可能导致数值溢出。标准做法是减去最大值：

$$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i - \max_j z_j)}{\sum_{j=1}^V \exp(z_j - \max_j z_j)}$$

这在数学上等价，但数值稳定。

## 3.6.4 权重共享（Tied Embeddings）

### 输入-输出权重共享

**权重共享**（Weight Tying）指让输入嵌入矩阵 $\mathbf{E}$ 和输出权重矩阵 $\mathbf{W}_o$ 共享参数：

$$\mathbf{W}_o = \mathbf{E}$$

即，token $i$ 的嵌入向量 $\mathbf{E}[i]$ 同时也是预测 token $i$ 时的输出权重。

### 理论动机

从语义一致性的角度：

- 输入嵌入将 token 映射到语义空间
- 输出层将语义空间映射回 token

如果两个映射是"对称"的，那么共享权重是合理的。直觉上，如果模型的内部表示 $\mathbf{h}$ 与某个 token $i$ 的嵌入 $\mathbf{E}[i]$ 相似，那么输出层应该给 token $i$ 高概率。

数学上，共享权重后，输出 logits 变为：

$$z_i = \mathbf{E}[i]^\top \mathbf{h} = \text{sim}(\mathbf{E}[i], \mathbf{h}) \cdot \|\mathbf{E}[i]\| \cdot \|\mathbf{h}\|$$

这可以理解为隐藏状态与嵌入向量的相似度（加权）。

### 实践效果

Press 和 Wolf（2017）的研究表明，权重共享：

1. 减少约 $V \times d$ 参数（对于 $V = 50,000$, $d = 1024$，约 50M 参数）
2. 性能持平或略有提升
3. 提供正则化效果

### 现代模型的选择

| 模型 | 权重共享 |
|------|----------|
| GPT-2 | 是 |
| BERT | 是 |
| GPT-3 | 否 |
| LLaMA | 否 |
| Qwen | 否 |

早期模型普遍使用权重共享。但随着模型规模增大，权重共享带来的参数节省相对较小，且可能限制表达能力。大规模模型趋向于不共享权重。

## 3.6.5 嵌入的缩放

### 缩放因子

原始 Transformer 论文对嵌入进行了缩放：

$$\text{ScaledEmbedding}(i) = \sqrt{d} \cdot \mathbf{E}[i]$$

理由：嵌入向量与位置编码相加后输入模型。位置编码使用 $\sin/\cos$ 函数，值域在 $[-1, 1]$。而随机初始化的嵌入向量方差约为 $1/d$，范数约为 $\sqrt{d} \cdot 1/\sqrt{d} = 1$。

但嵌入向量的每个分量方差为 $1/d$，而位置编码每个分量的期望方差为 $0.5$。为了平衡两者的量级，将嵌入乘以 $\sqrt{d}$。

### 现代做法

现代大模型（如 LLaMA）通常不使用显式的嵌入缩放，而是通过调整初始化标准差来隐式控制：

$$\mathbf{E}[i, j] \sim \mathcal{N}(0, 1/\sqrt{d})$$

配合 RMSNorm 和合理的初始化，模型在训练初期保持稳定。

## 3.6.6 大词表的挑战

### 参数量

嵌入矩阵的参数量为 $V \times d$。对于大词表：

| 词表大小 | 嵌入维度 | 嵌入参数量 |
|----------|----------|------------|
| 32,000 | 4,096 | 131M |
| 100,000 | 4,096 | 410M |
| 150,000 | 4,096 | 614M |

对于 7B 参数的模型，100K 词表的嵌入层约占 6%。对于 1B 参数的模型，这个比例上升到 40%。

### 训练效率

大词表带来两个训练挑战：

1. **Softmax 计算**：对 100K 个 logits 计算 softmax，开销不小
2. **梯度更新**：每个 batch 只有少数 token 参与，大部分嵌入向量的梯度为零

### 优化技术

**负采样**（Negative Sampling）：训练时只计算正样本和少量负样本的 loss，而非对整个词表求 softmax。Word2Vec 使用此技术。

**分层 Softmax**（Hierarchical Softmax）：将词表组织成树结构，将 $O(V)$ 复杂度降到 $O(\log V)$。

**自适应 Softmax**：将词表按频率分组，高频词用完整 softmax，低频词用简化版本。

现代大模型通常直接使用完整 softmax，因为 GPU 的并行能力足以处理大词表，且上述近似方法会损失精度。

## 3.6.7 嵌入的可解释性

### 最近邻分析

给定一个词的嵌入，找到嵌入空间中最近的词：

```python
def nearest_neighbors(embedding_matrix, word_idx, k=5):
    query = embedding_matrix[word_idx]
    similarities = embedding_matrix @ query
    top_k = similarities.argsort()[-k:][::-1]
    return top_k
```

如果"狗"的最近邻是"猫"、"宠物"、"动物"，说明嵌入捕获了语义相似性。

### 类比测试

经典的词向量类比测试：

$$\mathbf{v}_a - \mathbf{v}_b + \mathbf{v}_c \approx \mathbf{v}_d$$

例如：`king - man + woman ≈ queen`

现代 Transformer 的嵌入（经过多层变换后）在类比测试上表现不如专门训练的 Word2Vec，但在下游任务上表现更好。这表明语言模型学到的表示不一定是"可线性类比的"，但捕获了更丰富的上下文信息。

### 嵌入投影

将高维嵌入投影到 2D/3D 可视化（如 t-SNE、UMAP）：

- 同一语义类别的词（如国家名、动物名）往往聚集在一起
- 语法功能词（介词、连词）与内容词分离
- 多义词可能出现在不同聚类的"边界"

这种可视化帮助理解嵌入空间的结构，但要注意投影会损失信息，不能过度解读。

## 3.6.8 位置信息与嵌入

Transformer 的嵌入需要结合位置信息。最简单的方式是将 token 嵌入与位置编码相加：

$$\mathbf{x}_i = \mathbf{E}[\text{token}_i] + \mathbf{P}[i]$$

其中 $\mathbf{P}$ 是位置编码矩阵（可学习或固定）。

这种相加方式意味着位置和语义信息在同一空间中混合。后续章节将详细讨论各种位置编码方法。
