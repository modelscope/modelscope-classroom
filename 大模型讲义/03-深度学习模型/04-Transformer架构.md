# 3.4 Transformer 架构：编码器与解码器

2017 年 Vaswani 等人发表的《Attention Is All You Need》开创了深度学习的新纪元。Transformer 架构彻底抛弃了循环结构，完全基于注意力机制建模序列依赖，不仅在性能上超越了 LSTM，更因其高度可并行化而在大规模训练中展现出巨大优势。本节介绍 Transformer 的整体架构，后续章节将逐一深入各个组件。

## 3.4.1 从 Seq2Seq 到 Transformer

### Encoder-Decoder 范式

序列到序列（Seq2Seq）任务——如机器翻译——需要将输入序列映射到输出序列。经典的编码器-解码器（Encoder-Decoder）架构包含两个部分：

**编码器**（Encoder）：将输入序列 $\mathbf{x}_1, \ldots, \mathbf{x}_n$ 编码为上下文表示。RNN 时代，编码器的最终隐藏状态 $\mathbf{h}_n$ 作为"上下文向量"传递给解码器。

**解码器**（Decoder）：基于上下文表示，自回归地生成输出序列 $\mathbf{y}_1, \ldots, \mathbf{y}_m$。每一步生成依赖于上下文和已生成的部分序列。

RNN-based Seq2Seq 的瓶颈在于：无论输入多长，所有信息都被压缩进固定维度的 $\mathbf{h}_n$。这个"信息瓶颈"严重限制了长序列的处理能力。

### 注意力机制的引入

Bahdanau 等人（2014）提出**注意力机制**（Attention Mechanism）来缓解信息瓶颈。解码器在每一步不再只依赖 $\mathbf{h}_n$，而是动态地"关注"编码器的所有隐藏状态：

$$\mathbf{c}_t = \sum_{i=1}^n \alpha_{t,i} \mathbf{h}_i$$

其中 $\alpha_{t,i}$ 是注意力权重，表示解码第 $t$ 步时对输入位置 $i$ 的关注程度。

这是一个重要的范式转变：模型可以在每一步"回看"整个输入，而非只看一个压缩后的向量。

### Transformer 的革新

Transformer 将注意力机制推向极致：不仅解码器用注意力关注编码器，编码器内部也完全用注意力取代了循环。这就是**自注意力**（Self-Attention）。

关键创新点：

1. **完全基于注意力**：序列内任意两个位置可以直接交互，无需逐步传递
2. **高度可并行**：所有位置的计算可以同时进行，GPU 效率大幅提升
3. **模块化设计**：编码器和解码器都是相同模块的堆叠

## 3.4.2 Transformer 整体结构

### 编码器

Transformer 编码器由 $N$ 个相同的层堆叠而成。每一层包含两个子层：

1. **多头自注意力**（Multi-Head Self-Attention）
2. **前馈网络**（Feed-Forward Network, FFN）

每个子层都有**残差连接**（Residual Connection）和**层归一化**（Layer Normalization）：

$$\text{SubLayer}(\mathbf{x}) = \text{LayerNorm}(\mathbf{x} + f(\mathbf{x}))$$

设输入序列为 $\mathbf{X} \in \mathbb{R}^{n \times d}$，编码器第 $l$ 层的计算：

$$\mathbf{Z}^{(l)} = \text{LayerNorm}(\mathbf{X}^{(l-1)} + \text{MultiHeadAttn}(\mathbf{X}^{(l-1)}))$$
$$\mathbf{X}^{(l)} = \text{LayerNorm}(\mathbf{Z}^{(l)} + \text{FFN}(\mathbf{Z}^{(l)}))$$

### 解码器

解码器同样由 $N$ 个相同的层堆叠，每一层包含三个子层：

1. **掩码多头自注意力**（Masked Multi-Head Self-Attention）
2. **编码器-解码器注意力**（Cross-Attention）
3. **前馈网络**（FFN）

**掩码自注意力**确保解码器在预测位置 $t$ 时只能看到位置 $1, \ldots, t-1$，而非未来的位置。这通过将注意力矩阵的上三角部分设为 $-\infty$ 实现。

**编码器-解码器注意力**让解码器"关注"编码器的输出：Query 来自解码器的上一子层，Key 和 Value 来自编码器的最终输出。

### 输入输出处理

**输入嵌入**：将离散的 token 映射为连续向量。设词表大小为 $V$，嵌入维度为 $d$，嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{V \times d}$。

**位置编码**：Transformer 没有循环，无法感知位置顺序。位置编码（Positional Encoding）为每个位置注入位置信息。原始 Transformer 使用正弦-余弦编码：

$$PE_{pos, 2i} = \sin(pos / 10000^{2i/d})$$
$$PE_{pos, 2i+1} = \cos(pos / 10000^{2i/d})$$

**输出层**：解码器的输出经过线性变换和 softmax，得到下一个 token 的概率分布：

$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o)$$

### 权重共享

原始 Transformer 在三处共享权重：

1. 编码器和解码器的输入嵌入矩阵
2. 解码器的输入嵌入和输出层权重（tied embeddings）

权重共享减少了参数量，也提供了正则化效果。

## 3.4.3 三种 Transformer 变体

### Encoder-only（编码器）

代表：BERT、RoBERTa

只使用编码器，适合理解任务（分类、序列标注、问答抽取）。输入序列双向可见——每个位置可以"看到"所有其他位置。

BERT 的预训练目标是**掩码语言模型**（Masked Language Model, MLM）：随机遮盖 15% 的 token，让模型预测被遮盖的内容。

### Decoder-only（解码器）

代表：GPT 系列、LLaMA、Qwen

只使用解码器，适合生成任务。使用**因果掩码**（Causal Mask）：位置 $t$ 只能看到位置 $1, \ldots, t-1$。

预训练目标是**下一个 token 预测**（Next Token Prediction）：

$$\mathcal{L} = -\sum_{t=1}^T \log P(x_t | x_{<t})$$

Decoder-only 架构已成为大语言模型的主流选择。原因包括：

1. 统一的训练和推理范式（都是自回归生成）
2. 大规模文本数据天然适合语言建模目标
3. 涌现能力（emergent abilities）在 decoder-only 模型上更显著

### Encoder-Decoder（编解码器）

代表：T5、BART、mBART

完整的编解码器结构，适合序列到序列任务（翻译、摘要、问答生成）。

T5（Text-to-Text Transfer Transformer）将所有 NLP 任务统一为 text-to-text 格式。例如，情感分类变成：

- 输入："sentiment: This movie is great"
- 输出："positive"

## 3.4.4 Pre-Norm vs Post-Norm

层归一化的位置有两种选择：

### Post-Norm（原始 Transformer）

$$\mathbf{y} = \text{LayerNorm}(\mathbf{x} + f(\mathbf{x}))$$

归一化在残差相加之后。

### Pre-Norm（现代大模型）

$$\mathbf{y} = \mathbf{x} + f(\text{LayerNorm}(\mathbf{x}))$$

归一化在子层计算之前。

**实践发现**：Pre-Norm 在深层网络中更稳定。直观理解：Pre-Norm 的残差分支直接是 $f(\cdot)$，而 Post-Norm 的残差分支是 $\text{LayerNorm}(\mathbf{x} + f(\mathbf{x})) - \mathbf{x}$，后者的梯度流更复杂。

GPT-2 及之后的大模型普遍采用 Pre-Norm。代价是：Pre-Norm 的最终输出需要额外的 LayerNorm（在输出层之前），否则输出分布不稳定。

## 3.4.5 计算复杂度分析

### 自注意力的复杂度

对于长度为 $n$ 的序列，自注意力需要计算 $n \times n$ 的注意力矩阵：

- 时间复杂度：$O(n^2 \cdot d)$
- 空间复杂度：$O(n^2)$（存储注意力矩阵）

当 $n$ 很大时（如 $n = 32,768$），$n^2 = 10^9$，这是严重的瓶颈。后续章节将讨论线性注意力等改进方法。

### FFN 的复杂度

FFN 对每个位置独立计算，复杂度与序列长度线性相关：

- 时间复杂度：$O(n \cdot d \cdot d_{\text{ff}})$
- 空间复杂度：$O(d \cdot d_{\text{ff}})$（权重矩阵）

当 $d_{\text{ff}} = 4d$ 时，FFN 的计算量约为 $8nd^2$，与自注意力（约 $4n^2 d$）相当。当 $n > 2d$ 时，自注意力成为瓶颈；当 $n < 2d$ 时，FFN 是主要计算量。

### KV Cache

推理时的自回归生成需要反复计算注意力。朴素实现中，生成第 $t$ 个 token 需要重新计算前 $t-1$ 个位置的 Key 和 Value。

**KV Cache** 优化：缓存已计算的 Key 和 Value 向量。生成第 $t$ 个 token 时，只需计算位置 $t$ 的 Query，然后与缓存的 $K_{1:t-1}$、$V_{1:t-1}$ 计算注意力。

KV Cache 将生成复杂度从 $O(T^2)$ 降到 $O(T)$，但需要 $O(T \cdot d)$ 的显存存储缓存。对于长序列和大批次，KV Cache 的显存占用可能成为瓶颈。

## 3.4.6 Transformer 的 PyTorch 实现骨架

以下是 Decoder-only Transformer 的简化实现：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm style
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

这个骨架展示了现代大语言模型的核心结构。后续章节将详细介绍 `MultiHeadAttention`、`FeedForward` 等组件的实现细节。
