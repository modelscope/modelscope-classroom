# 7.2 Triton 优化与算子融合

**Triton** 是 OpenAI 开发的 GPU 编程语言和编译器，它让开发者能够以接近 Python 的简洁语法编写高效的 GPU 算子。结合**算子融合**（Operator Fusion），Triton 成为大模型优化的利器。

## 7.2.1 GPU 编程的挑战

### CUDA 的复杂性

传统 CUDA 编程需要处理：

- 线程块、线程束（warp）的组织
- 共享内存的分配和同步
- 内存访问模式的优化（合并访问、bank conflict）
- 寄存器分配
- 指令级并行

编写高效的 CUDA kernel 需要深厚的硬件知识，门槛很高。

### 框架算子的局限

PyTorch 等框架提供了丰富的算子库，但：

1. **Kernel Launch 开销**：每个算子一次 kernel launch，累积开销大
2. **中间结果**：算子之间需要写回显存、再读取
3. **灵活性不足**：框架算子是预定义的，特殊需求难以满足

## 7.2.2 Triton 简介

### 设计理念

Triton 的设计理念是**块级编程**（Block-Level Programming）：

- 程序员以"块"（block）为单位思考，而非单个线程
- 编译器自动处理线程级细节
- 内存访问模式自动优化

### 基本语法

Triton 使用 Python 语法装饰器定义 kernel：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一个 block
    pid = tl.program_id(0)
    
    # 计算该 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    output = x + y
    
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 编译与执行

Triton kernel 在首次调用时 JIT 编译为 PTX/CUDA：

```python
# 启动 kernel
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

编译器自动优化内存访问、寄存器分配等。

## 7.2.3 算子融合

### 融合的收益

**算子融合**（Operator Fusion）将多个连续算子合并为一个 kernel，收益：

1. **减少 kernel launch**：N 个 launch → 1 个 launch
2. **减少显存访问**：中间结果保持在寄存器/共享内存
3. **提高算术强度**：计算/访存比提升

### 融合示例

未融合：
```python
# 3 次 kernel launch，2 次中间结果写回显存
y = x * scale  # kernel 1
y = y + bias   # kernel 2
y = gelu(y)    # kernel 3
```

融合后：
```python
@triton.jit
def fused_scale_bias_gelu(x_ptr, scale_ptr, bias_ptr, y_ptr, ...):
    # 一次 kernel 完成全部计算
    x = tl.load(x_ptr + offsets)
    scale = tl.load(scale_ptr + offsets)
    bias = tl.load(bias_ptr + offsets)
    
    y = x * scale + bias
    y = gelu(y)
    
    tl.store(y_ptr + offsets, y)
```

### 常见融合模式

| 融合模式 | 原始算子 | 融合后 |
|----------|----------|--------|
| Bias + Activation | Linear → Add → GELU | FusedLinear |
| LayerNorm | Mean → Sub → Var → Div → Scale | FusedLayerNorm |
| Softmax | Max → Sub → Exp → Sum → Div | FusedSoftmax |
| Attention | QK → Softmax → V | FlashAttention |

## 7.2.4 Triton 实战：Fused Softmax

### 标准 Softmax

```python
def naive_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values
    x = x - x_max  # 数值稳定
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

这需要 4 次 kernel launch，多次显存读写。

### Triton Fused Softmax

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 加载一行
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # 计算 softmax（在寄存器中完成）
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_exp
    
    # 存储
    tl.store(output_ptr + row_start + col_offsets, softmax_row, mask=mask)
```

一次 kernel 完成全部计算，中间结果不离开寄存器。

## 7.2.5 Triton 在大模型中的应用

### Flash Attention

Flash Attention 的 Triton 实现展示了复杂算子的编写能力：

- 分块加载 Q、K、V
- 在线 softmax 计算
- 累积输出结果

Triton 版 Flash Attention 与 CUDA 版性能接近，但代码量少得多。

### RMS Norm

```python
@triton.jit
def rms_norm_kernel(x_ptr, weight_ptr, out_ptr, eps, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * N
    out_row_ptr = out_ptr + row * N
    
    # 加载并计算
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq) / N
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize
    out = x / rms * weight
    tl.store(out_row_ptr + offsets, out, mask=mask)
```

### SwiGLU

```python
@triton.jit
def swiglu_kernel(x_ptr, w1_ptr, w2_ptr, out_ptr, ...):
    # 融合 gate 和 up projection
    gate = tl.load(x_ptr + ...) @ tl.load(w1_ptr + ...)
    up = tl.load(x_ptr + ...) @ tl.load(w2_ptr + ...)
    
    # SiLU(gate) * up
    out = (gate * tl.sigmoid(gate)) * up
    tl.store(out_ptr + ..., out)
```

## 7.2.6 torch.compile 与算子融合

### 自动融合

PyTorch 2.0 的 `torch.compile` 可以自动进行算子融合：

```python
@torch.compile
def forward(x):
    x = self.norm(x)
    x = self.linear(x)
    x = F.gelu(x)
    return x
```

编译器会自动识别融合机会，生成优化的 kernel。

### TorchInductor

`torch.compile` 的默认后端 **TorchInductor** 可以生成 Triton kernel：

1. 将 PyTorch 代码转换为中间表示（IR）
2. 识别融合模式
3. 生成 Triton 代码
4. JIT 编译执行

### 与手写 Triton 的对比

| 维度 | torch.compile | 手写 Triton |
|------|---------------|-------------|
| 开发效率 | 高（自动） | 低（手动） |
| 性能 | 好 | 最优 |
| 灵活性 | 受限 | 完全控制 |
| 调试 | 困难 | 直接 |

对于关键路径（如注意力），手写 Triton 仍有价值；对于常规计算，`torch.compile` 足够。

## 7.2.7 实践建议

### 何时使用 Triton

1. **热点算子**：profiling 显示的耗时大户
2. **特殊计算模式**：框架没有提供的融合算子
3. **内存敏感**：显存是瓶颈时，融合可以减少中间结果

### 优化技巧

1. **选择合适的块大小**：通常是 2 的幂，需要调优
2. **利用共享内存**：跨线程复用的数据放入共享内存
3. **避免 bank conflict**：共享内存访问模式要注意
4. **向量化加载**：使用 `tl.load` 的向量化功能

### 调试方法

1. **对比基准**：与 PyTorch 实现对比输出
2. **数值稳定性**：检查 inf/nan
3. **性能分析**：使用 `triton.testing.do_bench` 计时
4. **CUDA 调试器**：Nsight Compute 分析 kernel

```python
# 性能测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 15)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        ...
    )
)
def benchmark(N, provider):
    ...
```
