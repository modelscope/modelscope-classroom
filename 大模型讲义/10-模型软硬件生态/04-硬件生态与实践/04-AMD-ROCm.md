# AMD生态与ROCm

AMD GPU通过ROCm（Radeon Open Compute）平台为AI计算提供支持，是CUDA之外的重要替代选择。

你可能会问：既然NVIDIA生态这么成熟，为什么还要关注AMD？原因很实际——首先是显存优势，MI300X提供192GB HBM3，是H100的80GB的两倍多，这意味着单卡就能装下70B参数模型的完整权重；其次是性价比，AMD在采购价格和可获得性上往往有优势，尤其在“GPU荒”时期；最后，ROCm是开源平台，适合需要对底层有控制权的团队。当然，生态成熟度确实还在追赶，本节帮你判断ROCm是否适合你的场景。

## 硬件概览

### AMD Instinct系列

| GPU | 显存 | FP16算力 | 互联 |
|-----|------|---------|------|
| MI300X | 192GB HBM3 | 1307 TFLOPS | Infinity Fabric |
| MI250X | 128GB HBM2e | 383 TFLOPS | Infinity Fabric |
| MI210 | 64GB HBM2e | 181 TFLOPS | Infinity Fabric |
| MI100 | 32GB HBM2 | 184 TFLOPS | Infinity Fabric |

### 消费级GPU

| GPU | 显存 | 适用 |
|-----|------|------|
| RX 7900 XTX | 24GB | 开发/小模型 |
| RX 7900 XT | 20GB | 开发/推理 |

## ROCm平台

### 概述

ROCm是AMD的开源计算平台，提供类似CUDA的编程模型。对于习惯CUDA生态的开发者来说，ROCm的设计哲学就是"尽可能兼容"——每个CUDA组件都有直接对应的ROCm等价物：

- **HIP**：类CUDA的C++编程接口，且API命名几乎是一一对应（`cudaMalloc` → `hipMalloc`）
- **rocBLAS**：BLAS线性代数库，对标cuBLAS
- **MIOpen**：深度学习原语库，对标cuDNN
- **RCCL**：集合通信库，对标NCCL，而且在PyTorch中直接用`backend='nccl'`就行，RCCL会自动接管

### 版本兼容

```
ROCm 6.0 → PyTorch 2.2+
ROCm 5.7 → PyTorch 2.0/2.1
ROCm 5.4 → PyTorch 1.13
```

## 环境配置

相比昇腾环境的手动安装流程，ROCm的安装体验更接近CUDA——通过包管理器就能完成，且PyTorch官方直接提供ROCm版本的预编译包。

### 安装ROCm

```bash
# Ubuntu 22.04
# 添加仓库
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# 安装ROCm
sudo amdgpu-install --usecase=rocm

# 添加用户到组
sudo usermod -a -G render,video $USER

# 重启后验证
rocm-smi
```

### 安装PyTorch

这里有一个让很多人意外的设计决策：ROCm版本的PyTorch完全复用了CUDA的接口。也就是说，你的代码中仍然写`torch.cuda.is_available()`，它会返回True并识别出AMD GPU。这种设计的好处是现有代码基本不用改，但调试时偏尔会因为"明明是AMD卡但代码写着cuda"而困惑。

```bash
# 从PyTorch官方安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 验证
python -c "import torch; print(torch.cuda.is_available())"  # ROCm兼容CUDA接口
```

## 基础使用

理解了ROCm的"兼容性设计"后，实际使用就非常简单了——你的PyTorch代码几乎不需要任何修改。

### 设备管理

```python
import torch

# ROCm使用CUDA兼容接口
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # 显示AMD GPU

# 使用方式与CUDA相同
device = torch.device("cuda:0")
model = model.to(device)
```

### HIP编程

HIP提供与CUDA高度相似的API。在实际项目中，大多数开发者不需要直接写HIP代码（因为PyTorch已经封装好了），但如果你的项目中有自定义的CUDA kernel，就需要用HIP重写或用hipify工具转换：

```cpp
// HIP代码（类似CUDA）
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // 内存分配
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);
    
    // 启动kernel
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threads), 0, 0,
                       d_a, d_b, d_c, n);
    
    hipFree(d_a);
    return 0;
}
```

### CUDA代码迁移

假设你有一个包含自定义CUDA kernel的项目，使用hipify工具可以自动完成大部分转换工作：

```bash
# 自动转换CUDA代码
hipify-perl cuda_code.cu > hip_code.cpp

# 或使用hipify-clang
hipify-clang cuda_code.cu -o hip_code.cpp
```

大多数CUDA API有直接对应，这也是hipify能做到高转化率的原因：

| CUDA | HIP |
|------|-----|
| cudaMalloc | hipMalloc |
| cudaMemcpy | hipMemcpy |
| cudaFree | hipFree |
| cudaDeviceSynchronize | hipDeviceSynchronize |

## 框架支持

在实际工程中，你很少需要直接写HIP代码。主流深度学习框架已经对ROCm有了较好的支持，而且得益于兼容性设计，代码通常不需要任何修改。

### PyTorch

```python
# 与CUDA代码完全相同——这就是ROCm兼容性设计的好处
import torch

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    x, y = batch
    x, y = x.cuda(), y.cuda()
    
    loss = model(x, y)
    loss.backward()
    optimizer.step()
```

### Transformers

```python
from transformers import AutoModelForCausalLM

# 直接使用，无需修改
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    device_map="auto"  # 自动识别AMD GPU
)
```

### vLLM

在实际场景中，vLLM对ROCm的支持意义重大——如果你用MI300X做模型服务，单卡192GB显存意味着可以装下70B模型而不需要多卡并行，部署复杂度大幅降低：

```bash
# vLLM支持ROCm
pip install vllm  # 确保安装ROCm版本

# 启动服务（与CUDA相同）
python -m vllm.entrypoints.openai.api_server --model model_path
```

## 分布式训练

### RCCL

RCCL是AMD的集合通信库，API与NCCL兼容。这里的兼容性做得非常彻底——你甚至不需要把`backend='nccl'`改成别的，RCCL会透明地替代NCCL：

```python
import torch.distributed as dist

# 初始化（使用nccl后端，RCCL会自动接管）
dist.init_process_group(backend='nccl')

# 使用方式与NVIDIA相同
model = torch.nn.parallel.DistributedDataParallel(model)
```

### 启动训练

```bash
# torchrun方式
torchrun --nproc_per_node=8 train.py
```

## 监控工具

### rocm-smi

```bash
# 查看GPU状态
rocm-smi

# 显示详细信息
rocm-smi --showallinfo

# 监控
watch -n 1 rocm-smi

# 查看特定指标
rocm-smi --showmeminfo vram
rocm-smi --showuse
```

### 性能分析

```bash
# rocprof性能分析
rocprof --stats python train.py

# 生成trace
rocprof --sys-trace python train.py
```

## 常见问题

在实际使用中，ROCm的主要问题集中在第三方库的兼容性上——PyTorch核心没问题，但许多基于CUDA的扩展库可能需要额外处理。

### 兼容性问题

你可能遇到过这种情况：某个CUDA扩展库（比如一些自定义的注意力实现）直接pip install后在AMD卡上报错。这时需要检查该库是否提供ROCm版本，或者尝试用hipify转换：

```python
# 某些CUDA扩展可能不支持
# 检查是否有ROCm版本或使用hipify转换

# 设置环境变量强制使用HIP
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # 针对特定GPU
```

### 性能调优

MIOpen需要在首次运行时对算子进行自动调优（auto-tuning），这会导致第一次训练明显较慢。的策略是将调优结果缓存起来：

```bash
# 环境变量
export MIOPEN_FIND_ENFORCE=3  # MIOpen调优
export HIP_VISIBLE_DEVICES=0,1,2,3  # 指定可见设备
export GPU_MAX_HW_QUEUES=8  # 硬件队列数
```

### Flash Attention

```bash
# AMD版Flash Attention
pip install flash-attn --no-build-isolation  # 需要ROCm支持
```

## 与CUDA生态的对比

| 特性 | CUDA | ROCm |
|-----|------|------|
| 生态成熟度 | 非常成熟 | 持续完善 |
| 框架支持 | 广泛 | 主流支持 |
| 文档质量 | 完善 | 较好 |
| 社区支持 | 大 | 中等 |
| 代码迁移 | - | 较易 |

ROCm为AI计算提供了CUDA之外的重要选择。从PyTorch用户的角度看，ROCm的最大优势是兼容性设计——现有代码几乎不需要修改。MI300X的192GB显存在大模型推理场景中优势明显，单卡就能服务70B级别的模型。主要的挑战在于第三方CUDA扩展库的兼容性，以及某些场景下的性能调优需要额外努力。如果你的项目主要用主流框架（PyTorch + Transformers + vLLM），且对显存容量有较高要求，AMD MI系列是很值得考虑的方案。
