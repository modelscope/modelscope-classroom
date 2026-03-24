# AMD生态与ROCm

AMD GPU通过ROCm（Radeon Open Compute）平台为AI计算提供支持，是CUDA之外的重要替代选择。

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

ROCm是AMD的开源计算平台，提供类似CUDA的编程模型：

- **HIP**：类CUDA的C++编程接口
- **rocBLAS**：BLAS线性代数库
- **MIOpen**：深度学习原语库
- **RCCL**：集合通信库（类NCCL）

### 版本兼容

```
ROCm 6.0 → PyTorch 2.2+
ROCm 5.7 → PyTorch 2.0/2.1
ROCm 5.4 → PyTorch 1.13
```

## 环境配置

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

```bash
# 从PyTorch官方安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 验证
python -c "import torch; print(torch.cuda.is_available())"  # ROCm兼容CUDA接口
```

## 基础使用

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

HIP提供与CUDA高度相似的API：

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

使用hipify工具将CUDA代码转换为HIP：

```bash
# 自动转换CUDA代码
hipify-perl cuda_code.cu > hip_code.cpp

# 或使用hipify-clang
hipify-clang cuda_code.cu -o hip_code.cpp
```

大多数CUDA API有直接对应：

| CUDA | HIP |
|------|-----|
| cudaMalloc | hipMalloc |
| cudaMemcpy | hipMemcpy |
| cudaFree | hipFree |
| cudaDeviceSynchronize | hipDeviceSynchronize |

## 框架支持

### PyTorch

```python
# 与CUDA代码完全相同
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

```bash
# vLLM支持ROCm
pip install vllm  # 确保安装ROCm版本

# 启动服务（与CUDA相同）
python -m vllm.entrypoints.openai.api_server --model model_path
```

## 分布式训练

### RCCL

RCCL是AMD的集合通信库，API与NCCL兼容：

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

### 兼容性问题

```python
# 某些CUDA扩展可能不支持
# 检查是否有ROCm版本或使用hipify转换

# 设置环境变量强制使用HIP
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # 针对特定GPU
```

### 性能调优

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

ROCm为AI计算提供了CUDA之外的重要选择。虽然生态成熟度还在追赶中，但对于主流深度学习框架的支持已经较为完善。AMD MI系列在性价比和显存容量上具有优势，适合特定的训练和推理场景。
