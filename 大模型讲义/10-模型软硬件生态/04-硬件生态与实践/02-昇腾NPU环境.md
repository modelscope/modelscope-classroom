# 昇腾NPU环境配置与使用

华为昇腾（Ascend）NPU是国产AI加速器的代表，在大模型训练和推理场景中展现出竞争力。本节介绍昇腾NPU的环境配置方法。

## 硬件概览

### 昇腾芯片系列

| 芯片 | 定位 | AI算力 | 显存 |
|-----|------|--------|------|
| 昇腾910B | 训练 | 256 TFLOPS (FP16) | 64GB HBM |
| 昇腾910A | 训练 | 256 TFLOPS (FP16) | 32GB HBM |
| 昇腾310P | 推理 | 22 TFLOPS (FP16) | 24GB |
| 昇腾310 | 推理 | 16 TFLOPS (INT8) | 8GB |

### 服务器配置

- **Atlas 800 训练服务器**：8卡910B配置
- **Atlas 300I Pro**：推理卡
- **Atlas 200 DK**：开发套件

## 软件栈

### CANN（Compute Architecture for Neural Networks）

CANN是昇腾的核心软件栈，包含：

- **AscendCL**：底层计算接口
- **算子库**：cuDNN等价层
- **图引擎**：计算图优化
- **调试工具**：性能分析

### 软件版本对应

```
CANN 8.0 → PyTorch 2.1/2.2
CANN 7.0 → PyTorch 2.0/2.1
CANN 6.0 → PyTorch 1.11
```

## 环境配置

### 驱动安装

```bash
# 下载驱动（需要在华为开发者社区注册）
# https://www.hiascend.com/software/cann/community

# 安装NPU驱动
chmod +x Ascend-hdk-910b-npu-driver_x.x.x_linux-x86_64.run
./Ascend-hdk-910b-npu-driver_x.x.x_linux-x86_64.run --full

# 验证
npu-smi info
```

### CANN安装

```bash
# 安装CANN Toolkit
chmod +x Ascend-cann-toolkit_x.x.x_linux-x86_64.run
./Ascend-cann-toolkit_x.x.x_linux-x86_64.run --install

# 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证
python -c "import acl; print('ACL OK')"
```

### PyTorch Ascend安装

```bash
# 安装torch_npu
pip install torch==2.1.0
pip install torch_npu==2.1.0

# 环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
```

## 基础使用

### 设备管理

```python
import torch
import torch_npu

# 检查NPU可用性
print(torch.npu.is_available())
print(torch.npu.device_count())

# 设置设备
device = torch.device("npu:0")
model = model.to(device)

# 数据转移
tensor = tensor.npu()
```

### 模型训练

```python
import torch
import torch_npu

# 模型定义（与CUDA相同）
model = MyModel().npu()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for batch in dataloader:
    inputs = batch['input'].npu()
    labels = batch['label'].npu()
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 混合精度

```python
from torch_npu.contrib import transfer_to_npu

# 自动混合精度
with torch.cuda.amp.autocast():  # torch_npu兼容此接口
    output = model(input)
    loss = criterion(output, target)
```

## 分布式训练

### 单机多卡

```python
import torch.distributed as dist
import torch_npu

# 初始化（使用HCCL后端）
dist.init_process_group(backend='hccl')
local_rank = int(os.environ['LOCAL_RANK'])

torch.npu.set_device(local_rank)
model = model.npu(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### 启动训练

```bash
# torchrun方式
torchrun --nproc_per_node=8 train.py

# 或使用昇腾工具
msrun --worker_num=8 train.py
```

## 模型适配

### Transformers适配

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch_npu

model = AutoModelForCausalLM.from_pretrained("model_path")
model = model.npu()

# 或使用device_map
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    device_map="npu:0"
)
```

### 常见框架支持

| 框架 | 支持状态 |
|-----|---------|
| PyTorch | 通过torch_npu支持 |
| MindSpore | 原生支持 |
| TensorFlow | 通过适配层支持 |
| ONNX Runtime | 支持 |

## 监控与调试

### npu-smi工具

```bash
# 查看NPU状态
npu-smi info

# 实时监控
watch -n 1 npu-smi info

# 查看进程
npu-smi info -t proc
```

### 性能分析

```python
# 使用Profiler
with torch.npu.profiler.profile() as prof:
    model(input)
    
print(prof.key_averages().table())
```

## 常见问题

### 算子兼容性

部分PyTorch算子可能在NPU上不支持或行为不同：

```python
# 检查算子支持
torch_npu.npu.is_supported_op(torch.ops.aten.xxx)

# 回退到CPU
if not supported:
    result = op(tensor.cpu()).npu()
```

### 性能调优

```python
# 启用图优化
torch_npu.npu.set_compile_mode(jit_compile=True)

# 设置精度模式
torch.npu.set_option("ACL_PRECISION_MODE", "allow_fp32_to_fp16")
```

## 与CUDA代码的迁移

大多数CUDA代码可以通过简单替换迁移到NPU：

```python
# CUDA代码
model = model.cuda()
tensor = tensor.cuda()
torch.cuda.synchronize()

# NPU代码
model = model.npu()
tensor = tensor.npu()
torch.npu.synchronize()
```

对于更复杂的情况，可以使用华为提供的迁移工具进行辅助。

昇腾NPU为大模型训练提供了国产化的替代方案。随着软件生态的不断完善，越来越多的框架和模型可以在昇腾平台上运行。对于有国产化需求的场景，昇腾是一个值得考虑的选择。
