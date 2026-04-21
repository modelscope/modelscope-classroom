# 昇腾NPU环境配置与使用

华为昇腾（Ascend）NPU是国产AI加速器的代表，在大模型训练和推理场景中展现出竞争力。本节介绍昇腾NPU的环境配置方法。

假设你所在的团队接到一个信创项目——要求在国产硬件上完成大模型的微调与部署。你手头有一台Atlas 800训练服务器，8张910B卡，但打开终端后第一反应大概是："这和CUDA环境差别大吗？torch代码要重写吗？"这正是本节要回答的核心问题。好消息是，昇腾生态在近两年进步显著，大多数PyTorch代码只需将`.cuda()`改为`.npu()`就能运行；坏消息是，版本配套、算子兼容等"暗坑"仍然不少，需要提前了解才能少走弯路。

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

CANN是昇腾的核心软件栈，地位类似于NVIDIA的CUDA Toolkit——它为上层框架提供算子实现、图优化和硬件抽象。如果把NPU比作发动机，CANN就是变速箱和传动系统，你不需要直接操作它，但版本选错了整辆车都跑不起来。CANN包含：

- **AscendCL**：底层计算接口，相当于CUDA Runtime API
- **算子库**：cuDNN的等价层，提供卷积、矩阵乘等高性能实现
- **图引擎**：计算图优化，自动进行算子融合和内存优化
- **调试工具**：性能分析与问题定位

### 软件版本对应

在实际项目中，版本配套是昇腾环境最容易出问题的环节——CANN、驱动、torch_npu三者的版本必须严格匹配，否则会遇到各种莫名其妙的段错误或算子不支持。建议在安装前先查阅华为官方的版本配套表：

```
CANN 8.0 → PyTorch 2.1/2.2 → torch_npu 2.1.0/2.2.0
CANN 7.0 → PyTorch 2.0/2.1 → torch_npu 2.0.0/2.1.0
CANN 6.0 → PyTorch 1.11   → torch_npu 1.11.0
```

> **踩坑提示**：不要尝试"跨版本搭配"，比如CANN 7.0配PyTorch 2.2。即使安装过程不报错，运行时也会出现随机的计算结果错误或core dump，排查起来极为痛苦。

## 环境配置

你可能遇到过这种情况：在NVIDIA机器上装环境只需要`conda install pytorch`就搞定，但昇腾环境需要手动安装驱动、CANN Toolkit、torch_npu三个组件，顺序不能错。下面我们按"从底层到上层"的顺序逐步安装。

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

torch_npu是连接PyTorch和昇腾NPU的桥梁。安装时最关键的一点是：torch和torch_npu的版本号必须完全一致。

```bash
# 安装torch_npu（版本号必须与torch严格对应）
pip install torch==2.1.0
pip install torch_npu==2.1.0

# 环境变量
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH

# 快速验证整条链路是否打通
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

> **实用建议**：如果你的团队需要同时维护CUDA和NPU两套环境，强烈建议使用Docker。华为提供了官方的昇腾开发镜像（`ascendhub.huawei.com`），驱动、CANN、torch_npu已经预装配好，可以省去大量版本配套的麻烦。

## 基础使用

掌握了安装流程，接下来看看日常开发中如何使用NPU。你会发现，如果之前写过CUDA代码，迁移成本其实很低——核心区别就是把`cuda`换成`npu`。

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

想象一下你在训练一个7B参数的模型，FP32精度下显存不够用。混合精度是最直接的解决方案。昇腾对AMP的支持比较完善，而且有一个便利的"魔法导入"——`transfer_to_npu`，它会自动将CUDA相关的调用重定向到NPU，这在迁移已有代码时特别省事：

```python
from torch_npu.contrib import transfer_to_npu  # 导入后，cuda调用自动转为npu

# 自动混合精度（接口与CUDA完全一致）
with torch.cuda.amp.autocast():  # torch_npu兼容此接口
    output = model(input)
    loss = criterion(output, target)
```

> **注意**：`transfer_to_npu`虽然方便，但在调试阶段建议显式使用`.npu()`，这样出错时更容易定位问题是出在NPU适配层还是模型本身。

## 分布式训练

在实际项目中，单卡训练大模型几乎不可能——一张910B的64GB HBM连7B模型的全参数微调都勉强。多卡并行是常态。昇腾的分布式通信库叫HCCL（Huawei Collective Communication Library），对标NVIDIA的NCCL，使用方式也高度类似。

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

举个例子：你想在昇腾上跑Qwen-7B做推理，代码该怎么写？其实和CUDA环境几乎一样，只需要确保`import torch_npu`在最前面。

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

昇腾环境的常见问题集中在三个方面：算子兼容性、精度差异和性能调优。下面逐一说明，这些都是从实际项目中总结出来的高频"坑点"。

### 算子兼容性

这是从CUDA迁移到NPU时最头疼的问题。部分PyTorch算子可能在NPU上不支持或行为不同——尤其是一些不太常见的算子（如某些自定义的scatter操作）或第三方库的CUDA kernel：

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

假设你有一个在A100上跑得好好的训练脚本，现在要迁移到昇腾910B上。大多数情况下，迁移工作量比你想象的要小——核心就是三步替换：

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

对于更复杂的情况（比如代码中使用了自定义CUDA kernel），可以使用华为提供的`msadvisor`迁移分析工具，它会扫描代码并标注哪些API需要修改、哪些算子可能不支持。

> **迁移清单**：实际迁移时，建议按以下顺序排查：(1) 将所有`.cuda()`替换为`.npu()`；(2) 将`nccl`后端改为`hccl`；(3) 运行一个小batch验证精度是否对齐；(4) 逐步增大batch size测试稳定性。不要一上来就跑完整训练，先确保前向传播的输出与CUDA环境一致。

昇腾NPU为大模型训练提供了国产化的替代方案。从实际使用体验来看，常规的Transformers模型训练和推理已经比较成熟，SWIFT、LLaMA-Factory等主流微调框架也已适配昇腾。主要的挑战集中在自定义算子的兼容性和某些边缘场景的精度对齐上。对于有信创需求的团队，建议从推理场景入手——迁移成本最低、验证最快，积累经验后再逐步扩展到微调和全量训练。
