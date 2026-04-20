# PyTorch与NumPy基础

## NumPy：科学计算的基石

NumPy是Python生态中数值计算的基础库，为多维数组运算提供了高效实现。尽管深度学习框架已经提供了更强大的张量操作，NumPy仍然是数据预处理、结果分析以及与其他科学计算库交互的关键工具。

假设你正在准备一批训练数据——从CSV文件里读出来的特征矩阵需要做归一化，标签需要做one-hot编码，最后还要按比例切分训练集和验证集。这些"上游"的脏活累活，几乎全靠NumPy来完成。可以说，NumPy是深度学习工作流的"前厅"：模型看到数据之前，数据先要经过NumPy的手。

### ndarray核心概念

NumPy的核心数据结构是ndarray（N-dimensional array）。你可以把它理解成一块连续的内存区域，上面贴了"形状"和"数据类型"两张标签——正是这种紧凑的内存布局，让NumPy的速度远超Python原生列表。ndarray具有以下关键属性：

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

print(arr.shape)    # (2, 3) - 形状
print(arr.dtype)    # float32 - 数据类型
print(arr.ndim)     # 2 - 维度数
print(arr.strides)  # (12, 4) - 步幅（字节）
print(arr.flags)    # 内存布局信息
```

**数据类型**（dtype）决定了数组元素的存储方式与精度：

| dtype | 字节数 | 范围 | 用途 |
|-------|-------|------|------|
| float16 | 2 | ~6.5×10⁴ | 混合精度训练 |
| float32 | 4 | ~3.4×10³⁸ | 默认浮点类型 |
| float64 | 8 | ~1.8×10³⁰⁸ | 高精度计算 |
| int32 | 4 | ±2.1×10⁹ | 整数索引 |
| int64 | 8 | ±9.2×10¹⁸ | 大范围整数 |

### 数组创建与变形

```python
# 创建方式
zeros = np.zeros((3, 4))           # 全零数组
ones = np.ones((2, 3))             # 全一数组
eye = np.eye(4)                    # 单位矩阵
arange = np.arange(0, 10, 2)       # 等差序列
linspace = np.linspace(0, 1, 5)    # 均匀分布
random = np.random.randn(3, 4)     # 标准正态分布

# 形状操作
reshaped = arr.reshape(3, 2)       # 改变形状
transposed = arr.T                 # 转置
flattened = arr.flatten()          # 展平
squeezed = arr[np.newaxis, :]      # 增加维度
```

**广播机制**（Broadcasting）允许不同形状的数组进行运算。在实际项目中，你经常会遇到这样的场景：一个形状为`(batch_size, feature_dim)`的特征矩阵，需要减去一个形状为`(feature_dim,)`的均值向量。如果没有广播机制，你就得手动把均值向量复制`batch_size`份再相减——广播替你省掉了这一步：

```python
a = np.array([[1], [2], [3]])      # shape: (3, 1)
b = np.array([10, 20, 30])         # shape: (3,)
c = a + b                          # shape: (3, 3) 广播结果

# 广播规则：
# 1. 从最后一个维度开始比较
# 2. 维度大小相等，或其中一个为1
# 3. 较小的维度被"复制"以匹配
```

### 高级索引

```python
arr = np.arange(12).reshape(3, 4)

# 基础索引（返回视图）
view = arr[1:, :2]

# 高级索引（返回副本）
fancy = arr[[0, 2], [1, 3]]        # 选取(0,1)和(2,3)
boolean = arr[arr > 5]             # 布尔索引

# 组合索引
mixed = arr[1:, [0, 2]]
```

### 向量化运算

向量化是NumPy高效的关键——避免Python循环，直接在C层面执行批量操作。举个例子，如果你对一个100万行的数组用Python的for循环逐行归一化，可能需要好几秒；换成向量化写法，往往几毫秒就搞定了。这不是锦上添花的技巧，而是实际开发中必须养成的习惯：

```python
# 避免：Python循环
def slow_normalize(arr):
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = arr[i, j] / arr.sum()
    return result

# 推荐：向量化
def fast_normalize(arr):
    return arr / arr.sum()
```

**通用函数**（ufunc）提供了逐元素操作的高效实现：

```python
# 数学运算
np.exp(arr)                        # 指数
np.log(arr + 1)                    # 对数
np.sin(arr)                        # 三角函数
np.sqrt(arr)                       # 平方根

# 聚合运算
arr.sum(axis=0)                    # 按列求和
arr.mean(axis=1)                   # 按行均值
arr.max()                          # 全局最大值
arr.argmax(axis=-1)                # 最大值索引
```

## PyTorch：深度学习的主力框架

PyTorch以其动态计算图、直观的API设计以及与Python生态的无缝集成，成为深度学习研究与应用的主流框架。

你可能会问：既然NumPy已经能做矩阵运算了，为什么还需要PyTorch？答案有两个字——**梯度**。训练神经网络的核心操作是反向传播，而NumPy不提供自动微分。PyTorch在NumPy式的张量运算基础上，加入了GPU加速和自动微分系统（autograd），使得从"算得出来"到"训得起来"只差一步`loss.backward()`的距离。

### Tensor基础

PyTorch的Tensor与NumPy的ndarray高度相似，但增加了GPU加速与自动微分支持：

```python
import torch

# 创建Tensor
x = torch.tensor([[1., 2.], [3., 4.]])
x = torch.zeros(3, 4, dtype=torch.float32)
x = torch.randn(3, 4, device='cuda')       # GPU张量

# 与NumPy互转
numpy_arr = x.cpu().numpy()
torch_tensor = torch.from_numpy(numpy_arr)

# 设备管理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

### 自动微分

PyTorch的autograd系统是其核心能力，通过记录计算图实现自动反向传播。想象你在搭积木：每做一步运算（加减乘除、矩阵乘法、激活函数），PyTorch都在背后默默记下"这一步的输入是什么、用了什么操作"。等你喊一声`backward()`，它就沿着这条记录链，自动算出每个参数对最终损失的贡献——这就是自动微分的本质：

```python
# 启用梯度跟踪
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1
z = y.sum()

# 反向传播
z.backward()
print(x.grad)  # tensor([6., 8.]) = 2x + 2

# 梯度控制
with torch.no_grad():
    # 推理时禁用梯度计算
    y = model(x)

# 分离计算图
detached = y.detach()
```

**计算图的动态性**意味着每次前向传播都可能构建不同的图结构，这对于条件分支、循环等控制流非常友好：

```python
def dynamic_network(x, use_relu=True):
    y = torch.matmul(x, weight)
    if use_relu:
        y = torch.relu(y)
    else:
        y = torch.tanh(y)
    return y
```

### nn.Module：模型构建

`nn.Module`是PyTorch模型的基类，提供了参数管理、子模块组织等功能。在实际开发中，几乎所有的模型——从最简单的两层MLP到千亿参数的大语言模型——都是`nn.Module`的子类。你只需要定义`__init__`里有哪些层，以及`forward`里数据怎么流过这些层，剩下的参数注册、设备迁移、序列化等琐事都由基类帮你打理：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)

# 参数访问
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 模型移动到GPU
model = model.cuda()
```

**常用层**：

```python
# 线性层
nn.Linear(in_features, out_features)

# 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

# 归一化
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# 注意力
nn.MultiheadAttention(embed_dim, num_heads)

# 激活函数
nn.ReLU(), nn.GELU(), nn.SiLU()
```

### 训练循环

标准的PyTorch训练循环模式：

```python
model = MLP(784, 256, 10).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
    
    scheduler.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # 验证逻辑
            pass
```

### 数据加载

`DataLoader`提供了高效的数据批处理与多进程加载：

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[idx]

dataset = CustomDataset(data, labels)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,        # 加速GPU传输
    drop_last=True
)
```

### 混合精度训练

使用`torch.cuda.amp`进行自动混合精度训练，在保持精度的同时降低显存占用。你可能遇到过这种情况：模型跑FP32刚好爆显存，砍batch size又影响收敛。混合精度训练是一个务实的解决方案——前向和反向计算用FP16（速度快、省显存），权重更新仍用FP32（保证精度），两全其美：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in loader:
    optimizer.zero_grad()
    
    # 自动混合精度前向
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # 缩放后反向传播
    scaler.scale(loss).backward()
    
    # 梯度反缩放与更新
    scaler.step(optimizer)
    scaler.update()
```

### 模型保存与加载

```python
# 保存完整模型（不推荐）
torch.save(model, 'model.pth')

# 保存状态字典（推荐）
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 分布式训练基础

当单卡显存装不下模型，或者训练速度太慢需要多卡加速时，分布式训练就成了刚需。PyTorch提供了多种分布式训练范式——从最简单的`DataParallel`到工业级的`DistributedDataParallel`（DDP），复杂度和效率逐级递增：

```python
# 数据并行（单机多卡）
model = nn.DataParallel(model)

# 分布式数据并行（推荐）
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

## PyTorch与NumPy的协作

两者在实际项目中经常配合使用：

```python
# 数据预处理用NumPy
data = np.load('data.npy')
data = (data - data.mean()) / data.std()

# 转换为Tensor进行训练
tensor_data = torch.from_numpy(data).float()

# 结果转回NumPy进行分析
predictions = model(tensor_data).detach().cpu().numpy()
np.save('predictions.npy', predictions)
```

**注意事项**：
- `torch.from_numpy()`创建的Tensor与原NumPy数组共享内存
- GPU Tensor需要先`.cpu()`再转NumPy
- 注意数据类型匹配（float64 vs float32）

PyTorch与NumPy共同构成了Python深度学习的计算基础。NumPy负责通用数值计算与数据处理，PyTorch则在此基础上提供了GPU加速、自动微分与模型抽象，两者的熟练掌握是进行大模型开发的必备技能。一个典型的工作流是：用NumPy做数据清洗和特征工程，用PyTorch搭建和训练模型，训练完成后再把预测结果转回NumPy做可视化分析。理解这条"数据搬运链"上每一环的职责，是高效开发的基础。
