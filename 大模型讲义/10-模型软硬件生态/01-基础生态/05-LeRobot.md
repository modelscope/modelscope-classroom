# LeRobot

LeRobot是Hugging Face推出的机器人学习开源平台，旨在将大模型时代的方法论引入机器人领域，降低机器人AI研究的门槛。

## 概述

LeRobot的核心理念是将机器人学习标准化，如同Transformers库对NLP模型所做的那样。它提供了：

- **预训练模型**：共享的操作策略模型
- **标准数据集**：统一格式的机器人数据集
- **仿真环境**：用于训练和评估的仿真平台
- **真实硬件支持**：低成本机械臂的驱动

### 设计理念

```
预训练模型（Policy）
       ↓
  LeRobot Hub     ←→   标准数据格式（LeRobotDataset）
       ↓
仿真/真实环境（Gym接口）
```

## 安装与配置

```bash
# 基础安装
pip install lerobot

# 完整安装（含仿真环境）
pip install "lerobot[all]"

# 开发安装
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[all]"
```

## 核心组件

### 策略模型（Policy）

LeRobot支持多种策略架构：

| 策略 | 描述 | 适用场景 |
|-----|------|---------|
| ACT | Action Chunking Transformer | 灵巧操作 |
| Diffusion Policy | 扩散模型策略 | 多模态动作分布 |
| TDMPC | 时序差分MPC | 需要规划的任务 |
| VQ-BeT | 向量量化行为Transformer | 离散动作空间 |

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")
```

### 数据集格式

LeRobotDataset定义了统一的数据格式：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")

# 数据结构
sample = dataset[0]
# {
#     "observation.images.top": tensor [C, H, W],
#     "observation.state": tensor [state_dim],
#     "action": tensor [action_dim],
#     "episode_index": int,
#     "frame_index": int,
#     "timestamp": float,
# }
```

**数据集元信息**：

```python
print(dataset.meta)
# {
#     "fps": 50,
#     "video_backend": "pyav",
#     "robot_type": "aloha",
#     "features": {...},
# }
```

### 仿真环境

LeRobot集成了多个仿真环境：

```python
import gymnasium as gym
import gym_aloha  # LeRobot提供的环境

env = gym.make("gym_aloha/AlohaTransferCube-v0")
observation, info = env.reset()

for _ in range(1000):
    action = policy.select_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
```

支持的环境：
- **ALOHA**：双臂操作仿真
- **PushT**：简单推动任务
- **xArm**：xArm机械臂仿真

## 训练流程

### 配置文件

```yaml
# configs/policy/act.yaml
policy:
  name: act
  
  # 网络结构
  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1
  
  # 动作块
  chunk_size: 100
  n_action_steps: 100
  
  # 输入配置
  input_shapes:
    observation.images.top: [3, 480, 640]
    observation.state: [14]
  output_shapes:
    action: [14]
```

### 训练脚本

```bash
python lerobot/scripts/train.py \
    --policy.name=act \
    --env.name=aloha \
    --env.task=AlohaTransferCube-v0 \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --training.num_epochs=100 \
    --training.batch_size=8 \
    --training.lr=1e-5 \
    --device=cuda
```

### Python训练

```python
from lerobot.scripts.train import train
from lerobot.common.utils.utils import init_hydra_config

cfg = init_hydra_config("lerobot/configs/default.yaml")
train(cfg)
```

## 数据采集

### 遥操作采集

LeRobot支持通过遥操作采集真实数据：

```bash
# 使用leader-follower采集
python lerobot/scripts/control_robot.py \
    --robot.name=aloha \
    --robot.type=real \
    --control.type=teleoperate \
    --output_dir=./my_dataset
```

### 数据上传

```python
from lerobot.common.datasets.push_dataset_to_hub import push_dataset_to_hub

push_dataset_to_hub(
    raw_dir="./my_dataset",
    repo_id="username/my_robot_dataset",
    fps=50,
    video=True,
)
```

## 评估与部署

### 仿真评估

```bash
python lerobot/scripts/eval.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
    --env.name=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=50
```

### 真实机器人部署

```python
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.policies.act.modeling_act import ACTPolicy

# 初始化机器人
robot = make_robot("aloha")
robot.connect()

# 加载策略
policy = ACTPolicy.from_pretrained("path/to/model")

# 控制循环
while True:
    observation = robot.get_observation()
    action = policy.select_action(observation)
    robot.send_action(action)
```

## 低成本硬件

LeRobot推动低成本机器人硬件的普及：

### Koch v1.1机械臂

一种低成本的开源机械臂设计，成本约几百美元：

```python
from lerobot.common.robot_devices.robots.koch import KochRobot

robot = KochRobot(
    leader_arms=["left", "right"],
    follower_arms=["left", "right"],
)
robot.connect()
```

### 支持的硬件

| 硬件 | 类型 | 特点 |
|-----|------|------|
| ALOHA | 双臂平台 | 专业级 |
| Koch | 机械臂 | 低成本、开源 |
| xArm | 机械臂 | 商业级 |
| SO-100 | 机械臂 | 教育级 |

## Hugging Face Hub集成

### 模型分享

```python
policy.push_to_hub("username/my_robot_policy")
```

### 数据集浏览

LeRobot数据集可以在Hugging Face Hub上浏览，支持：
- 视频预览
- 元数据查看
- 统计信息

## 与大模型的结合

LeRobot正在探索将视觉-语言模型与机器人策略结合：

```python
# 使用VLM进行任务规划
# 使用LeRobot策略执行低层动作
```

这种分层架构中，大模型负责理解指令和高层规划，LeRobot策略负责具体的运动控制。

LeRobot代表了将大模型方法论引入机器人学习的重要尝试。通过标准化数据格式、共享预训练模型、降低硬件门槛，它正在使机器人AI研究变得更加民主化。随着更多数据和模型的积累，跨实体、跨任务的泛化将成为可能。
