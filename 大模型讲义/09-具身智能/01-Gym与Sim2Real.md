# 仿真环境与Sim-to-Real

假设你要培训一名飞行员。直接让新手上真机练习显然太危险、太昂贵，所以航空业发明了飞行模拟器——在安全的环境中无限次练习起降、应对紧急情况、练习各种天气条件。具身智能的仿真训练思路完全一样：先在虚拟世界里让机器人反复练习，然后再迁移到真实世界。但模拟器和现实之间的差距（就像飞行模拟器再逼真也不是真飞机），是这条路径上最大的挑战。

## 强化学习环境接口标准

### Gym/Gymnasium接口

OpenAI Gym（现由Farama Foundation维护，更名为Gymnasium）确立了强化学习环境的标准接口。其核心抽象极为简洁，本质上就是两个动作的循环：智能体观察环境→做出决策→环境反馈结果→智能体再观察……这就像飞行模拟器的工作流程：飞行员看仪表盘（观察），拉操纵杆（动作），模拟器给出飞机的新状态（转移）：

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = policy(observation)  # 智能体策略
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

接口的核心方法包括：

| 方法 | 输入 | 输出 | 说明 |
|-----|------|------|------|
| `reset()` | seed, options | observation, info | 重置环境到初始状态 |
| `step()` | action | obs, reward, terminated, truncated, info | 执行动作，返回转移 |
| `render()` | - | frame/None | 可视化当前状态 |
| `close()` | - | - | 释放资源 |

**空间定义**是Gym的另一核心概念。动作空间（action_space）与观测空间（observation_space）明确定义了有效的动作与观测范围。以飞行模拟器来说：动作空间就是操纵杆能推的范围（连续空间），而“打开起落架”或“收起”就是离散空间：

```python
from gymnasium.spaces import Box, Discrete, Dict, Tuple

# 连续动作空间：[-1, 1]^3
action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

# 离散动作空间：{0, 1, 2, 3}
action_space = Discrete(4)

# 复合空间
observation_space = Dict({
    "image": Box(0, 255, (84, 84, 3), dtype=np.uint8),
    "state": Box(-np.inf, np.inf, (10,), dtype=np.float32)
})
```

### 环境包装器

Wrapper机制允许在不修改原始环境的前提下添加功能：

```python
class FrameStack(gym.Wrapper):
    """堆叠连续帧以提供时序信息"""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info
    
    def _get_obs(self):
        return np.stack(self.frames, axis=0)
```

常用的预置Wrapper包括：
- `TimeLimit`：限制episode最大步数
- `RecordVideo`：录制环境视频
- `NormalizeObservation`：观测归一化
- `ClipAction`：动作裁剪

### 向量化环境

强化学习训练需要大量样本，向量化环境通过并行多个环境实例提升采样效率。这就像航空学校同时开设 16 台飞行模拟器，每台上都有一个学员在练习，所有经验汇总起来共同优化一套飞行策略：

```python
envs = gym.vector.make("HalfCheetah-v4", num_envs=16, asynchronous=True)
observations, infos = envs.reset()

# 批量执行动作
actions = policy(observations)  # shape: (16, action_dim)
observations, rewards, terminateds, truncateds, infos = envs.step(actions)
```

同步模式（SyncVectorEnv）等待所有环境完成后返回，异步模式（AsyncVectorEnv）使用多进程实现真正的并行。

## 物理仿真引擎

### MuJoCo

MuJoCo（Multi-Joint dynamics with Contact）是DeepMind维护的高性能物理仿真引擎，以其数值稳定性和仿真速度著称。

**模型定义**采用MJCF（MuJoCo XML）格式：

```xml
<mujoco model="simple_arm">
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.4 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" ctrlrange="-1 1"/>
    <motor joint="joint2" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
```

**核心数据结构**包含模型参数（mjModel）与仿真状态（mjData）：

```python
import mujoco

model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

# 设置关节位置
data.qpos[:] = [0.1, 0.2]
# 前向动力学
mujoco.mj_forward(model, data)
# 获取末端位置
end_effector_pos = data.site_xpos[0]

# 仿真一步
data.ctrl[:] = [0.5, -0.3]  # 控制输入
mujoco.mj_step(model, data)
```

MuJoCo的优势在于：
- 快速准确的接触动力学
- 自动计算雅可比矩阵与质量矩阵
- 支持柔性体与肌腱建模
- 开源且持续维护

### Isaac Gym/Isaac Sim

NVIDIA Isaac平台利用GPU实现大规模并行物理仿真，仿真速度可达数万FPS。

**Isaac Gym**直接在GPU上运行物理仿真，与PyTorch深度集成：

```python
from isaacgym import gymapi, gymtorch

gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 创建多个环境实例
for i in range(num_envs):
    env = gym.create_env(sim, lower, upper, num_per_row)
    actor = gym.create_actor(env, asset, pose, "robot", i, 0)

# GPU张量直接获取状态
root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
```

**Isaac Sim**基于Omniverse平台，提供更丰富的渲染能力和场景编辑功能，支持光线追踪、合成数据生成等高级特性。

### 其他仿真平台

| 平台 | 特点 | 适用场景 |
|-----|------|---------|
| PyBullet | 开源、易用 | 快速原型验证 |
| SAPIEN | 关节物体仿真 | 操作任务 |
| Habitat | 大规模室内场景 | 导航任务 |
| AI2-THOR | 交互式场景 | 家居机器人 |
| CARLA | 自动驾驶仿真 | 车辆控制 |

## Sim-to-Real迁移技术

### Reality Gap分析

仿真与现实的差距主要来源于三个层面。回想飞行模拟器的场景：模拟器里的风是简化的数学模型，真实的乱流却复杂得多；模拟器里的天空质感均匀平整，真实的云层、光线千变万化；模拟器里按下按钮立即响应，真机的液压系统总有那么一丝延迟。具体来说：

**物理差距**：
- 接触模型简化（刚体假设、离散化）
- 参数不确定性（质量、摩擦、阻尼）
- 未建模动力学（柔性、延迟、非线性）

**视觉差距**：
- 渲染与真实图像的分布差异
- 光照条件变化
- 传感器噪声与畸变

**控制差距**：
- 执行器响应差异
- 通信延迟
- 采样频率不匹配

### 域随机化

域随机化（Domain Randomization）通过在仿真中引入大量随机变化，使策略对变化具有鲁棒性。核心假设是：如果策略能够处理仿真中的大量变化，真实世界将只是其中一种情况。

假设你正在训练飞行员应对侧风着陆。你不是只练 10 节的侧风，而是随机在 0～30 节之间变化，有时还加上突然的阵风。练多了之后，飞行员对任何强度的侧风都能从容应对——包括从未见过的 17.3 节。这就是域随机化的精髓。

**物理域随机化**：

```python
class RandomizedEnv(gym.Wrapper):
    def reset(self, **kwargs):
        # 随机化物理参数
        self.model.body_mass[1] *= np.random.uniform(0.8, 1.2)
        self.model.geom_friction[:] *= np.random.uniform(0.5, 2.0)
        self.model.dof_damping[:] *= np.random.uniform(0.5, 1.5)
        
        # 随机化初始状态
        qpos = self.init_qpos + np.random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.1, 0.1, self.model.nv)
        
        return super().reset(**kwargs)
```

**视觉域随机化**涉及：
- 纹理随机化：物体表面、背景
- 光照随机化：位置、强度、颜色
- 相机随机化：视角、焦距、噪声
- 干扰物随机化：遮挡、杂乱背景

**动力学随机化**可以通过网络适应层处理：

$$\mathbf{a} = \pi(\mathbf{o}, \mathbf{z}), \quad \mathbf{z} = \phi(\mathbf{o}_{1:t}, \mathbf{a}_{1:t-1})$$

其中 $\mathbf{z}$ 是从历史轨迹推断出的隐式环境参数。

### 系统辨识与仿真校准

精确测量真实系统参数可以显著缩小Reality Gap：

**几何参数**：
- 激光扫描获取精确尺寸
- 运动捕捉标定关节位置

**动力学参数**：
- 摆锤实验测量转动惯量
- 阶跃响应辨识执行器模型
- 接触实验测量摩擦系数

**仿真校准**流程：

```
真实数据采集 → 参数优化 → 仿真更新 → 误差评估 → 迭代
```

目标函数通常为轨迹匹配误差：

$$\mathcal{L} = \sum_{t} \|\mathbf{s}_t^{\text{real}} - \mathbf{s}_t^{\text{sim}}(\theta)\|^2$$

其中 $\theta$ 是待辨识的仿真参数。

### 迁移学习方法

**微调（Fine-tuning）**：在仿真中预训练后，使用少量真实数据继续训练。需要平衡适应速度与灾难性遗忘。

**渐进式网络（Progressive Networks）**：为真实域添加新的网络列，同时保持仿真知识：

$$\mathbf{h}_i^{(k)} = \sigma\left(\mathbf{W}_i^{(k)} \mathbf{h}_{i-1}^{(k)} + \sum_{j<k} \mathbf{U}_i^{(j:k)} \mathbf{h}_{i-1}^{(j)}\right)$$

**对抗域适应**：学习域不变的特征表示，使判别器无法区分仿真与真实特征。

### 实例：灵巧手操作的Sim-to-Real

OpenAI在魔方求解任务中展示了成功的Sim-to-Real迁移。关键技术包括：

1. **自动域随机化**（ADR）：根据策略性能自适应调整随机化范围
2. **记忆增强策略**：LSTM处理部分可观测性
3. **触觉反馈**：利用关节力矩推断接触状态
4. **课程学习**：从简单到复杂逐步增加任务难度

训练规模：约10,000年仿真时间（利用大规模并行），迁移到真实Shadow Hand后实现约60%成功率。

## 基准任务与评估

### 运动控制基准

MuJoCo提供的经典连续控制任务：

| 环境 | 观测维度 | 动作维度 | 任务描述 |
|-----|---------|---------|---------|
| HalfCheetah | 17 | 6 | 半猎豹前进 |
| Ant | 111 | 8 | 蚂蚁移动 |
| Humanoid | 376 | 17 | 人形行走 |
| Walker2d | 17 | 6 | 双足行走 |

这些任务的奖励通常结合前进速度、能量消耗、存活奖励等多个因素。

### 操作任务基准

**Meta-World**提供50种桌面操作任务，支持多任务与元学习研究：
- 单任务学习（ML1）：单一任务多种初始化
- 多任务学习（MT10/MT50）：同时学习多种技能
- 元学习（ML10/ML45）：快速适应新任务

**RLBench**基于CoppeliaSim，提供100+操作任务，支持多种观测模态（RGB、深度、点云、状态）。

### 导航基准

**Habitat Challenge**评估室内导航能力：
- PointNav：到达指定坐标
- ObjectNav：找到指定类别物体
- ImageNav：到达图像显示位置

评估指标包括成功率（Success）、SPL（Success weighted by Path Length）、软SPL等。

仿真环境与Sim-to-Real技术是具身智能研究的基础设施。标准化的接口降低了算法开发门槛，高效的物理仿真实现了大规模数据生成，而 Sim-to-Real 方法则是将仿真成果转化为实际应用的桥梁。就像航空业从最早的简陋模拟器发展到今天几乎以假乱真的全动模拟，机器人的仿真训练也在快速走向成熟。随着仿真精度的提升与迁移技术的进步，仿真训练正在成为机器人技能获取的主流范式。
