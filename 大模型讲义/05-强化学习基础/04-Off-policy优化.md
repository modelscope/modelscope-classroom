# 5.4 Off-Policy 优化

**Off-policy** 算法可以使用非当前策略采集的数据进行学习，大幅提高样本效率。本节讨论 Off-policy 的核心方法：从经典的 Q-Learning 到 LLM 对齐中的 DPO，再到游戏 AI 中的 AlphaZero，以及知识蒸馏技术。

## 5.4.1 Q-Learning 与深度 Q 网络

### 表格 Q-Learning

Q-Learning 的更新规则：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Off-policy 的本质**：TD 目标 $r + \gamma \max_{a'} Q(s', a')$ 使用的是最优动作的价值，与实际采取的动作 $a$ 无关。因此，无论数据来自什么策略（随机探索、专家演示、历史数据），都可以用于更新。

### DQN 回顾

DQN 用神经网络 $Q_\theta(s, a)$ 近似 $Q$ 函数，关键技术：

1. **经验回放**：存储历史数据，随机采样训练
2. **目标网络**：$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$，$\theta^-$ 定期从 $\theta$ 复制

损失函数：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ (y - Q_\theta(s, a))^2 \right]$$

### 重要性采样视角

Off-policy 学习可以用重要性采样理解。设数据来自行为策略 $\mu(a|s)$，目标策略为 $\pi(a|s)$：

$$\mathbb{E}_\mu [f(s, a)] = \mathbb{E}_\pi \left[ \frac{\mu(a|s)}{\pi(a|s)} f(s, a) \right]$$

Q-Learning 的巧妙之处在于：它不需要显式的重要性权重，因为 TD 目标只依赖 $\max_{a'} Q(s', a')$，而非实际动作的概率。

## 5.4.2 DPO：直接偏好优化

### RLHF 的复杂性

标准 RLHF 流程包含三个阶段：

1. **SFT**：监督微调
2. **奖励模型训练**：学习人类偏好
3. **RL 优化**：用 PPO 最大化奖励

其中第三步需要在线采样、训练 Critic、多次迭代，实现复杂。

### DPO 的思想

**DPO**（Direct Preference Optimization，Rafailov et al., 2023）提出：能否跳过奖励模型，直接用偏好数据优化策略？

关键洞察：最优策略和奖励函数之间存在解析关系。在 KL 约束下：

$$\pi^*(a|s) \propto \pi_{\text{ref}}(a|s) \exp\left(\frac{1}{\beta} r(s, a)\right)$$

反解奖励：

$$r(s, a) = \beta \log \frac{\pi^*(a|s)}{\pi_{\text{ref}}(a|s)} + \beta \log Z(s)$$

其中 $Z(s)$ 是配分函数，不影响偏好比较。

### Bradley-Terry 模型

人类偏好可以用 **Bradley-Terry 模型**描述：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

其中 $y_w$ 是被偏好的回复，$y_l$ 是较差的回复，$\sigma$ 是 sigmoid 函数。

### DPO 损失函数

将奖励的表达式代入 Bradley-Terry 模型，配分函数抵消，得到 **DPO 损失**：

$$L_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

简化记号：

$$L_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta (\hat{r}_\theta(y_w) - \hat{r}_\theta(y_l)) \right) \right]$$

其中 $\hat{r}_\theta(y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是隐式奖励。

### DPO 的优势

1. **无需奖励模型**：直接用偏好数据训练
2. **无需在线采样**：完全 off-policy，只需偏好数据集
3. **无需 Critic**：不需要价值网络
4. **实现简单**：与 SFT 类似的训练流程

### DPO 的局限

1. **数据分布偏移**：训练数据来自 $\pi_{\text{ref}}$，而非 $\pi_\theta$
2. **无法迭代改进**：不能用新策略生成更好的数据
3. **对比较噪声敏感**：偏好标注的噪声直接影响学习

### DPO 变体

**IPO**（Identity Preference Optimization）：用不同的损失函数，更鲁棒。

**KTO**（Kahneman-Tversky Optimization）：只需要"好/差"的单边标注，不需要成对比较。

**ORPO**（Odds Ratio Preference Optimization）：用 odds ratio 替代 log probability ratio。

## 5.4.3 AlphaZero 与 AlphaGo

### AlphaGo 的架构

AlphaGo（2016）结合了深度学习和蒙特卡洛树搜索（MCTS）：

1. **策略网络** $p_\theta(a|s)$：预测下一步落子概率
2. **价值网络** $v_\phi(s)$：预测胜率
3. **MCTS**：用网络引导搜索，提升决策质量

训练流程：

1. 用人类棋谱监督学习初始化
2. 自我对弈生成数据
3. 用自我对弈数据更新网络

### AlphaZero 的简化

AlphaZero（2017）进一步简化，**从零开始学习**（tabula rasa）：

1. **无需人类棋谱**：只用自我对弈
2. **统一的网络**：策略和价值共享主干
3. **通用性**：同一算法适用于围棋、国际象棋、日本将棋

### MCTS 与策略改进

MCTS 的作用是**策略改进**（Policy Improvement）。设当前策略网络为 $p_\theta$，MCTS 搜索后得到改进的策略 $\pi_{\text{MCTS}}$：

$$\pi_{\text{MCTS}}(a|s) \propto N(s, a)^{1/\tau}$$

其中 $N(s, a)$ 是动作 $a$ 在搜索中被访问的次数，$\tau$ 是温度。

策略网络的训练目标是模仿 MCTS 策略：

$$L_p = -\sum_a \pi_{\text{MCTS}}(a|s) \log p_\theta(a|s)$$

价值网络的目标是预测游戏结果 $z \in \{-1, +1\}$：

$$L_v = (v_\phi(s) - z)^2$$

### Off-Policy 的体现

AlphaZero 的数据来自**历史版本**的策略，而非当前策略：

1. 用当前策略自我对弈，生成游戏数据
2. 数据存入回放缓冲区
3. 从缓冲区采样训练
4. 更新后的策略继续对弈

缓冲区中可能包含数万局游戏、数百万个状态，来自不同历史版本的策略。这是典型的 off-policy 学习。

### 与 LLM 的联系

AlphaZero 的思想已被应用于 LLM 推理：

- **AlphaCode**：用大规模采样和筛选解决编程问题
- **Process Reward Model**：对推理过程的每一步打分
- **MCTS 解码**：用树搜索探索不同的推理路径

## 5.4.4 软蒸馏与数据蒸馏

### 知识蒸馏

**知识蒸馏**（Knowledge Distillation, Hinton et al., 2015）将大模型（教师）的知识迁移到小模型（学生）：

$$L_{\text{KD}} = \text{KL}(p_{\text{teacher}}(\cdot|x) \| p_{\text{student}}(\cdot|x))$$

或者用温度 softmax：

$$L_{\text{KD}} = -\sum_y p_{\text{teacher}}(y|x; \tau) \log p_{\text{student}}(y|x; \tau)$$

高温度使分布更"软"，保留更多信息。

### 软蒸馏 vs 硬蒸馏

**硬蒸馏**：学生模仿教师的最终输出（argmax）

$$L_{\text{hard}} = -\log p_{\text{student}}(y_{\text{teacher}}|x)$$

**软蒸馏**：学生模仿教师的完整分布

$$L_{\text{soft}} = \text{KL}(p_{\text{teacher}} \| p_{\text{student}})$$

软蒸馏保留了教师的"不确定性"信息，通常效果更好。

### 数据蒸馏

**数据蒸馏**是一种特殊的蒸馏方式：用教师生成大量数据，学生在这些数据上训练。

流程：
1. 教师模型生成回复 $y \sim p_{\text{teacher}}(\cdot|x)$
2. （可选）筛选高质量回复
3. 学生在 $(x, y)$ 对上做 SFT

这实际上是 off-policy 学习：学生用教师策略的数据训练。

### 在 LLM 中的应用

**Self-Instruct / Alpaca**：用 GPT-4 生成指令-回复对，训练小模型。

**On-Policy 蒸馏**：让学生自己生成回复，教师打分或改写，迭代训练。

**Rejection Sampling**：学生生成多个回复，用教师（或奖励模型）选择最好的，作为 SFT 数据。

### 与 RL 的关系

蒸馏可以视为一种简化的 RL：

- **奖励**：与教师分布的匹配程度
- **策略**：学生模型
- **优化**：最大化期望"奖励"

DPO 也可以理解为一种蒸馏：从偏好数据中蒸馏人类的隐式奖励函数。

## 5.4.5 Off-Policy 的挑战

### 分布偏移

Off-policy 方法面临**分布偏移**（Distribution Shift）：训练分布与目标分布不同。

在 Q-Learning 中，这导致对未见过的 $(s, a)$ 的 $Q$ 值估计不准确。

在 DPO 中，偏好数据来自 $\pi_{\text{ref}}$，而优化的是 $\pi_\theta$，两者差异越大，估计越不可靠。

### 外推误差

当策略访问训练数据未覆盖的区域时，价值估计可能出现严重偏差——称为**外推误差**（Extrapolation Error）。

缓解方法：
- **保守 Q 学习**（CQL）：惩罚 OOD 状态-动作的 $Q$ 值
- **KL 约束**：限制 $\pi_\theta$ 与 $\pi_{\text{ref}}$ 的差异
- **Behavior Cloning 正则化**：加入模仿学习损失

### 重要性权重方差

如果使用重要性采样修正分布偏移，权重 $\frac{\pi_\theta(a|s)}{\pi_b(a|s)}$ 可能方差很大，导致训练不稳定。

缓解方法：
- **权重裁剪**：$\min(\rho, c)$ 或 $\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$
- **V-trace**：递归地裁剪权重
- **PPO**：用裁剪目标函数，隐式控制重要性权重
