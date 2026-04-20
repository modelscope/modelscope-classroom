# Agentic RL

Agentic RL（智能体强化学习）将强化学习方法引入LLM智能体的优化过程。与传统的监督学习不同，Agentic RL让智能体通过与环境的交互来学习最优策略，这对于处理复杂、开放式任务尤为重要。

假设你在教一个小孩子下棋。监督学习的方式是给他看成千上万盘大师对弈，让他模仿每一步。而强化学习的方式是让他自己下——赢了奖励、输了总结教训，下得越多就越强。传统的RLHF只优化“每一步棋好不好”，而Agentic RL优化的是“整盘棋的策略好不好”——某一步看似吃亏，但如果它服务于整体胜利，那就是好棋。

## 从RLHF到Agentic RL

### RLHF回顾

RLHF（Reinforcement Learning from Human Feedback）是优化LLM的主流方法。简单来说，RLHF的过程就像让人类老师给学生的作文打分，然后学生根据分数调整写作风格：

$$
\mathcal{L}_{\text{RLHF}} = -\mathbb{E}_{x \sim D, y \sim \pi_\theta}[r(x, y)] + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]
$$

其中$r(x, y)$是奖励模型的评分，KL散度项防止策略偏离参考模型太远。

RLHF优化的是单轮对话的质量，而Agentic RL需要考虑多步骤交互的累积奖励。这就像作文打分与项目管理的区别：作文打分看的是单篇质量，而项目管理要评估的是整个项目从开始到结束的整体效果。

### Agentic RL的特点

| 特性 | RLHF | Agentic RL |
|------|------|------------|
| 交互长度 | 单轮 | 多轮/多步骤 |
| 状态空间 | 输入文本 | 对话历史+环境状态 |
| 动作空间 | 生成回复 | 文本生成+工具调用 |
| 奖励信号 | 人类偏好 | 任务完成度+中间反馈 |
| 环境 | 静态 | 动态、可交互 |

## 核心技术

### 轨迹级优化

与token级优化不同，Agentic RL在完整交互轨迹上进行优化。打个比方，token级优化就像优化棋手的每一手棋，而轨迹级优化是站在整盘棋的角度评估这一系列操作是否最终赢得了胜利：

```python
class TrajectoryOptimizer:
    """轨迹级优化器"""
    
    def __init__(self, agent, reward_model, gamma=0.99):
        self.agent = agent
        self.reward_model = reward_model
        self.gamma = gamma
        
    def collect_trajectory(self, task) -> dict:
        """收集一条完整的交互轨迹"""
        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": []
        }
        
        state = self.agent.reset(task)
        done = False
        
        while not done:
            # 记录状态
            trajectory["states"].append(state)
            
            # Agent决策
            action = self.agent.act(state)
            trajectory["actions"].append(action)
            
            # 环境反馈
            next_state, reward, done = self.agent.step(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)
            
            state = next_state
            
        return trajectory
        
    def compute_returns(self, rewards: list) -> list:
        """计算折扣累积回报"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
        
    def optimize(self, trajectories: list):
        """基于轨迹优化策略"""
        all_states = []
        all_actions = []
        all_returns = []
        
        for traj in trajectories:
            returns = self.compute_returns(traj["rewards"])
            all_states.extend(traj["states"])
            all_actions.extend(traj["actions"])
            all_returns.extend(returns)
            
        # 策略梯度更新
        self.agent.update(all_states, all_actions, all_returns)
```

### 过程奖励模型（PRM）

对于复杂任务，仅基于最终结果给予奖励可能导致奖励稀疏。这就像你练习一道十步证明题，如果只在最后告诉你“对”或“错”，你很难知道哪一步出了问题。过程奖励模型解决的就是这个问题——它评估每个中间步骤的质量：

```python
class ProcessRewardModel:
    """过程奖励模型：评估每个推理步骤的质量"""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate_step(self, state: str, action: str, context: str) -> float:
        """评估单个步骤的质量"""
        prompt = f"""评估以下推理步骤的质量（0-1分）：

问题上下文：
{context}

当前状态：
{state}

执行的动作：
{action}

评分标准：
- 逻辑正确性：动作是否合理
- 进展性：是否朝目标前进
- 效率：是否有不必要的步骤

请输出一个0-1之间的分数。"""
        
        score = float(self.model.generate(prompt).strip())
        return score
        
    def evaluate_trajectory(self, trajectory: dict) -> list:
        """评估整个轨迹的每个步骤"""
        step_rewards = []
        
        for i, (state, action) in enumerate(zip(trajectory["states"], trajectory["actions"])):
            context = self._build_context(trajectory, i)
            reward = self.evaluate_step(state, action, context)
            step_rewards.append(reward)
            
        return step_rewards
```

### 蒙特卡洛树搜索（MCTS）

MCTS用于在动作空间中搜索最优策略。还记得AlphaGo吗？它就是用MCTS来决定下一步棋走哪里的。在Agent场景中，MCTS帮助智能体在多个可能的行动中搜索最优选择——每次决策时“在脑中模拟几步”，看看哪个方向最有前途：

```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        
    def ucb_score(self, c=1.41):
        """Upper Confidence Bound"""
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class AgentMCTS:
    """用于Agent决策的MCTS"""
    
    def __init__(self, agent, simulator, reward_fn, num_simulations=100):
        self.agent = agent
        self.simulator = simulator
        self.reward_fn = reward_fn
        self.num_simulations = num_simulations
        
    def search(self, state) -> str:
        """搜索最优动作"""
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            node = self._select(root)
            if not self._is_terminal(node):
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            
        # 选择访问次数最多的子节点
        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return best_action
        
    def _select(self, node):
        """选择：沿着UCB最高的路径下降"""
        while node.children and not self._is_terminal(node):
            node = max(node.children.values(), key=lambda n: n.ucb_score())
        return node
        
    def _expand(self, node):
        """扩展：添加新的子节点"""
        possible_actions = self.agent.get_possible_actions(node.state)
        for action in possible_actions:
            if action not in node.children:
                next_state = self.simulator.step(node.state, action)
                child = MCTSNode(next_state, parent=node)
                node.children[action] = child
        return random.choice(list(node.children.values()))
        
    def _simulate(self, node):
        """模拟：随机策略rollout"""
        state = node.state
        total_reward = 0
        
        for _ in range(10):  # 最多模拟10步
            if self._is_terminal_state(state):
                break
            action = self.agent.sample_action(state)
            state, reward, done = self.simulator.step(state, action)
            total_reward += reward
            if done:
                break
                
        return total_reward
        
    def _backpropagate(self, node, reward):
        """回溯：更新路径上所有节点的统计"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
```

## 典型算法

### GRPO用于Agent优化

GRPO（Group Relative Policy Optimization）可以扩展到Agent场景。它的核心思想很直觉：让Agent对同一个任务尝试多种不同的解法，然后比较哪个解法效果更好——加强好的、削弱差的。就像一个厨师尝试四种不同的配方做同一道菜，让食客打分后，记住得分高的配方的做法：

```python
class AgentGRPO:
    """用于Agent的GRPO优化"""
    
    def __init__(self, agent, reward_model, group_size=4):
        self.agent = agent
        self.reward_model = reward_model
        self.group_size = group_size
        
    def optimize_step(self, task):
        """单步优化"""
        # 采样多条轨迹
        trajectories = []
        for _ in range(self.group_size):
            traj = self.collect_trajectory(task)
            trajectories.append(traj)
            
        # 计算轨迹奖励
        rewards = [self.evaluate_trajectory(t) for t in trajectories]
        
        # 组内相对排序
        mean_reward = np.mean(rewards)
        advantages = [(r - mean_reward) / (np.std(rewards) + 1e-8) for r in rewards]
        
        # 策略更新：正优势增加概率，负优势减少概率
        for traj, adv in zip(trajectories, advantages):
            if adv > 0:
                self.agent.reinforce(traj, weight=adv)
            else:
                self.agent.penalize(traj, weight=-adv)
                
    def evaluate_trajectory(self, trajectory) -> float:
        """评估轨迹总奖励"""
        # 可以使用最终奖励
        final_reward = trajectory["rewards"][-1] if trajectory["rewards"] else 0
        
        # 也可以结合过程奖励
        process_reward = sum(trajectory["rewards"]) / len(trajectory["rewards"])
        
        return 0.5 * final_reward + 0.5 * process_reward
```

### STaR：自我教学推理

STaR（Self-Taught Reasoner）让模型通过自己的成功经验学习。这就像一个学生做了100道题，其中40道做对了。他不是单纯记住答案，而是仔细分析“我做对的那些题，解题思路是什么”，然后将这些成功的思路内化为自己的能力：

```python
class STaRTrainer:
    """STaR训练器：从成功轨迹中学习"""
    
    def __init__(self, agent, verifier):
        self.agent = agent
        self.verifier = verifier
        self.successful_trajectories = []
        
    def collect_and_filter(self, tasks: list, num_samples=10):
        """收集并筛选成功的轨迹"""
        for task in tasks:
            for _ in range(num_samples):
                trajectory = self.agent.generate_trajectory(task)
                
                # 验证是否成功
                if self.verifier.check(task, trajectory):
                    self.successful_trajectories.append({
                        "task": task,
                        "trajectory": trajectory
                    })
                    
    def rationalize(self, task, correct_answer):
        """为已知答案生成推理过程"""
        prompt = f"""任务：{task}
正确答案：{correct_answer}

请生成得到这个答案的推理过程。"""
        
        rationalization = self.agent.generate(prompt)
        return rationalization
        
    def train(self):
        """用成功轨迹微调模型"""
        training_data = []
        
        for item in self.successful_trajectories:
            # 将轨迹转换为训练样本
            input_text = item["task"]
            output_text = self._format_trajectory(item["trajectory"])
            training_data.append((input_text, output_text))
            
        self.agent.finetune(training_data)
```

## 环境设计

Agentic RL的一个关键要素是环境设计——智能体需要一个可以交互、能给出反馈的“训练场”。这就像飞行员训练需要飞行模拟器，智能体也需要一个能够模拟真实任务场景的环境。

### 任务环境接口

```python
from abc import ABC, abstractmethod

class AgentEnvironment(ABC):
    """Agent交互环境的抽象接口"""
    
    @abstractmethod
    def reset(self, task: str) -> str:
        """重置环境，返回初始状态"""
        pass
        
    @abstractmethod
    def step(self, action: str) -> tuple:
        """执行动作，返回(next_state, reward, done)"""
        pass
        
    @abstractmethod
    def get_available_actions(self) -> list:
        """获取当前可用的动作列表"""
        pass


class CodeExecutionEnvironment(AgentEnvironment):
    """代码执行环境"""
    
    def __init__(self):
        self.state = None
        self.history = []
        self.sandbox = CodeSandbox()
        
    def reset(self, task: str) -> str:
        self.state = {"task": task, "code": "", "outputs": []}
        self.history = []
        return self._format_state()
        
    def step(self, action: str) -> tuple:
        # 解析动作
        action_type, action_content = self._parse_action(action)
        
        if action_type == "write_code":
            self.state["code"] = action_content
            reward = 0
            done = False
            
        elif action_type == "execute":
            result = self.sandbox.execute(self.state["code"])
            self.state["outputs"].append(result)
            
            # 根据执行结果给予奖励
            if result["success"]:
                reward = 0.5
                # 检查是否完成任务
                done = self._check_completion()
                if done:
                    reward = 1.0
            else:
                reward = -0.1
                done = False
                
        elif action_type == "submit":
            done = True
            reward = self._evaluate_submission()
            
        self.history.append({"action": action, "state": self._format_state()})
        
        return self._format_state(), reward, done
```

### 奖励塑形

设计良好的奖励函数是Agentic RL成功的关键。这就像设计一个好的评分体系：只看最终成绩太粗糙，还得考虑学习态度、进步幅度、方法是否高效等多个维度：

```python
class RewardShaper:
    """奖励塑形器：将稀疏奖励转化为密集奖励"""
    
    def __init__(self, task_evaluator, progress_evaluator):
        self.task_eval = task_evaluator
        self.progress_eval = progress_evaluator
        
    def compute_reward(self, state, action, next_state, done) -> float:
        reward = 0
        
        # 1. 任务完成奖励（稀疏）
        if done:
            reward += self.task_eval.evaluate(next_state) * 10
            
        # 2. 进度奖励（密集）
        progress_before = self.progress_eval.estimate(state)
        progress_after = self.progress_eval.estimate(next_state)
        progress_reward = progress_after - progress_before
        reward += progress_reward
        
        # 3. 效率惩罚（鼓励简洁）
        action_length = len(action)
        efficiency_penalty = -0.001 * action_length
        reward += efficiency_penalty
        
        # 4. 错误惩罚
        if self._is_error(next_state):
            reward -= 0.5
            
        return reward
```

Agentic RL为智能体提供了从环境反馈中持续学习的能力。尽管训练成本较高，但它能够让智能体习得难以通过监督学习获取的复杂策略。就像一个棋手只靠背棋谱无法成为大师，必须通过实战中不断输赢来练就直觉和判断力——Agentic RL正是给了智能体这种“在实战中成长”的能力，是构建高级智能体的重要技术路径。
