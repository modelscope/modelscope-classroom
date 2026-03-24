# 通用Benchmark

大模型评测需要标准化的基准数据集（Benchmark）来量化模型能力。本节将系统介绍业界主流的评测基准，涵盖知识推理、思维链评测、多语言能力以及Agent评测等多个维度。

## 知识与推理评测

### MMLU

MMLU（Massive Multitask Language Understanding）是衡量大模型知识广度和推理能力的核心基准。数据集涵盖57个学科，包括STEM、人文、社会科学和其他领域，共约16000道多选题。

| 学科类别 | 示例科目 | 题目数量 |
|----------|----------|----------|
| STEM | 数学、物理、计算机科学、化学 | ~4000 |
| 人文学科 | 历史、哲学、法律 | ~3000 |
| 社会科学 | 经济学、心理学、政治学 | ~3500 |
| 其他 | 临床医学、职业资格 | ~5500 |

MMLU采用5-shot评测方式，即提供5个示例后让模型回答新问题。评测指标为准确率。

```python
# MMLU评测示例
prompt = """
The following are multiple choice questions (with answers) about world history.

Question: What was the main cause of World War I?
A. The assassination of Archduke Franz Ferdinand
B. The invasion of Poland
C. The signing of the Treaty of Versailles
D. The fall of the Berlin Wall
Answer: A

Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Answer:"""
```

### MMLU-Pro

MMLU-Pro是MMLU的增强版本，具有以下特点：
- 选项从4个增加到10个，降低随机猜测成功率
- 问题难度提升，需要更深层次的推理
- 减少了可通过模式匹配解决的简单问题

### ARC（AI2 Reasoning Challenge）

ARC专注于科学推理能力评测，分为Easy和Challenge两个难度级别：
- **ARC-Easy**：约5500道相对简单的科学题
- **ARC-Challenge**：约2600道需要复杂推理的题目

### HellaSwag

HellaSwag评测模型的常识推理能力，要求模型选择最合理的句子续写：

```
Context: A man is seen sitting on a couch playing a guitar. He plays for 
a bit then stops and...
A. begins to talk to the camera.
B. takes off his shoes and throws them.
C. starts doing jumping jacks.
D. continues playing the guitar.
```

该数据集通过对抗性筛选确保人类容易回答但模型容易出错。

### WinoGrande

WinoGrande评测代词消解能力，是对Winograd Schema Challenge的大规模扩展：

```
The trophy doesn't fit into the brown suitcase because it is too large.
Question: What is "it" referring to?
A. trophy
B. suitcase
```

## 数学推理评测

### GSM8K

GSM8K（Grade School Math 8K）包含约8500道小学数学应用题，要求模型展示多步数学推理能力：

```
Question: James has 30 teeth. His dentist drills 4 of them and caps 7 
more teeth than he drills. What percentage of James' teeth does the 
dentist fix?

Answer: The dentist caps 4 + 7 = 11 teeth.
So the dentist fixes 4 + 11 = 15 teeth.
The percentage is 15 / 30 * 100 = 50%.
The answer is 50.
```

GSM8K常与思维链（Chain-of-Thought）提示结合使用，评测模型的逐步推理能力。

### MATH

MATH数据集包含12500道竞赛级数学题，难度从高中到数学竞赛，按难度分为1-5级：

| 难度等级 | 描述 | 典型主题 |
|----------|------|----------|
| Level 1 | 基础 | 简单代数、几何 |
| Level 2 | 中等 | 多项式、概率 |
| Level 3 | 较难 | 数论、组合 |
| Level 4 | 困难 | 抽象代数、复分析 |
| Level 5 | 竞赛级 | IMO级别问题 |

### MathBench

MathBench是针对中文数学能力的评测集，涵盖小学到大学各阶段：
- 算术计算
- 代数运算
- 几何证明
- 微积分

## 思维链评测

### CoT评测方法

思维链（Chain-of-Thought）评测不仅关注最终答案的正确性，还评估推理过程的质量。评测维度包括：

**推理完整性**：推理步骤是否完整覆盖问题所需的所有环节

**逻辑一致性**：各步骤之间是否存在逻辑矛盾

**计算准确性**：中间计算步骤是否正确

```python
def evaluate_cot(model_output, reference):
    """评测思维链输出"""
    # 提取最终答案
    final_answer = extract_answer(model_output)
    answer_correct = final_answer == reference['answer']
    
    # 评估推理步骤（可使用LLM-as-Judge）
    steps = extract_reasoning_steps(model_output)
    step_scores = []
    for i, step in enumerate(steps):
        score = evaluate_step(step, reference.get('steps', []))
        step_scores.append(score)
    
    return {
        'answer_correct': answer_correct,
        'reasoning_score': np.mean(step_scores) if step_scores else 0,
        'num_steps': len(steps)
    }
```

### BBH（BIG-Bench Hard）

BBH是从BIG-Bench中筛选出的23个挑战性任务子集，这些任务在使用思维链提示后性能显著提升：

- 日期理解
- 因果判断
- 逻辑推演
- 导航推理
- 体育理解

### ThoughtBench

ThoughtBench专门评测模型的深度思考能力，包含需要多轮迭代推理的复杂问题。

## 代码能力评测

### HumanEval

HumanEval包含164道手工编写的Python编程题，评测模型的代码生成能力：

```python
def candidate_function(prompt, num_samples=100):
    """
    生成代码候选并执行测试
    """
    completions = model.generate(prompt, n=num_samples)
    
    passed = 0
    for completion in completions:
        try:
            # 执行生成的代码
            exec(prompt + completion)
            # 运行测试用例
            if run_tests():
                passed += 1
        except:
            continue
    
    # 计算pass@k
    return passed / num_samples
```

**Pass@k**指标：生成k个候选代码，至少有一个通过测试的概率。

### MBPP

MBPP（Mostly Basic Python Problems）包含974道入门级Python题，覆盖基础数据结构和算法。

### MultiPL-E

MultiPL-E将HumanEval扩展到18种编程语言，评测多语言代码生成能力。

### CodeContests

CodeContests来自Codeforces等在线编程竞赛平台，难度较高，需要算法设计能力。

## 中文能力评测

### C-Eval

C-Eval是中文能力综合评测基准，涵盖52个学科，共约14000道题：

| 类别 | 学科示例 | 题目数 |
|------|----------|--------|
| STEM | 高等数学、大学物理、程序设计 | ~5000 |
| 社会科学 | 马克思主义、毛泽东思想、政治 | ~3000 |
| 人文学科 | 中国历史、法律、教育学 | ~3000 |
| 其他 | 临床医学、注册会计师 | ~3000 |

### CMMLU

CMMLU专注于中国文化和知识背景下的语言理解，包含67个主题。

### AGIEval

AGIEval收集了中国高考、公务员考试、法律职业资格考试等真实考试题目，评测实际应用场景下的能力。

### GAOKAO-Bench

GAOKAO-Bench使用中国高考真题，涵盖语文、数学、英语、理综、文综等科目。

## 英文综合评测

### GLUE与SuperGLUE

GLUE（General Language Understanding Evaluation）是经典的英文理解评测套件：

| 任务 | 类型 | 描述 |
|------|------|------|
| CoLA | 可接受性 | 判断句子语法是否正确 |
| SST-2 | 情感分析 | 电影评论情感分类 |
| MRPC | 释义检测 | 判断句子对是否语义等价 |
| QQP | 问题匹配 | 判断问题对是否相似 |
| STS-B | 语义相似度 | 评分0-5的相似度 |
| MNLI | 自然语言推理 | 蕴含/矛盾/中性 |
| QNLI | 问答NLI | 答案是否在段落中 |
| RTE | 文本蕴含 | 二分类蕴含判断 |
| WNLI | 代词消解 | Winograd风格任务 |

SuperGLUE在GLUE基础上提升了难度，增加了BoolQ、CB、COPA等更具挑战性的任务。

### TriviaQA

TriviaQA是大规模问答数据集，包含95K个问答对，评测事实知识检索能力。

### NaturalQuestions

来自Google搜索的真实用户问题，答案来自Wikipedia，分为长答案和短答案两种形式。

## Agent评测

### AgentBench

AgentBench评测大模型作为智能体执行复杂任务的能力，涵盖8个场景：

| 场景 | 描述 | 评测内容 |
|------|------|----------|
| OS操作 | 命令行交互 | bash命令执行 |
| 数据库 | SQL查询 | 数据检索与分析 |
| 知识图谱 | 图谱推理 | 实体关系查询 |
| 数字游戏 | 博弈策略 | 决策规划 |
| 横向思维 | 创意推理 | 开放性问题 |
| 家居环境 | 智能家居 | 设备控制 |
| 网页浏览 | 信息检索 | 网页交互 |
| 购物助手 | 电商场景 | 商品推荐 |

### ToolBench

ToolBench评测模型使用外部工具的能力，包含16000个工具API和超过50万个工具调用实例。

评测维度：
- 工具选择准确性
- 参数填充正确性
- 多工具组合能力
- 错误恢复能力

### WebArena

WebArena在真实网站环境中评测Agent能力，任务包括：
- 购物网站操作
- 社交媒体发帖
- 代码仓库管理
- 论坛讨论交互

```python
# Agent评测框架示例
class AgentEvaluator:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        
    def evaluate_episode(self, task):
        observation = self.env.reset(task)
        done = False
        trajectory = []
        
        while not done:
            # 模型决策
            action = self.model.act(observation)
            trajectory.append(action)
            
            # 环境交互
            observation, reward, done, info = self.env.step(action)
            
        # 评估
        success = self.env.check_success(task)
        efficiency = len(trajectory)
        
        return {
            'success': success,
            'steps': efficiency,
            'trajectory': trajectory
        }
```

### GAIA

GAIA评测真实世界助手能力，任务需要：
- 网络搜索
- 文件处理
- 多步骤规划
- 工具组合使用

## 多模态评测

### VQA与VQAv2

视觉问答任务，给定图像和问题，模型需要生成答案。

### MMBench

MMBench是综合多模态评测基准，评测视觉语言模型在20+细粒度能力维度上的表现。

### SEED-Bench

SEED-Bench涵盖12个评测维度，包含约19000道选择题，支持图像和视频理解评测。

## 评测基准对比

| 评测集 | 主要能力 | 规模 | 形式 | 语言 |
|--------|----------|------|------|------|
| MMLU | 知识推理 | 16K | 多选 | 英文 |
| C-Eval | 知识推理 | 14K | 多选 | 中文 |
| GSM8K | 数学推理 | 8.5K | 生成 | 英文 |
| HumanEval | 代码生成 | 164 | 生成 | Python |
| HellaSwag | 常识推理 | 70K | 多选 | 英文 |
| AgentBench | Agent能力 | 8场景 | 交互 | 多语言 |

## 评测实践建议

**选择合适的评测集**：根据模型目标用途选择相关的评测基准，避免过度优化单一指标。

**注意数据污染**：使用最新发布的评测集，或对已知评测集进行变体设计。

**多维度综合评估**：单一评测集难以全面反映模型能力，应组合多个基准进行评测。

**关注评测细节**：相同评测集在不同prompt、采样参数下结果可能差异显著，需要标准化评测流程。

```python
# 标准化评测配置示例
eval_config = {
    'temperature': 0.0,          # 确定性输出
    'max_tokens': 1024,
    'prompt_template': 'standard_v1',
    'num_few_shot': 5,
    'seed': 42,
    'batch_size': 32
}
```

通过系统化地使用这些评测基准，可以获得对大模型能力的全面认识，为模型开发和选型提供可靠依据。
