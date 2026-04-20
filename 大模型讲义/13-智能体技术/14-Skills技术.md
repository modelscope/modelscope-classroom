# Skills技术

## 1. 从工具到技能的演进

假设你刚招了一位新员工，他会用所有办公软件（工具），但不懂你们行业的业务流程和专业知识。而一位资深专家不仅会用工具，还掌握了完整的工作方法论和领域经验。Skills 技术解决的正是这个问题——如何让智能体从“会用工具的新手”变成“掌握专业知识的专家”。

MCP协议解决了智能体与外部工具的标准化交互问题，但在实际应用中，开发者面临一个新挑战：**如何让智能体像人类专家一样，根据任务需求动态加载和组合专业能力？**

工具（Tools）提供的是原子化的操作能力，如“读取文件”、“查询数据库”；而技能（Skills）则是更高层次的抽象——包含领域知识、工作流程和最佳实践的完整能力包。举个例子，`read_file` 是一个工具，而 `code_review`（包括安全审查清单、性能分析框架、代码风格规范）则是一个技能：

| 概念 | 粒度 | 示例 | 特点 |
|------|------|------|------|
| Tool | 原子操作 | `read_file`, `http_request` | 单一功能，无领域知识 |
| Skill | 领域能力 | `code_review`, `financial_analysis` | 包含专业知识和工作流 |

## 2. Agent Skills 协议

### 2.1 设计理念

Anthropic 提出的 Agent Skills 协议是一种让智能体动态获取专业能力的机制。继续用专家的比喻：你不可能让每位员工都同时掌握所有领域的知识，而是根据任务需求调动相应的专家。其核心思想是：

1. **按需加载**：不是所有技能都预置在系统提示中，而是根据任务需求动态加载
2. **领域专精**：每个 Skill 封装特定领域的专业知识
3. **可组合性**：多个 Skills 可以协同工作，解决复杂问题
4. **标准化格式**：统一的 Skill 定义规范，便于共享和复用

### 2.2 Skill 的结构

一个标准的 Skill 定义就像一位专家的“履历”——包含他的名称、专业领域、擅长什么、在什么情况下应该找他、以及他的工作方法论。以下是一个代码审查技能的完整定义：

```yaml
# skill.yaml
name: code-review
version: "1.0.0"
description: "专业的代码审查技能，提供安全性、性能和可维护性分析"

# 触发条件：何时自动应用此技能
triggers:
  - pattern: "review.*code"
  - pattern: "check.*security"
  - file_types: [".py", ".js", ".ts", ".go"]

# 技能指令：注入到Agent的专业知识
instructions: |
  你现在是一位资深代码审查专家。进行代码审查时，请从以下维度分析：
  
  ## 安全性检查
  - SQL注入风险
  - XSS漏洞
  - 敏感信息泄露
  - 权限控制缺陷
  
  ## 性能分析
  - 算法复杂度
  - 内存使用
  - 数据库查询效率
  - 缓存策略
  
  ## 代码质量
  - 命名规范
  - 函数单一职责
  - 错误处理完整性
  - 测试覆盖率建议
  
  对于每个发现的问题，提供：
  1. 问题描述
  2. 风险等级（高/中/低）
  3. 具体修改建议
  4. 修改后的代码示例

# 关联的工具
tools:
  - name: static_analysis
    description: "运行静态代码分析"
  - name: security_scan
    description: "执行安全漏洞扫描"

# 输出格式模板
output_format: |
  ## 代码审查报告
  
  ### 概述
  - 审查文件：{files}
  - 发现问题：{issue_count}
  
  ### 问题详情
  {issues}
  
  ### 改进建议
  {suggestions}
```

### 2.3 Skill 加载机制

理解了 Skill 的结构，接下来看它如何在代码中实现。核心机制是：当用户提出请求时，系统自动匹配触发条件，加载相应的技能，并将专业知识注入到系统提示中。这就像公司前台根据客户需求自动派单给对口专家：

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import yaml
import re

@dataclass
class Skill:
    """技能定义"""
    name: str
    version: str
    description: str
    instructions: str
    triggers: List[dict] = field(default_factory=list)
    tools: List[dict] = field(default_factory=list)
    output_format: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Skill':
        """从YAML文件加载技能"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def matches(self, query: str, file_types: List[str] = None) -> bool:
        """检查是否匹配触发条件"""
        for trigger in self.triggers:
            # 模式匹配
            if 'pattern' in trigger:
                if re.search(trigger['pattern'], query, re.IGNORECASE):
                    return True
            # 文件类型匹配
            if 'file_types' in trigger and file_types:
                if any(ft in trigger['file_types'] for ft in file_types):
                    return True
        return False


class SkillRegistry:
    """技能注册表"""
    
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        
    def register(self, skill: Skill):
        """注册技能"""
        self.skills[skill.name] = skill
        
    def load_from_directory(self, directory: str):
        """从目录加载所有技能"""
        import os
        for filename in os.listdir(directory):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                skill_path = os.path.join(directory, filename)
                skill = Skill.from_yaml(skill_path)
                self.register(skill)
                
    def find_matching_skills(
        self, 
        query: str, 
        file_types: List[str] = None,
        max_skills: int = 3
    ) -> List[Skill]:
        """查找匹配的技能"""
        matching = []
        for skill in self.skills.values():
            if skill.matches(query, file_types):
                matching.append(skill)
                if len(matching) >= max_skills:
                    break
        return matching


class SkillAwareAgent:
    """具备技能感知能力的Agent"""
    
    def __init__(self, llm_client, skill_registry: SkillRegistry):
        self.llm = llm_client
        self.registry = skill_registry
        self.active_skills: List[Skill] = []
        
    def _build_system_prompt(self, base_prompt: str) -> str:
        """构建包含技能指令的系统提示"""
        if not self.active_skills:
            return base_prompt
            
        skill_instructions = "\n\n".join([
            f"## {skill.name}\n{skill.instructions}"
            for skill in self.active_skills
        ])
        
        return f"""{base_prompt}

你已加载以下专业技能，请在回答中运用这些专业知识：

{skill_instructions}
"""
    
    def process(self, query: str, context: dict = None) -> str:
        """处理用户请求"""
        # 1. 自动匹配并加载技能
        file_types = context.get('file_types', []) if context else []
        self.active_skills = self.registry.find_matching_skills(query, file_types)
        
        # 2. 构建增强的系统提示
        system_prompt = self._build_system_prompt("你是一个智能助手。")
        
        # 3. 调用LLM
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        
        # 4. 如有输出格式要求，进行格式化
        if self.active_skills and self.active_skills[0].output_format:
            response = self._format_output(response, self.active_skills[0])
            
        return response
```

## 3. Skill 类型与设计模式

### 3.1 按领域分类

就像公司有不同部门的专家一样，Skills 也按领域分类：

| 领域 | 技能示例 | 核心能力 |
|------|----------|----------|
| 软件开发 | `code-review`, `refactoring`, `testing` | 代码分析、重构建议、测试生成 |
| 数据分析 | `data-cleaning`, `visualization`, `statistics` | 数据处理、图表生成、统计分析 |
| 文档写作 | `technical-writing`, `copywriting`, `translation` | 专业写作、风格转换、多语言支持 |
| 研究调研 | `literature-review`, `fact-checking`, `synthesis` | 文献分析、事实核查、信息综合 |

### 3.2 Skill 组合模式

复杂任务往往需要多种技能协同工作。这就像一个项目需要多个部门配合——查代码时可能需要安全专家、性能专家和架构师同时给出意见。组合方式主要有两种：

**串行组合**：多个技能按顺序执行，就像流水线上的工序——先分析需求、再生成代码、然后审查、最后生成测试：

```python
class SkillPipeline:
    """技能流水线"""
    
    def __init__(self, skills: List[Skill]):
        self.skills = skills
        
    def execute(self, agent, initial_input: str) -> str:
        result = initial_input
        for skill in self.skills:
            agent.active_skills = [skill]
            result = agent.process(result)
        return result

# 示例：代码开发流水线
pipeline = SkillPipeline([
    skill_registry.get("requirement-analysis"),  # 需求分析
    skill_registry.get("code-generation"),        # 代码生成
    skill_registry.get("code-review"),            # 代码审查
    skill_registry.get("test-generation"),        # 测试生成
])
```

**并行组合**：多个技能同时应用

```python
class ParallelSkills:
    """并行技能执行"""
    
    def __init__(self, skills: List[Skill]):
        self.skills = skills
        
    async def execute(self, agent, query: str) -> List[str]:
        import asyncio
        
        async def run_skill(skill):
            agent_copy = copy.deepcopy(agent)
            agent_copy.active_skills = [skill]
            return await asyncio.to_thread(agent_copy.process, query)
            
        tasks = [run_skill(skill) for skill in self.skills]
        results = await asyncio.gather(*tasks)
        return results

# 示例：多角度代码分析
parallel = ParallelSkills([
    skill_registry.get("security-analysis"),
    skill_registry.get("performance-analysis"),
    skill_registry.get("maintainability-analysis"),
])
```

### 3.3 自适应 Skill 选择

```python
class AdaptiveSkillSelector:
    """自适应技能选择器"""
    
    def __init__(self, registry: SkillRegistry, llm_client):
        self.registry = registry
        self.llm = llm_client
        
    def select_skills(self, query: str, max_skills: int = 3) -> List[Skill]:
        """使用LLM智能选择技能"""
        
        # 构建技能列表描述
        skill_descriptions = "\n".join([
            f"- {name}: {skill.description}"
            for name, skill in self.registry.skills.items()
        ])
        
        prompt = f"""分析以下用户请求，选择最适合的技能（最多{max_skills}个）：

用户请求：{query}

可用技能：
{skill_descriptions}

以JSON格式返回选中的技能名称列表，例如：["skill1", "skill2"]
只返回JSON，不要其他内容。"""

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        import json
        try:
            selected_names = json.loads(response)
            return [self.registry.skills[name] for name in selected_names 
                    if name in self.registry.skills]
        except (json.JSONDecodeError, KeyError):
            # 降级到规则匹配
            return self.registry.find_matching_skills(query)
```

## 4. 实践：构建专业技能

理解了原理之后，来看两个实际的 Skill 示例。第一个是金融分析技能——注意它的 instructions 里包含了完整的财务分析框架，这就是“专业知识”的具体体现。

### 4.1 金融分析技能

```yaml
# financial_analysis.yaml
name: financial-analysis
version: "1.0.0"
description: "专业金融数据分析，包括财务报表解读、估值模型和风险评估"

triggers:
  - pattern: "分析.*财报"
  - pattern: "估值.*公司"
  - pattern: "financial.*analysis"

instructions: |
  你是一位资深金融分析师，具备CFA持证人的专业能力。
  
  ## 财务报表分析框架
  
  ### 盈利能力指标
  - 毛利率 = (营业收入 - 营业成本) / 营业收入
  - 净利率 = 净利润 / 营业收入
  - ROE = 净利润 / 平均股东权益
  - ROA = 净利润 / 平均总资产
  
  ### 偿债能力指标
  - 流动比率 = 流动资产 / 流动负债（健康值 > 2）
  - 速动比率 = (流动资产 - 存货) / 流动负债（健康值 > 1）
  - 资产负债率 = 总负债 / 总资产
  
  ### 运营效率指标
  - 存货周转率 = 营业成本 / 平均存货
  - 应收账款周转率 = 营业收入 / 平均应收账款
  - 总资产周转率 = 营业收入 / 平均总资产
  
  ### 估值指标
  - P/E = 股价 / 每股收益
  - P/B = 股价 / 每股净资产
  - EV/EBITDA = 企业价值 / 息税折旧摊销前利润
  
  ## 分析要点
  1. 纵向对比：与公司历史数据对比，识别趋势
  2. 横向对比：与同行业公司对比，评估竞争地位
  3. 杜邦分析：分解ROE，找出驱动因素
  4. 风险识别：关注异常波动和潜在风险

output_format: |
  ## 财务分析报告
  
  ### 公司概况
  {company_overview}
  
  ### 核心指标
  | 指标 | 数值 | 同比变化 | 行业对比 |
  |------|------|----------|----------|
  {metrics_table}
  
  ### 分析结论
  {analysis}
  
  ### 风险提示
  {risks}
  
  ### 投资建议
  {recommendation}
```

### 4.2 代码重构技能

第二个示例是代码重构技能。注意它与前面 code-review 的区别：review 关注“发现问题”，而 refactoring 关注“改善结构”。这种细粒度的专业分工正是 Skills 的价值所在：

```yaml
# code_refactoring.yaml
name: code-refactoring
version: "1.0.0"
description: "识别代码坏味道并提供重构建议"

triggers:
  - pattern: "重构"
  - pattern: "refactor"
  - pattern: "优化.*代码"

instructions: |
  你是一位精通设计模式和重构技术的架构师。
  
  ## 常见代码坏味道
  
  ### 1. 重复代码 (Duplicated Code)
  - 识别方法：相似代码块出现多次
  - 重构手法：Extract Method, Pull Up Method
  
  ### 2. 过长函数 (Long Method)
  - 识别方法：函数超过20行，包含多个职责
  - 重构手法：Extract Method, Replace Temp with Query
  
  ### 3. 过大类 (Large Class)
  - 识别方法：类承担过多职责，字段过多
  - 重构手法：Extract Class, Extract Subclass
  
  ### 4. 过长参数列表 (Long Parameter List)
  - 识别方法：方法参数超过3个
  - 重构手法：Introduce Parameter Object, Preserve Whole Object
  
  ### 5. 发散式变化 (Divergent Change)
  - 识别方法：一个类因多种原因被修改
  - 重构手法：Extract Class
  
  ### 6. 霰弹式修改 (Shotgun Surgery)
  - 识别方法：一个改动需要修改多个类
  - 重构手法：Move Method, Move Field, Inline Class
  
  ## 重构原则
  1. 小步前进：每次只做一个小改动
  2. 保持测试通过：重构前确保有测试覆盖
  3. 提交频繁：每完成一个重构就提交
  4. 不要同时重构和添加功能

tools:
  - name: analyze_complexity
    description: "分析代码复杂度"
  - name: find_duplicates
    description: "检测重复代码"
```

## 5. Skill 生态与分发

### 5.1 Skill Hub 架构

当 Skill 越来越多时，自然需要一个“市场”来管理和分发。这就像手机应用商店——官方提供基础技能，社区贡献专业技能，企业内部有私有技能：

```
┌────────────────────────────────────────────────────────┐
│                    Skill Hub                            │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Official     │  │ Community    │  │ Enterprise   │ │
│  │ Skills       │  │ Skills       │  │ Skills       │ │
│  │              │  │              │  │              │ │
│  │ • code-review│  │ • vue-expert │  │ • internal-  │ │
│  │ • data-      │  │ • react-     │  │   policy     │ │
│  │   analysis   │  │   patterns   │  │ • compliance │ │
│  │ • writing    │  │ • ml-ops     │  │   -check     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              Skill Metadata Index                │  │
│  │  • 版本管理  • 依赖解析  • 安全审计  • 使用统计   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 5.2 Skill 包管理

就像 `pip install` 安装 Python 包一样，Skill 也需要一个包管理器来搜索、安装和更新：

```python
class SkillPackageManager:
    """技能包管理器"""
    
    def __init__(self, hub_url: str, local_path: str):
        self.hub_url = hub_url
        self.local_path = local_path
        self.installed: dict[str, Skill] = {}
        
    def search(self, keyword: str) -> List[dict]:
        """搜索技能"""
        import requests
        response = requests.get(
            f"{self.hub_url}/api/skills/search",
            params={"q": keyword}
        )
        return response.json()
        
    def install(self, skill_name: str, version: str = "latest"):
        """安装技能"""
        import requests
        import os
        
        # 下载技能定义
        response = requests.get(
            f"{self.hub_url}/api/skills/{skill_name}/{version}"
        )
        skill_data = response.json()
        
        # 保存到本地
        skill_path = os.path.join(self.local_path, f"{skill_name}.yaml")
        with open(skill_path, 'w', encoding='utf-8') as f:
            yaml.dump(skill_data, f, allow_unicode=True)
            
        # 加载到内存
        self.installed[skill_name] = Skill.from_yaml(skill_path)
        
        print(f"已安装技能: {skill_name}@{version}")
        
    def update(self, skill_name: str = None):
        """更新技能"""
        if skill_name:
            self.install(skill_name, "latest")
        else:
            for name in list(self.installed.keys()):
                self.install(name, "latest")
                
    def list_installed(self) -> List[str]:
        """列出已安装技能"""
        return list(self.installed.keys())
```

## 6. Skills 与 MCP 的协同

Skills 和 MCP 是互补的两个层次——Skills 提供“脑力”（专业知识和方法论），MCP 提供“手力”（具体工具执行）。比如你请一位安全专家审查代码，Skill 提供了审查清单和专业知识，而 MCP 提供了实际读取文件和运行扫描工具的能力：

```
用户请求: "审查这个 Python 文件的安全性"
          │
          ▼
┌─────────────────────────────────────────┐
│         Skill Layer (高层抽象)          │
│                                         │
│  code-review Skill 被激活               │
│  注入专业知识：                          │
│  • 安全审查清单                          │
│  • 常见漏洞模式                          │
│  • 风险评估标准                          │
│                                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         MCP Layer (底层工具)            │
│                                         │
│  Agent 调用 MCP 工具：                   │
│  • filesystem.read_file                 │
│  • static_analyzer.scan                 │
│  • security_scanner.check               │
│                                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Output (结构化输出)             │
│                                         │
│  按 Skill 定义的格式输出审查报告         │
│                                         │
└─────────────────────────────────────────┘
```

**协同实现示例**：

```python
class SkillMCPAgent:
    """集成 Skills 和 MCP 的 Agent"""
    
    def __init__(self, llm, skill_registry, mcp_client):
        self.llm = llm
        self.skills = skill_registry
        self.mcp = mcp_client
        
    def process(self, query: str, context: dict = None):
        # 1. 选择技能
        active_skills = self.skills.find_matching_skills(query)
        
        # 2. 收集技能关联的工具
        skill_tools = []
        for skill in active_skills:
            skill_tools.extend(skill.tools)
            
        # 3. 合并 MCP 工具
        mcp_tools = self.mcp.list_tools()
        all_tools = self._merge_tools(mcp_tools, skill_tools)
        
        # 4. 构建增强提示
        system_prompt = self._build_prompt(active_skills)
        
        # 5. 执行 ReACT 循环
        return self._react_loop(system_prompt, query, all_tools)
```

## 7. 最佳实践

### 7.1 Skill 设计原则

1. **单一领域**：每个 Skill 聚焦一个专业领域
2. **明确触发**：触发条件清晰，避免误激活
3. **指令精炼**：instructions 简洁有力，避免冗长
4. **格式规范**：提供清晰的输出格式模板
5. **版本管理**：使用语义化版本号

### 7.2 常见陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|----------|
| Skill 过载 | 一次加载过多技能，提示词过长 | 限制同时激活的技能数量 |
| 触发冲突 | 多个技能同时匹配 | 设置优先级，使用互斥组 |
| 知识过时 | Skill 中的专业知识已过期 | 建立定期更新机制 |
| 工具缺失 | Skill 引用的工具未安装 | 依赖检查和自动安装 |

回顾本节，Skills 技术是智能体从“通用助手”向“领域专家”演进的关键。它的核心价值在于将专业知识封装为可复用、可组合的能力包——就像招募专家一样，每个 Skill 都带着完整的领域经验和工作方法论。理解 Skills 与 MCP 的分工协作，是构建专业级智能体应用的重要一步。
