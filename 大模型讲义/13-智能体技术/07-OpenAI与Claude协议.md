# OpenAI与Claude协议

主流LLM提供商（OpenAI、Anthropic等）定义了各自的函数调用（Function Calling）和工具使用协议。理解这些协议的设计理念和使用方式，是构建跨平台智能体应用的基础。

## OpenAI Function Calling

### 基本概念

OpenAI的Function Calling允许模型识别何时需要调用外部函数，并生成符合函数签名的结构化参数。

### API格式

```python
from openai import OpenAI

client = OpenAI()

# 定义可用的函数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 调用API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"  # 让模型自行决定是否调用工具
)
```

### 响应处理

```python
message = response.choices[0].message

# 检查是否有工具调用
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # 执行函数
        if function_name == "get_weather":
            result = get_weather(**arguments)
            
        # 将结果返回给模型
        messages.append(message)  # 先添加助手的响应
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": json.dumps(result)
        })
        
    # 继续对话，让模型整合结果
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

### 并行函数调用

OpenAI支持模型同时请求多个函数调用：

```python
# 模型可能返回多个tool_calls
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "查询北京和上海的天气"}
    ],
    tools=tools,
    parallel_tool_calls=True  # 允许并行调用
)

# 处理多个调用
tool_calls = response.choices[0].message.tool_calls
results = []
for tc in tool_calls:
    # 可以并行执行
    result = execute_tool(tc.function.name, tc.function.arguments)
    results.append({
        "tool_call_id": tc.id,
        "name": tc.function.name,
        "content": json.dumps(result)
    })
```

### 强制函数调用

```python
# 强制模型调用特定函数
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)

# 禁止函数调用
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="none"
)
```

## Claude Tool Use

### 基本概念

Anthropic的Claude采用类似但略有不同的工具使用协议。Claude强调"思考优先"，会在调用工具前展示推理过程。

### API格式

```python
import anthropic

client = anthropic.Anthropic()

# 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "获取指定位置的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "上海现在的天气如何？"}
    ]
)
```

### 响应结构

Claude的响应可能包含多个内容块：

```python
for content_block in response.content:
    if content_block.type == "text":
        # 模型的文本响应（包括思考过程）
        print(f"Text: {content_block.text}")
        
    elif content_block.type == "tool_use":
        # 工具调用请求
        tool_name = content_block.name
        tool_input = content_block.input
        tool_use_id = content_block.id
        
        # 执行工具
        result = execute_tool(tool_name, tool_input)
        
        # 构建工具结果消息
        tool_result = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": json.dumps(result)
        }
```

### 完整对话流程

```python
def chat_with_tools(user_message: str, tools: list):
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # 检查是否需要调用工具
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        
        if not tool_uses:
            # 没有工具调用，返回最终响应
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks)
            
        # 处理工具调用
        assistant_content = response.content
        tool_results = []
        
        for tool_use in tool_uses:
            result = execute_tool(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result)
            })
            
        # 添加助手响应和工具结果
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})
```

## 协议对比

| 特性 | OpenAI | Claude |
|------|--------|--------|
| 参数定义 | JSON Schema | JSON Schema |
| 参数字段名 | parameters | input_schema |
| 响应格式 | tool_calls数组 | content块列表 |
| 结果传递 | tool角色消息 | tool_result内容类型 |
| 并行调用 | 明确支持 | 隐式支持 |
| 思考展示 | 可选 | 默认展示 |

### 统一抽象层

为了实现跨平台兼容，可以构建统一的抽象层：

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ToolDefinition:
    """统一的工具定义格式"""
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters
        
    def to_openai_format(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
        
    def to_claude_format(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


class ToolCall:
    """统一的工具调用格式"""
    def __init__(self, id: str, name: str, arguments: Dict):
        self.id = id
        self.name = name
        self.arguments = arguments


class LLMProvider(ABC):
    """LLM提供商抽象接口"""
    
    @abstractmethod
    def chat(self, messages: List[Dict], tools: List[ToolDefinition]) -> tuple:
        """
        返回: (text_response, tool_calls)
        """
        pass
        
    @abstractmethod
    def continue_with_tool_results(self, messages: List[Dict], tool_results: List[Dict]) -> tuple:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4"):
        self.client = OpenAI()
        self.model = model
        
    def chat(self, messages, tools):
        openai_tools = [t.to_openai_format() for t in tools]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools if openai_tools else None
        )
        
        message = response.choices[0].message
        
        text = message.content or ""
        tool_calls = []
        
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))
                
        return text, tool_calls


class ClaudeProvider(LLMProvider):
    def __init__(self, model="claude-3-opus-20240229"):
        self.client = anthropic.Anthropic()
        self.model = model
        
    def chat(self, messages, tools):
        claude_tools = [t.to_claude_format() for t in tools]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=claude_tools if claude_tools else None,
            messages=messages
        )
        
        text_parts = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))
                
        return "\n".join(text_parts), tool_calls
```

## Anthropic Agent Skills协议

Anthropic提出的Agent Skills协议是一种更高级的智能体能力描述方式：

```yaml
# skill.yaml
name: "web_search"
version: "1.0"
description: "搜索互联网获取最新信息"

triggers:
  - "搜索"
  - "查找"
  - "最新信息"
  
capabilities:
  - type: "tool"
    tool:
      name: "search"
      description: "执行网络搜索"
      parameters:
        query:
          type: "string"
          required: true

instructions: |
  当用户需要查找信息时：
  1. 分析用户意图，提取关键词
  2. 调用search工具
  3. 整理搜索结果，提供简洁回答
  
examples:
  - input: "最近有什么科技新闻？"
    output: |
      [调用search("科技新闻 2024")]
      根据搜索结果，以下是最近的科技新闻...
```

### Skills加载与执行

```python
class SkillManager:
    """技能管理器"""
    
    def __init__(self):
        self.skills = {}
        
    def load_skill(self, skill_path: str):
        """加载技能定义"""
        with open(skill_path) as f:
            skill_def = yaml.safe_load(f)
            
        skill = AgentSkill(skill_def)
        self.skills[skill.name] = skill
        
    def match_skill(self, user_input: str) -> Optional[AgentSkill]:
        """根据用户输入匹配合适的技能"""
        for skill in self.skills.values():
            if skill.matches(user_input):
                return skill
        return None
        
    def get_context_for_llm(self, matched_skills: List[AgentSkill]) -> str:
        """生成LLM的上下文指令"""
        context_parts = []
        
        for skill in matched_skills:
            context_parts.append(f"""
## 技能: {skill.name}
{skill.instructions}

可用工具:
{skill.format_tools()}

示例:
{skill.format_examples()}
""")
        
        return "\n".join(context_parts)
```

## 最佳实践

### 函数描述优化

清晰的函数描述能显著提升调用准确率：

```python
# 差的描述
{
    "name": "search",
    "description": "搜索",
    "parameters": {"query": {"type": "string"}}
}

# 好的描述
{
    "name": "web_search",
    "description": "搜索互联网获取最新信息。适用于：查找新闻、获取实时数据、验证事实。不适用于：主观问题、需要推理的问题。",
    "parameters": {
        "query": {
            "type": "string",
            "description": "搜索关键词，应包含核心概念，避免过长"
        },
        "time_range": {
            "type": "string",
            "enum": ["day", "week", "month", "year"],
            "description": "限制搜索的时间范围"
        }
    }
}
```

### 错误处理

```python
def safe_tool_execution(tool_name: str, arguments: dict) -> dict:
    """安全的工具执行包装"""
    try:
        result = tools[tool_name](**arguments)
        return {"success": True, "result": result}
    except KeyError:
        return {"success": False, "error": f"未知工具: {tool_name}"}
    except TypeError as e:
        return {"success": False, "error": f"参数错误: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"执行错误: {str(e)}"}
```

### 重试机制

```python
def chat_with_retry(messages, tools, max_retries=3):
    """带重试的对话"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools
            )
            return response
        except openai.RateLimitError:
            time.sleep(2 ** attempt)  # 指数退避
        except openai.APIError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    
    raise Exception("达到最大重试次数")
```

掌握主流LLM的函数调用协议，是构建可靠智能体应用的基础。在实际开发中，建议使用统一的抽象层来隔离不同提供商的差异，提高代码的可维护性和可移植性。
