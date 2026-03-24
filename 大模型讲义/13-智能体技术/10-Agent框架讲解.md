# Agent框架讲解

市面上涌现了众多Agent开发框架，各有侧重和特色。本节将介绍三个主流框架——LangChain、LlamaIndex和Dify，帮助读者了解其设计理念和适用场景。

## LangChain

LangChain是目前最流行的LLM应用开发框架，提供了丰富的组件和抽象。

### 核心抽象

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 工具定义
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="搜索互联网获取信息"
    ),
    Tool(
        name="Calculator",
        func=calculator_function,
        description="执行数学计算"
    )
]

# 从Hub获取Prompt模板
prompt = hub.pull("hwchase17/react")

# 创建Agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行
result = executor.invoke({"input": "北京到上海的距离是多少公里？"})
```

### LCEL（LangChain Expression Language）

LangChain的管道式组合语法：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 定义组件
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手。"),
    ("human", "{input}")
])

output_parser = StrOutputParser()

# 使用 | 组合成链
chain = prompt | llm | output_parser

# 执行
response = chain.invoke({"input": "你好"})

# 更复杂的链
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
```

### 记忆管理

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# 简单缓冲记忆
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 摘要记忆（适合长对话）
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# 在Agent中使用
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=buffer_memory,
    verbose=True
)
```

### 适用场景

- 需要灵活组合各种组件
- 快速原型开发
- 已有丰富的生态集成需求

## LlamaIndex

LlamaIndex专注于数据索引和检索，是构建RAG应用的首选框架。

### 核心概念

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(
    llm=OpenAI(model="gpt-4"),
    similarity_top_k=5
)

# 查询
response = query_engine.query("文档中提到了哪些技术？")
```

### 高级索引

```python
from llama_index.core import (
    TreeIndex,
    KeywordTableIndex,
    KnowledgeGraphIndex
)

# 树形索引（适合层次结构文档）
tree_index = TreeIndex.from_documents(documents)

# 关键词表索引（适合精确匹配）
keyword_index = KeywordTableIndex.from_documents(documents)

# 知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2
)
```

### 数据连接器

```python
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.database import DatabaseReader

# 网页读取器
web_reader = SimpleWebPageReader()
web_docs = web_reader.load_data(["https://example.com/page1"])

# 数据库读取器
db_reader = DatabaseReader(
    connection_string="postgresql://user:pass@host/db"
)
db_docs = db_reader.load_data(query="SELECT * FROM articles")
```

### Agent功能

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 将索引包装为工具
tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="document_search",
        description="搜索文档库获取相关信息"
    )
)

# 创建Agent
agent = ReActAgent.from_tools(
    tools=[tool],
    llm=llm,
    verbose=True
)

# 使用Agent
response = agent.chat("文档中关于机器学习的内容有哪些？")
```

### 适用场景

- 知识库问答系统
- 文档检索和分析
- 需要复杂索引结构的应用

## Dify

Dify是一个低代码/无代码的AI应用开发平台，提供可视化的Workflow编排能力。

### 核心功能

| 功能 | 描述 |
|------|------|
| Prompt IDE | 可视化的Prompt编辑和测试 |
| RAG Pipeline | 内置的知识库管理 |
| Workflow | 可视化流程编排 |
| Agent | 内置ReACT和Function Call |
| API接口 | 一键发布为API服务 |

### Workflow示例

```yaml
# Dify Workflow配置示例
workflow:
  name: "智能客服"
  description: "处理用户咨询"
  
  nodes:
    - id: start
      type: start
      outputs:
        - name: user_input
          type: string
          
    - id: classify_intent
      type: llm
      inputs:
        query: "{{start.user_input}}"
      prompt: |
        分析用户意图：
        {{query}}
        
        可能的意图：咨询、投诉、建议
        只输出意图类型。
      outputs:
        - name: intent
          type: string
          
    - id: branch
      type: condition
      conditions:
        - expression: "{{classify_intent.intent}} == '咨询'"
          next: query_kb
        - expression: "{{classify_intent.intent}} == '投诉'"
          next: create_ticket
        - default: direct_reply
          
    - id: query_kb
      type: knowledge_retrieval
      dataset_id: "customer_service_kb"
      query: "{{start.user_input}}"
      top_k: 3
      
    - id: generate_reply
      type: llm
      inputs:
        question: "{{start.user_input}}"
        context: "{{query_kb.results}}"
      prompt: |
        基于以下资料回答用户问题：
        
        资料：
        {{context}}
        
        问题：{{question}}
```

### API集成

```python
import requests

# Dify应用API调用
def call_dify_app(app_id: str, api_key: str, query: str):
    response = requests.post(
        f"https://api.dify.ai/v1/chat-messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "user": "user-123"
        }
    )
    return response.json()
```

### 适用场景

- 快速构建AI应用原型
- 非技术人员参与开发
- 需要可视化管理的场景

## 框架对比

| 特性 | LangChain | LlamaIndex | Dify |
|------|-----------|------------|------|
| 定位 | 通用LLM应用框架 | 数据索引与检索 | 低代码AI平台 |
| 学习曲线 | 中等 | 中等 | 低 |
| 灵活性 | 高 | 中 | 低 |
| RAG能力 | 中 | 高 | 中 |
| 可视化 | 无 | 无 | 强 |
| 生态丰富度 | 高 | 中 | 中 |
| 部署方式 | 自主部署 | 自主部署 | SaaS/自部署 |

## 选型建议

**选择LangChain**：
- 需要高度定制化
- 已有Python开发能力
- 需要与多种数据源和服务集成

**选择LlamaIndex**：
- 核心需求是文档检索和问答
- 需要处理大量非结构化数据
- 对检索精度有较高要求

**选择Dify**：
- 需要快速上线
- 团队缺乏深度技术能力
- 需要可视化管理和监控

**混合使用**：
很多项目会结合使用多个框架：
- 用LlamaIndex处理数据索引
- 用LangChain编排Agent逻辑
- 用Dify管理应用和监控

```python
# 混合使用示例
from langchain.tools import Tool
from llama_index.core import VectorStoreIndex

# 用LlamaIndex构建索引
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 包装为LangChain工具
search_tool = Tool(
    name="KnowledgeSearch",
    func=lambda q: str(query_engine.query(q)),
    description="搜索知识库"
)

# 在LangChain Agent中使用
agent = create_react_agent(llm, [search_tool], prompt)
```

选择合适的框架能显著提升开发效率。建议根据项目需求、团队能力和长期维护成本综合考虑。
