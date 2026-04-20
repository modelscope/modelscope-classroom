# Agent框架讲解

假设你决定开一家餐厅。你可以从零开始装修、采购设备、招募人员，也可以加盟一个现成的品牌——后者提供标准化的装修方案、供应链和管理流程，你只需专注于菜品和服务。Agent 开发框架正是这样的"加盟方案"——它们提供了现成的组件和抽象，让开发者不必从零搭建每个细节。当前市面上涌现了众多框架，各有侧重。本节将介绍三个主流框架——LangChain、LlamaIndex 和 Dify，帮助读者了解其设计理念和适用场景。

## LangChain

LangChain 是目前最流行的 LLM 应用开发框架，提供了丰富的组件和抽象。如果把它比作一套万能积木，那么它的特点就是"拼接自由度极高"——你可以把 LLM、工具、记忆、检索器等组件像拼积木一样自由组合。

### 核心抽象

下面这段代码展示了 LangChain 的基本用法——定义工具、加载 Prompt 模板、创建 Agent、执行查询，整个流程就像组装一条流水线：

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

LangChain 最巧妙的设计之一是 LCEL——用 `|` 符号把组件串联起来，就像 Linux 的管道操作一样直观。想象一下工厂流水线：原料（Prompt）进入→ 加工（LLM）→ 包装（OutputParser）→ 成品：

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

前面的记忆检索与上下文一节中我们讨论过记忆的重要性。LangChain 提供了开箱即用的记忆组件，就像给 Agent 配了一个笔记本——简单场景用“全部记录”，长对话用“摘要记录”：

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

LlamaIndex 专注于数据索引和检索，是构建 RAG 应用的首选框架。如果说 LangChain 是万能积木，那么 LlamaIndex 就是专业的图书管理系统——它的核心能力是把各种数据源组织成可检索的索引，让模型能快速找到所需信息。

### 核心概念

下面的示例展示了 LlamaIndex 最典型的用法：加载文档、构建索引、查询。只需三步就能搭建一个基本的知识库问答系统：

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

LlamaIndex 的强大之处在于提供了多种索引类型，就像图书馆不只有一种检索方式——有按分类号查的、有按主题词查的、还有按关联关系查的：

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

在实际开发中，数据往往分散在不同地方——网页、数据库、PDF 文件等。LlamaIndex 提供了丰富的连接器，就像一套万能转接头，让各种数据源都能接入统一的索引体系：

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

Dify 是一个低代码/无代码的 AI 应用开发平台。如果说 LangChain 是给程序员用的积木，那 Dify 就是给产品经理和业务人员用的"拖拽式设计师"——不需要写代码，通过可视化界面就能搭建完整的 AI 应用。

### 核心功能

| 功能 | 描述 |
|------|------|
| Prompt IDE | 可视化的Prompt编辑和测试 |
| RAG Pipeline | 内置的知识库管理 |
| Workflow | 可视化流程编排 |
| Agent | 内置ReACT和Function Call |
| API接口 | 一键发布为API服务 |

### Workflow示例

下面是一个智能客服的 Workflow 配置示例。注意它的逻辑多么清晰：先分类用户意图，然后根据意图走不同分支——咨询就查知识库，投诉就创建工单。这种可视化编排让非技术人员也能看懂和参与：

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

说了这么多，到底怎么选？假设你要开三家不同类型的餐厅：LangChain 像一家可以自由定制菜单的创意餐厅，LlamaIndex 像一家专注食材采购和储存的供应链服务商，而 Dify 像一家提供标准化方案的连锁品牌。下表细化了它们的差异：

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

在实际项目中，框架选择往往不是"非此即彼"而是"因地制宜"。以下是一些实用的判断原则：

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
很多成熟项目会结合使用多个框架——就像一家好餐厅可能同时使用专业供应商的食材、自家的独家配方、和第三方的配送平台：
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

回顾本节，框架选择的核心原则是“匹配需求，而非追逐流行”。小团队快速验证想法可以先用 Dify；需要精细控制 RAG 流程时上 LlamaIndex；复杂的多工具、多步骤 Agent 场景则选 LangChain。而在生产环境中，混合使用往往是最实际的方案。
