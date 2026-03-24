# OpenClaw 与 ACP 协议

## 1. OpenClaw 概述

OpenClaw 是一个开源的自托管 AI Agent 框架，采用 MIT 许可证。与传统的 AI 助手不同，OpenClaw 专注于构建能够自主执行任务的智能体系统，支持与 IDE、终端、通信工具等多种客户端的无缝集成。

### 1.1 核心特性

| 特性 | 描述 |
|------|------|
| 自托管 | 完全运行在本地或私有服务器，数据不外泄 |
| 多 Agent 编排 | 支持调度多个专业 Agent 协同工作 |
| 工具生态 | 兼容 MCP 协议，接入海量工具 |
| IDE 集成 | 通过 ACP 协议与 VS Code、Zed 等编辑器直连 |
| Skills 支持 | 动态加载专业技能，增强领域能力 |

### 1.2 架构总览

```
┌────────────────────────────────────────────────────────────┐
│                    OpenClaw 系统架构                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Client Layer                        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │  │
│  │  │ VS Code │  │   Zed   │  │Terminal │  │Telegram │  │  │
│  │  │  (ACP)  │  │  (ACP)  │  │  (CLI)  │  │  (Bot)  │  │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │  │
│  └───────┼────────────┼────────────┼────────────┼───────┘  │
│          │            │            │            │          │
│          └────────────┴─────┬──────┴────────────┘          │
│                             │                              │
│  ┌──────────────────────────▼───────────────────────────┐  │
│  │                  OpenClaw Gateway                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │ Session  │  │ Router   │  │  Agent Runtime   │    │  │
│  │  │ Manager  │  │          │  │                  │    │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                             │                              │
│  ┌──────────────────────────▼───────────────────────────┐  │
│  │                   Extension Layer                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │   MCP    │  │  Skills  │  │  External Agents │    │  │
│  │  │ Servers  │  │          │  │  (Claude, GPT)   │    │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 2. ACP 协议详解

### 2.1 什么是 ACP

ACP（Agent Client Protocol）是 OpenClaw 提出的标准化通信协议，用于连接 IDE 和 AI Agent。其设计理念类似于 LSP（Language Server Protocol）：

| 协议 | 用途 | 类比 |
|------|------|------|
| LSP | 编辑器 ↔ 语言服务器 | 让编辑器获得语言智能（补全、跳转） |
| ACP | 编辑器 ↔ AI Agent | 让编辑器获得 Agent 智能（代码生成、重构） |

**核心价值**：开发者无需在 IDE 和 Agent 对话窗口之间反复切换，所有交互都在编辑器内完成。

### 2.2 ACP vs MCP vs Skills

三者是互补的协议/机制，解决不同层次的问题：

```
┌─────────────────────────────────────────────────────────────┐
│                       协议栈层次                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户层     ┌─────────────────────────────────────────┐     │
│            │              ACP 协议                    │     │
│            │   人 → Agent（自上而下的指令传递）        │     │
│            └─────────────────────────────────────────┘     │
│                              │                              │
│  能力层     ┌─────────────────▼─────────────────────┐       │
│            │            Skills 技术                 │       │
│            │   Agent 内部（专业知识与工作流）        │       │
│            └─────────────────────────────────────────┘     │
│                              │                              │
│  工具层     ┌─────────────────▼─────────────────────┐       │
│            │             MCP 协议                   │       │
│            │   Agent → 工具（自内而外的能力扩展）    │       │
│            └─────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| 维度 | ACP | MCP | Skills |
|------|-----|-----|--------|
| 解决问题 | IDE 如何与 Agent 对话 | Agent 如何调用外部工具 | Agent 如何获取专业能力 |
| 通信方向 | 人 → Agent | Agent → 工具 | Agent 内部 |
| 协议格式 | JSON-RPC over stdio | JSON-RPC over stdio/SSE | Markdown 指令 |
| 典型实现 | VS Code 扩展 | 文件系统、数据库工具 | 代码审查技能 |

### 2.3 ACP 通信架构

```
┌─────────────────────────────────────────────────────────────┐
│                     ACP 通信链路                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     stdio      ┌──────────────────┐      │
│  │   VS Code    │ ←────────────→ │  openclaw acp    │      │
│  │              │    JSON-RPC    │  (Bridge CLI)    │      │
│  │ ┌──────────┐ │                │                  │      │
│  │ │ Agent    │ │                │ 功能：           │      │
│  │ │ Panel    │ │                │ • stdio↔WS 转换  │      │
│  │ └──────────┘ │                │ • 认证管理       │      │
│  └──────────────┘                │ • 消息路由       │      │
│                                  └────────┬─────────┘      │
│                                           │                 │
│                                           │ WebSocket       │
│                                           │                 │
│                                           ▼                 │
│                                  ┌──────────────────┐      │
│                                  │ OpenClaw Gateway │      │
│                                  │                  │      │
│                                  │ ┌──────────────┐ │      │
│                                  │ │Agent Runtime │ │      │
│                                  │ └──────────────┘ │      │
│                                  └──────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键组件**：

1. **IDE (ACP Client)**：发送 prompt、显示响应、提供编辑器上下文
2. **ACP Bridge (CLI)**：协议翻译器，将 stdio 消息转为 WebSocket
3. **OpenClaw Gateway (ACP Server)**：接收请求、调用 Agent、返回结果

### 2.4 ACP 消息格式

ACP 使用 JSON-RPC 2.0 协议：

```json
// 请求：发送 prompt
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "agent/prompt",
    "params": {
        "prompt": "重构这个函数，提取公共逻辑",
        "context": {
            "file": "/src/utils.py",
            "selection": {
                "start": {"line": 10, "character": 0},
                "end": {"line": 25, "character": 0}
            },
            "content": "def process_data(data):\n    ..."
        }
    }
}

// 响应：流式返回
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "type": "stream",
        "content": "我来帮你重构这个函数...",
        "actions": [
            {
                "type": "edit",
                "file": "/src/utils.py",
                "changes": [...]
            }
        ]
    }
}
```

## 3. ACP 运行模式

### 3.1 Bridge 模式

标准模式，ACP CLI 作为 IDE 和 Gateway 之间的桥梁：

```python
# 配置示例：VS Code settings.json
{
    "openclaw.acp.mode": "bridge",
    "openclaw.acp.gatewayUrl": "ws://localhost:8080",
    "openclaw.acp.apiKey": "your-api-key"
}
```

**工作流程**：

```
1. VS Code 启动 ACP Bridge 子进程
2. 用户在编辑器中选择代码，输入指令
3. VS Code 通过 stdio 发送 JSON-RPC 请求给 Bridge
4. Bridge 转换为 WebSocket 消息，发送给 Gateway
5. Gateway 调用 Agent 处理请求
6. 响应流式返回，Bridge 转发给 VS Code
7. VS Code 显示响应，可自动应用代码修改
```

### 3.2 Client 模式

ACP CLI 直接作为 Agent 客户端，适用于终端使用：

```bash
# 启动交互式会话
openclaw acp --mode client --gateway ws://localhost:8080

# 单次调用
openclaw acp prompt "生成一个 FastAPI 的用户认证模块"
```

## 4. 实现 ACP Client

### 4.1 基础客户端

```python
import subprocess
import json
import threading
from typing import Callable, Optional
from dataclasses import dataclass

@dataclass
class EditorContext:
    """编辑器上下文"""
    file: str
    selection_start: tuple  # (line, character)
    selection_end: tuple
    content: str
    workspace: Optional[str] = None

class ACPClient:
    """ACP 客户端实现"""
    
    def __init__(self, bridge_command: list):
        self.process = subprocess.Popen(
            bridge_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.request_id = 0
        self.callbacks = {}
        self._start_reader()
        
    def _start_reader(self):
        """启动响应读取线程"""
        def reader():
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                try:
                    response = json.loads(line)
                    self._handle_response(response)
                except json.JSONDecodeError:
                    pass
                    
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        
    def _handle_response(self, response: dict):
        """处理响应"""
        req_id = response.get("id")
        if req_id in self.callbacks:
            callback = self.callbacks.pop(req_id)
            callback(response.get("result"), response.get("error"))
            
    def send_prompt(
        self,
        prompt: str,
        context: Optional[EditorContext] = None,
        callback: Optional[Callable] = None
    ):
        """发送 prompt 到 Agent"""
        self.request_id += 1
        
        params = {"prompt": prompt}
        if context:
            params["context"] = {
                "file": context.file,
                "selection": {
                    "start": {"line": context.selection_start[0], 
                             "character": context.selection_start[1]},
                    "end": {"line": context.selection_end[0], 
                           "character": context.selection_end[1]}
                },
                "content": context.content,
                "workspace": context.workspace
            }
            
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "agent/prompt",
            "params": params
        }
        
        if callback:
            self.callbacks[self.request_id] = callback
            
        self._send(request)
        return self.request_id
        
    def _send(self, request: dict):
        """发送请求"""
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        
    def list_agents(self) -> list:
        """列出可用 Agent"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "agent/list",
            "params": {}
        }
        self._send(request)
        # 同步等待响应（简化实现）
        line = self.process.stdout.readline()
        response = json.loads(line)
        return response.get("result", {}).get("agents", [])
        
    def close(self):
        """关闭客户端"""
        self.process.terminate()
```

### 4.2 VS Code 扩展集成

```typescript
// extension.ts
import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';

class ACPExtension {
    private bridge: ChildProcess | null = null;
    private requestId = 0;
    private pendingRequests = new Map<number, (result: any) => void>();

    activate(context: vscode.ExtensionContext) {
        // Register command
        const command = vscode.commands.registerCommand(
            'openclaw.sendPrompt',
            () => this.sendPrompt()
        );
        context.subscriptions.push(command);

        // Start bridge
        this.startBridge();
    }

    private startBridge() {
        const config = vscode.workspace.getConfiguration('openclaw');
        const gatewayUrl = config.get<string>('gatewayUrl', 'ws://localhost:8080');

        this.bridge = spawn('openclaw', ['acp', '--gateway', gatewayUrl], {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        this.bridge.stdout?.on('data', (data) => {
            const lines = data.toString().split('\n');
            for (const line of lines) {
                if (line.trim()) {
                    this.handleResponse(JSON.parse(line));
                }
            }
        });
    }

    private async sendPrompt() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        // Get user input
        const prompt = await vscode.window.showInputBox({
            prompt: 'Enter your instruction for the AI agent'
        });
        if (!prompt) return;

        // Build context
        const selection = editor.selection;
        const context = {
            file: editor.document.uri.fsPath,
            selection: {
                start: { line: selection.start.line, character: selection.start.character },
                end: { line: selection.end.line, character: selection.end.character }
            },
            content: editor.document.getText(selection)
        };

        // Send request
        this.requestId++;
        const request = {
            jsonrpc: '2.0',
            id: this.requestId,
            method: 'agent/prompt',
            params: { prompt, context }
        };

        this.bridge?.stdin?.write(JSON.stringify(request) + '\n');

        // Show progress
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'OpenClaw is thinking...'
        }, () => {
            return new Promise<void>((resolve) => {
                this.pendingRequests.set(this.requestId, (result) => {
                    this.applyResult(result);
                    resolve();
                });
            });
        });
    }

    private handleResponse(response: any) {
        const callback = this.pendingRequests.get(response.id);
        if (callback) {
            this.pendingRequests.delete(response.id);
            callback(response.result);
        }
    }

    private applyResult(result: any) {
        // Apply code edits
        if (result.actions) {
            for (const action of result.actions) {
                if (action.type === 'edit') {
                    this.applyEdit(action);
                }
            }
        }

        // Show response in panel
        if (result.content) {
            vscode.window.showInformationMessage(result.content);
        }
    }

    private async applyEdit(action: any) {
        const uri = vscode.Uri.file(action.file);
        const document = await vscode.workspace.openTextDocument(uri);
        const edit = new vscode.WorkspaceEdit();

        for (const change of action.changes) {
            const range = new vscode.Range(
                change.range.start.line,
                change.range.start.character,
                change.range.end.line,
                change.range.end.character
            );
            edit.replace(uri, range, change.newText);
        }

        await vscode.workspace.applyEdit(edit);
    }
}
```

## 5. OpenClaw Gateway 实现

### 5.1 Gateway 核心

```python
import asyncio
import json
from typing import Dict, Optional
from dataclasses import dataclass, field
import websockets

@dataclass
class Session:
    """会话管理"""
    session_id: str
    agent_id: str
    workspace: Optional[str] = None
    history: list = field(default_factory=list)

class OpenClawGateway:
    """OpenClaw Gateway 服务端"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.sessions: Dict[str, Session] = {}
        self.agents: Dict[str, 'Agent'] = {}
        
    async def start(self):
        """启动 Gateway 服务"""
        async with websockets.serve(self.handle_connection, self.host, self.port):
            print(f"OpenClaw Gateway running at ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
            
    async def handle_connection(self, websocket, path):
        """处理 WebSocket 连接"""
        session_id = None
        
        try:
            async for message in websocket:
                request = json.loads(message)
                response = await self.handle_request(request, websocket)
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if session_id and session_id in self.sessions:
                del self.sessions[session_id]
                
    async def handle_request(self, request: dict, websocket) -> dict:
        """处理 JSON-RPC 请求"""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        
        try:
            if method == "agent/prompt":
                result = await self.handle_prompt(params, websocket)
            elif method == "agent/list":
                result = self.handle_list_agents()
            elif method == "session/create":
                result = self.handle_create_session(params)
            else:
                return self._error_response(req_id, -32601, f"Method not found: {method}")
                
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
            
        except Exception as e:
            return self._error_response(req_id, -32000, str(e))
            
    async def handle_prompt(self, params: dict, websocket) -> dict:
        """处理 prompt 请求"""
        prompt = params.get("prompt")
        context = params.get("context")
        session_id = params.get("session_id")
        
        # Get or create session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = Session(
                session_id=self._generate_session_id(),
                agent_id="default"
            )
            self.sessions[session.session_id] = session
            
        # Get agent
        agent = self.agents.get(session.agent_id)
        if not agent:
            agent = self._create_default_agent()
            self.agents[session.agent_id] = agent
            
        # Execute agent
        result = await agent.execute(prompt, context, session.history)
        
        # Update history
        session.history.append({"role": "user", "content": prompt})
        session.history.append({"role": "assistant", "content": result["content"]})
        
        return result
        
    def handle_list_agents(self) -> dict:
        """列出可用 Agent"""
        return {
            "agents": [
                {"id": "default", "name": "Default Agent", "description": "通用代码助手"},
                {"id": "code-review", "name": "Code Review Agent", "description": "专业代码审查"},
                {"id": "refactor", "name": "Refactor Agent", "description": "代码重构专家"}
            ]
        }
        
    def _generate_session_id(self) -> str:
        import uuid
        return str(uuid.uuid4())
        
    def _create_default_agent(self) -> 'Agent':
        from .agent import CodeAgent
        return CodeAgent()
        
    def _error_response(self, req_id, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message}
        }
```

### 5.2 Agent 执行器

```python
class Agent:
    """Agent 基类"""
    
    def __init__(self, llm_client, tools=None, skills=None):
        self.llm = llm_client
        self.tools = tools or []
        self.skills = skills or []
        
    async def execute(self, prompt: str, context: dict, history: list) -> dict:
        """执行 Agent 任务"""
        raise NotImplementedError


class CodeAgent(Agent):
    """代码 Agent 实现"""
    
    SYSTEM_PROMPT = """你是一个专业的编程助手，运行在 OpenClaw 系统中。
    
你的能力：
1. 理解用户的代码修改需求
2. 分析提供的代码上下文
3. 生成高质量的代码修改建议
4. 解释代码逻辑和最佳实践

输出格式：
- 先简要说明你的理解和方案
- 如果需要修改代码，以 JSON 格式提供 actions

修改代码时使用以下格式：
```json
{
    "actions": [
        {
            "type": "edit",
            "file": "文件路径",
            "changes": [
                {
                    "range": {
                        "start": {"line": 起始行, "character": 起始列},
                        "end": {"line": 结束行, "character": 结束列}
                    },
                    "newText": "新的代码内容"
                }
            ]
        }
    ]
}
```"""

    async def execute(self, prompt: str, context: dict, history: list) -> dict:
        """执行代码任务"""
        
        # Build messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(history[-10:])  # Keep last 10 turns
        
        # Add context
        user_message = prompt
        if context:
            user_message = f"""用户请求：{prompt}

上下文信息：
- 文件：{context.get('file', 'unknown')}
- 选中代码：
```
{context.get('content', '')}
```"""
        
        messages.append({"role": "user", "content": user_message})
        
        # Call LLM
        response = await self.llm.chat_async(messages=messages)
        content = response["content"]
        
        # Parse actions
        actions = self._parse_actions(content)
        
        return {
            "content": content,
            "actions": actions
        }
        
    def _parse_actions(self, content: str) -> list:
        """从响应中解析 actions"""
        import re
        
        # Find JSON block
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            try:
                data = json.loads(match.group(1))
                return data.get("actions", [])
            except json.JSONDecodeError:
                pass
                
        return []
```

## 6. 多 Agent 协作

### 6.1 Agent 路由

```python
class AgentRouter:
    """Agent 路由器：根据任务类型选择合适的 Agent"""
    
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.classifier = TaskClassifier()
        
    async def route(self, prompt: str, context: dict) -> Agent:
        """路由到合适的 Agent"""
        
        # Classify task
        task_type = await self.classifier.classify(prompt, context)
        
        # Route mapping
        route_map = {
            "code_generation": "coder",
            "code_review": "reviewer",
            "refactoring": "refactor",
            "debugging": "debugger",
            "documentation": "doc_writer",
            "general": "default"
        }
        
        agent_id = route_map.get(task_type, "default")
        return self.agents.get(agent_id, self.agents["default"])


class TaskClassifier:
    """任务分类器"""
    
    PATTERNS = {
        "code_generation": ["生成", "创建", "编写", "实现", "generate", "create", "write"],
        "code_review": ["审查", "检查", "review", "check", "分析"],
        "refactoring": ["重构", "优化", "refactor", "optimize", "改进"],
        "debugging": ["调试", "修复", "debug", "fix", "错误", "bug"],
        "documentation": ["文档", "注释", "说明", "document", "comment"]
    }
    
    async def classify(self, prompt: str, context: dict) -> str:
        """分类任务类型"""
        prompt_lower = prompt.lower()
        
        for task_type, patterns in self.PATTERNS.items():
            if any(p in prompt_lower for p in patterns):
                return task_type
                
        return "general"
```

### 6.2 Agent 协作流水线

```python
class AgentPipeline:
    """Agent 协作流水线"""
    
    def __init__(self, stages: list):
        """
        stages: List of (agent, transform_fn) tuples
        transform_fn transforms previous result to next input
        """
        self.stages = stages
        
    async def execute(self, initial_prompt: str, context: dict) -> dict:
        """执行流水线"""
        current_input = {"prompt": initial_prompt, "context": context}
        results = []
        
        for agent, transform_fn in self.stages:
            # Execute stage
            result = await agent.execute(
                current_input["prompt"],
                current_input.get("context"),
                []
            )
            results.append(result)
            
            # Transform for next stage
            if transform_fn:
                current_input = transform_fn(result, context)
                
        # Aggregate results
        return self._aggregate_results(results)
        
    def _aggregate_results(self, results: list) -> dict:
        """聚合流水线结果"""
        all_actions = []
        all_content = []
        
        for result in results:
            if result.get("content"):
                all_content.append(result["content"])
            if result.get("actions"):
                all_actions.extend(result["actions"])
                
        return {
            "content": "\n\n---\n\n".join(all_content),
            "actions": all_actions
        }


# 使用示例：代码生成 → 审查 → 测试生成 流水线
async def create_code_pipeline(agents):
    pipeline = AgentPipeline([
        (agents["coder"], None),
        (agents["reviewer"], lambda r, c: {
            "prompt": f"审查以下代码：\n{r['content']}",
            "context": c
        }),
        (agents["tester"], lambda r, c: {
            "prompt": f"为以下代码生成测试：\n{c.get('content', '')}",
            "context": c
        })
    ])
    return pipeline
```

## 7. 配置与部署

### 7.1 Gateway 配置

```yaml
# gateway.yaml
server:
  host: "0.0.0.0"
  port: 8080
  ssl:
    enabled: false
    cert_path: "/path/to/cert.pem"
    key_path: "/path/to/key.pem"

auth:
  enabled: true
  api_keys:
    - "key-1-xxx"
    - "key-2-xxx"

agents:
  default:
    type: "code"
    llm:
      provider: "openai"
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
    tools:
      - "filesystem"
      - "terminal"
    skills:
      - "code-review"
      - "refactoring"

  reviewer:
    type: "code"
    llm:
      provider: "anthropic"
      model: "claude-3-opus"
      api_key: "${ANTHROPIC_API_KEY}"
    skills:
      - "code-review"

mcp:
  servers:
    - name: "filesystem"
      command: ["npx", "-y", "@anthropic-ai/mcp-server-filesystem"]
      args: ["/workspace"]
    - name: "github"
      command: ["npx", "-y", "@anthropic-ai/mcp-server-github"]

limits:
  max_sessions: 100
  session_timeout: 3600
  max_tokens_per_request: 8000
```

### 7.2 VS Code 配置

```json
// .vscode/settings.json
{
    "openclaw.enabled": true,
    "openclaw.acp.gatewayUrl": "ws://localhost:8080",
    "openclaw.acp.apiKey": "your-api-key",
    "openclaw.autoApplyEdits": false,
    "openclaw.showInlineHints": true,
    "openclaw.defaultAgent": "default"
}
```

## 8. 最佳实践

### 8.1 安全注意事项

| 风险 | 防护措施 |
|------|----------|
| API Key 泄露 | 使用环境变量，不硬编码 |
| 代码注入 | 沙箱执行 Agent 生成的代码 |
| 会话劫持 | 使用 TLS 加密，会话 Token 验证 |
| 资源耗尽 | 设置请求限流和 Token 限制 |

### 8.2 性能优化

1. **连接复用**：保持 WebSocket 长连接，避免频繁握手
2. **流式响应**：使用 SSE 或 WebSocket 流式传输，提升用户体验
3. **缓存热点**：缓存常用 Skill 和工具配置
4. **异步执行**：工具调用使用异步 IO

### 8.3 调试技巧

```bash
# 启用详细日志
openclaw acp --gateway ws://localhost:8080 --verbose

# 检查连接状态
openclaw acp status

# 测试工具可用性
openclaw acp test-tool filesystem.read_file --args '{"path": "/tmp/test.txt"}'
```

ACP 协议和 OpenClaw 框架代表了 AI Agent 与开发工具集成的未来方向。通过标准化的协议，开发者可以在熟悉的 IDE 环境中无缝使用 AI 能力，大幅提升开发效率。
