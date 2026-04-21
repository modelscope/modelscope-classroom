# 实践：使用SWIFT进行模型训练并完成Gradio应用

本节通过一个完整的实践项目，演示从模型微调到应用部署的全流程。

假设你是一个团队的开发者，老板说："我们需要一个能回答编程问题的AI助手，下周给客户演示。"你该怎么做？最快的路径就是：用SWIFT微调一个基座模型，再用Gradio包一层美观的交互界面。本节就手把手带你走完这个流程。

## 项目目标

下图展示了使用SWIFT训练并部署应用的完整流程：

```mermaid
graph LR
    A[数据准备] --> B[配置训练参数]
    B --> C[SWIFT LoRA训练]
    C --> D[模型测试]
    D -->|不满意| A
    D -->|满意| E[Gradio应用开发]
    E --> F[Docker打包部署]
```

构建一个专业的代码助手，能够：
- 回答编程相关问题
- 解释代码逻辑
- 生成代码片段

## 环境准备

### 安装依赖

```bash
# 安装SWIFT
pip install ms-swift

# 安装Gradio
pip install gradio

# 验证安装
swift sft --help
```

### 硬件要求

| 模型规模 | 最低显存 | 推荐显存 |
|---------|---------|---------|
| 1.5B | 8GB | 16GB |
| 7B (LoRA) | 16GB | 24GB |
| 7B (Full) | 60GB | 80GB |

## 数据准备

"巧妇难为无米之炊"——数据是微调的基础。你需要准备一批与目标任务相关的问答对。对于代码助手，就是"用户问编程问题 + 高质量回答"的配对。数据质量直接决定微调效果，与其数量多但质量差，不如少而精。

### 数据格式

准备代码问答数据，采用SWIFT支持的格式：

```json
// data/code_qa.json
[
    {
        "query": "如何在Python中读取JSON文件？",
        "response": "在Python中读取JSON文件可以使用内置的json模块：\n\n```python\nimport json\n\nwith open('data.json', 'r', encoding='utf-8') as f:\n    data = json.load(f)\n```\n\n这里使用`with`语句确保文件正确关闭，`encoding='utf-8'`处理中文字符。"
    },
    {
        "query": "解释一下Python的装饰器",
        "response": "装饰器是Python的一种高级特性，用于在不修改原函数代码的情况下扩展其功能。\n\n基本语法：\n```python\ndef my_decorator(func):\n    def wrapper(*args, **kwargs):\n        print('函数调用前')\n        result = func(*args, **kwargs)\n        print('函数调用后')\n        return result\n    return wrapper\n\n@my_decorator\ndef say_hello():\n    print('Hello!')\n```\n\n`@my_decorator`等价于`say_hello = my_decorator(say_hello)`。"
    }
]
```

### 数据收集建议

```python
# 从现有代码库提取问答对
import ast
import json

def extract_qa_from_code(code_file):
    """从带docstring的代码中提取问答对"""
    with open(code_file, 'r') as f:
        tree = ast.parse(f.read())
    
    qa_pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if ast.get_docstring(node):
                qa_pairs.append({
                    'query': f"解释函数 {node.name} 的功能",
                    'response': ast.get_docstring(node)
                })
    
    return qa_pairs
```

## 模型微调

现在进入核心环节。SWIFT支持多种训练方式，对于我们这个场景，LoRA微调是最实用的选择——它只更新很小一部分参数，训练速度快、显存占用少，但效果往往已经足够好。

### 配置文件

```yaml
# config/sft_config.yaml
model_type: qwen2-1_5b-instruct
dataset: ['/path/to/data/code_qa.json']

# 训练参数
train_type: lora
lora_rank: 8
lora_alpha: 32
lora_dropout: 0.05

# 训练配置
batch_size: 4
gradient_accumulation_steps: 4
num_train_epochs: 3
learning_rate: 1e-4
warmup_ratio: 0.1

# 输出
output_dir: ./output/code_assistant
logging_steps: 10
save_strategy: epoch
```

### 启动训练

```bash
# 命令行训练
swift sft \
    --model_type qwen2-1_5b-instruct \
    --dataset /path/to/data/code_qa.json \
    --train_type lora \
    --lora_rank 8 \
    --output_dir ./output/code_assistant \
    --num_train_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4
```

### Python API训练

```python
from swift.llm import sft_main, SftArguments

args = SftArguments(
    model_type='qwen2-1_5b-instruct',
    dataset=['/path/to/data/code_qa.json'],
    train_type='lora',
    lora_rank=8,
    output_dir='./output/code_assistant',
    num_train_epochs=3,
    batch_size=4,
    learning_rate=1e-4,
)

output = sft_main(args)
print(f"模型保存到: {output['best_model_checkpoint']}")
```

### 训练监控

```python
# 使用TensorBoard监控
# tensorboard --logdir ./output/code_assistant/runs

# 或在训练时打印关键指标
"""
训练日志示例：
Step 100/1000 | Loss: 1.234 | LR: 9.5e-5 | Time: 2.3s/step
Step 200/1000 | Loss: 0.856 | LR: 9.0e-5 | Time: 2.2s/step
...
"""
```

## 模型测试

训练完成后，别急着做应用——先测试一下模型的实际效果。这就像厨师烧好菜后自己先尝一口，确认味道没问题再端给客人。

### 命令行测试

```bash
swift infer \
    --model_type qwen2-1_5b-instruct \
    --adapters ./output/code_assistant/checkpoint-xxx
```

### Python测试

```python
from swift.llm import InferArguments, infer_main

args = InferArguments(
    model_type='qwen2-1_5b-instruct',
    adapters='./output/code_assistant/checkpoint-xxx',
)

# 交互式测试
infer_main(args)
```

### 批量评估

```python
from swift.llm import get_model_tokenizer

model, tokenizer = get_model_tokenizer(
    model_type='qwen2-1_5b-instruct',
    adapters='./output/code_assistant/checkpoint-xxx'
)

test_cases = [
    "如何用Python实现快速排序？",
    "解释Python中的列表推导式",
    "什么是闭包？"
]

for query in test_cases:
    response = model.generate(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True
        )
    )
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

## Gradio应用开发

模型测试通过后，就可以把它包装成一个漂亮的交互应用了。想象你要把一份精心烹制的菜装盘上桌——味道很重要，但卖相也很重要。Gradio就是帮你做"装盘"的工具。

### 基础聊天界面

```python
import gradio as gr
from swift.llm import get_model_tokenizer

# 加载模型
model, tokenizer = get_model_tokenizer(
    model_type='qwen2-1_5b-instruct',
    adapters='./output/code_assistant/checkpoint-xxx'
)

def chat(message, history):
    # 构建对话历史
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    # 生成回复
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    response = model.generate(prompt, max_new_tokens=512)
    return response

# 创建界面
demo = gr.ChatInterface(
    fn=chat,
    title="代码助手",
    description="基于微调模型的编程问答助手",
    examples=[
        "如何在Python中实现单例模式？",
        "解释一下Git的工作原理",
        "写一个二分查找的Python实现"
    ]
)

demo.launch(share=True)
```

### 增强版界面

```python
import gradio as gr

def create_app(model, tokenizer):
    def chat(message, history, system_prompt, temperature, max_tokens):
        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": message})
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
        
        return response
    
    with gr.Blocks(title="代码助手 Pro") as demo:
        gr.Markdown("# 代码助手 Pro")
        gr.Markdown("基于SWIFT微调的专业编程助手")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入你的编程问题...",
                    lines=3
                )
                with gr.Row():
                    submit = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空对话")
            
            with gr.Column(scale=1):
                gr.Markdown("### 设置")
                system_prompt = gr.Textbox(
                    label="系统提示",
                    value="你是一个专业的编程助手，擅长回答各种编程问题，解释代码逻辑，并提供高质量的代码示例。",
                    lines=4
                )
                temperature = gr.Slider(
                    0, 1, value=0.7,
                    label="Temperature",
                    info="较低值更确定，较高值更创意"
                )
                max_tokens = gr.Slider(
                    100, 2000, value=512,
                    label="最大生成长度"
                )
        
        def respond(message, history, sys, temp, max_tok):
            response = chat(message, history, sys, temp, max_tok)
            history.append((message, response))
            return "", history
        
        submit.click(
            respond,
            [msg, chatbot, system_prompt, temperature, max_tokens],
            [msg, chatbot]
        )
        msg.submit(
            respond,
            [msg, chatbot, system_prompt, temperature, max_tokens],
            [msg, chatbot]
        )
        clear.click(lambda: [], None, chatbot)
        
        gr.Markdown("### 示例问题")
        gr.Examples(
            [
                "如何用Python实现LRU缓存？",
                "解释一下async/await的工作原理",
                "写一个线程安全的单例模式"
            ],
            msg
        )
    
    return demo

# 启动应用
demo = create_app(model, tokenizer)
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 流式输出

在实际体验中，流式输出是很重要的体验优化。想象你跟一个人聊天，如果对方每次都得想半天然后一口气把一大段话说完，体验肯定不好。流式输出让模型像人说话一样一个字一个字地"蹦"出来，用户体验好得多：

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_chat(message, history, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    response = ""
    for text in streamer:
        response += text
        yield response
```

## 部署上线

应用开发完成后，最后一步是把它部署到服务器上，让其他人也能访问。Docker是最常用的打包方式——把你的代码、模型和所有依赖打包成一个“集装箱”，拿到任何服务器上都能直接跑起来。

### Docker打包

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制模型和代码
COPY model/ ./model/
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
```

### 启动脚本

```bash
#!/bin/bash
# run.sh

export CUDA_VISIBLE_DEVICES=0
python app.py --model_path ./model --port 7860
```

通过这个完整的实践，读者可以掌握从数据准备、模型微调到应用部署的全流程。建议先用一个小模型（如Qwen2-1.5B）和少量数据跑通整个流程，确认每个环节都没问题后，再逐步扮大数据规模和模型规模。这个流程可以根据具体需求进行调整和扩展。
