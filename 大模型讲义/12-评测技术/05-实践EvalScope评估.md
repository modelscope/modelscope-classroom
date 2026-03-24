# 实践：使用EvalScope进行模型评估

本节将通过完整的实践案例，演示如何使用EvalScope评测框架对大模型进行系统评估，包括标准基准评测、自定义数据集评测以及评测结果分析。

## 环境准备

### 安装EvalScope

```bash
# 基础安装
pip install evalscope

# 安装完整依赖（包含多模态支持）
pip install evalscope[all]

# 验证安装
python -c "import evalscope; print(evalscope.__version__)"
```

### 配置环境

```python
import os

# 设置ModelScope缓存目录（可选）
os.environ['MODELSCOPE_CACHE'] = '/data/modelscope_cache'

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 关闭wandb（可选）
os.environ['WANDB_DISABLED'] = 'true'
```

## 基础评测实践

### 评测单个模型

使用命令行进行快速评测：

```bash
# 评测Qwen2.5-7B在MMLU和GSM8K上的表现
evalscope run \
    --model Qwen/Qwen2.5-7B-Instruct \
    --datasets mmlu gsm8k \
    --output-dir ./eval_results/qwen2.5-7b
```

使用Python API进行更灵活的评测：

```python
from evalscope import Evaluator
from evalscope.config import EvalConfig

# 创建评测配置
config = EvalConfig(
    model_id='Qwen/Qwen2.5-7B-Instruct',
    datasets=['mmlu', 'gsm8k', 'humaneval'],
    output_dir='./eval_results',
    
    # 推理参数
    generation_config={
        'max_new_tokens': 512,
        'temperature': 0.0,
        'do_sample': False
    },
    
    # 评测参数
    num_fewshot=5,  # few-shot数量
    batch_size=8,
    
    # 资源配置
    num_gpus=1,
    tensor_parallel_size=1
)

# 创建评估器并运行
evaluator = Evaluator(config)
results = evaluator.run()

# 打印结果摘要
print(results.summary())
```

### 评测结果解析

```python
import json
from pathlib import Path

def analyze_results(output_dir):
    """分析评测结果"""
    output_path = Path(output_dir)
    
    # 读取各数据集结果
    results = {}
    for result_file in output_path.glob('**/results.json'):
        dataset_name = result_file.parent.name
        with open(result_file) as f:
            results[dataset_name] = json.load(f)
            
    # 汇总统计
    summary = {}
    for dataset, data in results.items():
        summary[dataset] = {
            'accuracy': data.get('accuracy', 0),
            'total_samples': data.get('total', 0),
            'correct_samples': data.get('correct', 0)
        }
        
    return summary

# 分析结果
summary = analyze_results('./eval_results/qwen2.5-7b')
for dataset, metrics in summary.items():
    print(f"{dataset}: {metrics['accuracy']:.2%}")
```

## 多模型对比评测

### 批量评测多个模型

```python
from evalscope import BatchEvaluator

# 定义待评测模型列表
models = [
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct'
]

# 定义评测数据集
datasets = ['mmlu', 'gsm8k', 'humaneval', 'hellaswag']

# 批量评测
batch_evaluator = BatchEvaluator(
    models=models,
    datasets=datasets,
    output_dir='./comparison_results'
)

all_results = batch_evaluator.run()

# 生成对比表格
comparison_table = batch_evaluator.generate_comparison_table()
print(comparison_table)
```

### 可视化对比结果

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_model_comparison(results, models, datasets):
    """绘制模型对比图"""
    # 构建数据矩阵
    data = []
    for model in models:
        row = []
        for dataset in datasets:
            score = results[model][dataset].get('accuracy', 0)
            row.append(score)
        data.append(row)
        
    df = pd.DataFrame(data, index=models, columns=datasets)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(df.values, cmap='YlGn', aspect='auto')
    
    # 设置标签
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticklabels([m.split('/')[-1] for m in models])
    
    # 添加数值标注
    for i in range(len(models)):
        for j in range(len(datasets)):
            text = ax.text(j, i, f'{df.values[i, j]:.1%}',
                          ha='center', va='center', color='black')
    
    plt.colorbar(im)
    plt.title('Model Comparison Across Benchmarks')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

# 绘制对比图
plot_model_comparison(all_results, models, datasets)
```

## 自定义数据集评测

### 准备评测数据

首先准备符合格式要求的评测数据：

```python
# 创建自定义评测数据集
import json

eval_data = [
    {
        "id": "001",
        "question": "什么是机器学习中的过拟合？如何避免？",
        "reference": "过拟合是指模型在训练数据上表现很好，但在新数据上表现差的现象。避免方法包括：1）增加训练数据；2）使用正则化；3）早停；4）Dropout；5）数据增强等。",
        "category": "machine_learning",
        "difficulty": "medium"
    },
    {
        "id": "002", 
        "question": "请解释Transformer中的自注意力机制",
        "reference": "自注意力机制通过Query、Key、Value三个矩阵计算序列中各位置之间的关联权重。计算公式为Attention(Q,K,V) = softmax(QK^T/√d_k)V。它使模型能够捕获长距离依赖关系。",
        "category": "deep_learning",
        "difficulty": "hard"
    },
    {
        "id": "003",
        "question": "Python中列表和元组的区别是什么？",
        "reference": "列表(list)是可变的，可以增删改元素；元组(tuple)是不可变的，创建后不能修改。列表用方括号[]，元组用圆括号()。元组的性能略优于列表，且可以作为字典的键。",
        "category": "programming",
        "difficulty": "easy"
    }
]

# 保存为JSONL格式
with open('custom_eval_data.jsonl', 'w', encoding='utf-8') as f:
    for item in eval_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### 定义自定义评测任务

```python
from evalscope.datasets import BaseDataset
from evalscope.metrics import BaseMetric
from evalscope.tasks import EvalTask

class TechQADataset(BaseDataset):
    """技术问答评测数据集"""
    
    def __init__(self, data_path):
        super().__init__()
        self.data = self._load_data(data_path)
        
    def _load_data(self, path):
        import json
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'id': item['id'],
            'input': item['question'],
            'reference': item['reference'],
            'metadata': {
                'category': item.get('category'),
                'difficulty': item.get('difficulty')
            }
        }
        
    def get_prompt_template(self):
        return "请回答以下技术问题：\n\n{input}\n\n回答："


class TechQAMetric(BaseMetric):
    """技术问答评测指标"""
    
    def __init__(self, use_llm_judge=True, judge_model=None):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        
    def compute(self, predictions, references, metadata=None):
        results = []
        
        for pred, ref, meta in zip(predictions, references, metadata or [{}]*len(predictions)):
            if self.use_llm_judge:
                score = self._llm_judge(pred, ref)
            else:
                score = self._rule_based_score(pred, ref)
                
            results.append({
                'score': score,
                'category': meta.get('category'),
                'difficulty': meta.get('difficulty')
            })
            
        return self._aggregate_results(results)
        
    def _llm_judge(self, prediction, reference):
        """使用LLM作为评判者"""
        prompt = f"""
请评估以下回答的质量，与参考答案对比。

问题回答：
{prediction}

参考答案：
{reference}

请从以下维度评分（1-5分）：
1. 准确性：回答是否准确
2. 完整性：是否涵盖了关键点
3. 清晰度：表述是否清晰易懂

输出JSON格式：{{"accuracy": X, "completeness": X, "clarity": X, "overall": X}}
"""
        response = self.judge_model.generate(prompt)
        scores = json.loads(response)
        return scores['overall'] / 5.0  # 归一化到0-1
        
    def _rule_based_score(self, prediction, reference):
        """基于规则的评分"""
        # 关键词匹配
        ref_keywords = set(reference.lower().split())
        pred_keywords = set(prediction.lower().split())
        
        overlap = len(ref_keywords & pred_keywords)
        precision = overlap / len(pred_keywords) if pred_keywords else 0
        recall = overlap / len(ref_keywords) if ref_keywords else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
        
    def _aggregate_results(self, results):
        """聚合结果"""
        import numpy as np
        from collections import defaultdict
        
        # 总体统计
        overall_score = np.mean([r['score'] for r in results])
        
        # 按类别统计
        by_category = defaultdict(list)
        for r in results:
            if r.get('category'):
                by_category[r['category']].append(r['score'])
                
        category_scores = {cat: np.mean(scores) for cat, scores in by_category.items()}
        
        # 按难度统计
        by_difficulty = defaultdict(list)
        for r in results:
            if r.get('difficulty'):
                by_difficulty[r['difficulty']].append(r['score'])
                
        difficulty_scores = {diff: np.mean(scores) for diff, scores in by_difficulty.items()}
        
        return {
            'overall_score': overall_score,
            'by_category': category_scores,
            'by_difficulty': difficulty_scores,
            'num_samples': len(results)
        }
```

### 运行自定义评测

```python
from evalscope import Evaluator
from evalscope.models import load_model

# 加载评测模型
eval_model = load_model('Qwen/Qwen2.5-7B-Instruct')

# 加载评判模型（用于LLM-as-Judge）
judge_model = load_model('Qwen/Qwen2.5-72B-Instruct')

# 创建数据集和指标
dataset = TechQADataset('custom_eval_data.jsonl')
metric = TechQAMetric(use_llm_judge=True, judge_model=judge_model)

# 创建评测任务
task = EvalTask(
    name='tech_qa_eval',
    dataset=dataset,
    metrics=[metric],
    generation_config={
        'max_new_tokens': 512,
        'temperature': 0.0
    }
)

# 运行评测
evaluator = Evaluator(model=eval_model)
results = evaluator.evaluate(task)

# 输出结果
print("=" * 50)
print("评测结果")
print("=" * 50)
print(f"总体得分: {results['overall_score']:.2%}")
print("\n按类别得分:")
for cat, score in results['by_category'].items():
    print(f"  {cat}: {score:.2%}")
print("\n按难度得分:")
for diff, score in results['by_difficulty'].items():
    print(f"  {diff}: {score:.2%}")
```

## 评测报告生成

### 生成详细报告

```python
from evalscope.report import ReportGenerator
import datetime

class CustomReportGenerator(ReportGenerator):
    def generate_markdown_report(self, results, output_path):
        """生成Markdown格式报告"""
        report = f"""
# 模型评测报告

生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评测概览

| 指标 | 数值 |
|------|------|
| 评测模型 | {results.get('model_id', 'N/A')} |
| 评测样本数 | {results.get('num_samples', 0)} |
| 总体得分 | {results.get('overall_score', 0):.2%} |

## 分类别表现

| 类别 | 得分 | 样本数 |
|------|------|--------|
"""
        for cat, data in results.get('by_category', {}).items():
            report += f"| {cat} | {data['score']:.2%} | {data['count']} |\n"

        report += f"""
## 分难度表现

| 难度 | 得分 | 样本数 |
|------|------|--------|
"""
        for diff, data in results.get('by_difficulty', {}).items():
            report += f"| {diff} | {data['score']:.2%} | {data['count']} |\n"

        report += """
## 错误分析

### 低分样本示例

"""
        for example in results.get('low_score_examples', [])[:5]:
            report += f"""
**问题**: {example['question']}

**模型回答**: {example['prediction']}

**参考答案**: {example['reference']}

**得分**: {example['score']:.2f}

---
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report

# 生成报告
report_gen = CustomReportGenerator()
report = report_gen.generate_markdown_report(results, 'eval_report.md')
print("报告已生成: eval_report.md")
```

### 生成可视化报告

```python
import matplotlib.pyplot as plt
import numpy as np

def generate_visual_report(results, output_dir='./reports'):
    """生成可视化评测报告"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 总体得分仪表盘
    fig, ax = plt.subplots(figsize=(6, 6))
    score = results['overall_score']
    
    # 绘制环形图
    colors = ['#2ecc71' if score >= 0.8 else '#f39c12' if score >= 0.6 else '#e74c3c', '#ecf0f1']
    ax.pie([score, 1-score], colors=colors, startangle=90, 
           wedgeprops={'width': 0.3, 'edgecolor': 'white'})
    ax.text(0, 0, f'{score:.1%}', ha='center', va='center', fontsize=32, fontweight='bold')
    ax.set_title('Overall Score', fontsize=16, pad=20)
    plt.savefig(f'{output_dir}/overall_score.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 分类别雷达图
    categories = list(results['by_category'].keys())
    scores = [results['by_category'][cat]['score'] for cat in categories]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, scores_plot, alpha=0.25)
    ax.plot(angles, scores_plot, 'o-', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Performance by Category', size=16, y=1.1)
    plt.savefig(f'{output_dir}/category_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 难度分布柱状图
    difficulties = list(results['by_difficulty'].keys())
    diff_scores = [results['by_difficulty'][d]['score'] for d in difficulties]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(difficulties, diff_scores, color=['#27ae60', '#f39c12', '#e74c3c'])
    ax.set_ylabel('Score')
    ax.set_xlabel('Difficulty')
    ax.set_ylim(0, 1)
    ax.set_title('Performance by Difficulty')
    
    for bar, score in zip(bars, diff_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.1%}', ha='center', va='bottom')
    
    plt.savefig(f'{output_dir}/difficulty_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化报告已生成到 {output_dir}/")

# 生成可视化报告
generate_visual_report(results)
```

## 完整评测流程示例

以下是一个完整的评测脚本，整合了上述所有步骤：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的模型评测流程示例
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from evalscope import Evaluator
from evalscope.config import EvalConfig
from evalscope.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--model', type=str, required=True, help='Model ID or path')
    parser.add_argument('--datasets', nargs='+', default=['mmlu', 'gsm8k'])
    parser.add_argument('--custom-data', type=str, help='Path to custom eval data')
    parser.add_argument('--output-dir', type=str, default='./eval_outputs')
    parser.add_argument('--num-fewshot', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-llm-judge', action='store_true')
    parser.add_argument('--judge-model', type=str, default='Qwen/Qwen2.5-72B-Instruct')
    return parser.parse_args()


def run_standard_eval(model_id, datasets, config):
    """运行标准基准评测"""
    print(f"\n{'='*50}")
    print(f"Running standard benchmark evaluation")
    print(f"{'='*50}")
    
    eval_config = EvalConfig(
        model_id=model_id,
        datasets=datasets,
        output_dir=config['output_dir'],
        num_fewshot=config['num_fewshot'],
        batch_size=config['batch_size'],
        generation_config={
            'max_new_tokens': 512,
            'temperature': 0.0
        }
    )
    
    evaluator = Evaluator(eval_config)
    results = evaluator.run()
    
    return results


def run_custom_eval(model, data_path, config):
    """运行自定义数据评测"""
    print(f"\n{'='*50}")
    print(f"Running custom dataset evaluation")
    print(f"{'='*50}")
    
    # 加载自定义数据
    dataset = TechQADataset(data_path)
    
    # 设置评测指标
    if config.get('use_llm_judge'):
        judge_model = load_model(config['judge_model'])
        metric = TechQAMetric(use_llm_judge=True, judge_model=judge_model)
    else:
        metric = TechQAMetric(use_llm_judge=False)
    
    # 创建评测任务
    task = EvalTask(
        name='custom_eval',
        dataset=dataset,
        metrics=[metric]
    )
    
    evaluator = Evaluator(model=model)
    results = evaluator.evaluate(task)
    
    return results


def save_results(results, output_dir):
    """保存评测结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON结果
    results_file = output_path / 'results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成Markdown报告
    report_file = output_path / 'report.md'
    report_gen = CustomReportGenerator()
    report_gen.generate_markdown_report(results, str(report_file))
    
    # 生成可视化报告
    generate_visual_report(results, str(output_path / 'visualizations'))
    
    print(f"\nResults saved to {output_dir}")


def main():
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.model.replace('/', '_')}_{timestamp}"
    
    config = {
        'output_dir': str(output_dir),
        'num_fewshot': args.num_fewshot,
        'batch_size': args.batch_size,
        'use_llm_judge': args.use_llm_judge,
        'judge_model': args.judge_model
    }
    
    all_results = {
        'model_id': args.model,
        'timestamp': timestamp,
        'config': config
    }
    
    # 运行标准基准评测
    if args.datasets:
        standard_results = run_standard_eval(args.model, args.datasets, config)
        all_results['standard_benchmarks'] = standard_results
    
    # 运行自定义数据评测
    if args.custom_data:
        model = load_model(args.model)
        custom_results = run_custom_eval(model, args.custom_data, config)
        all_results['custom_eval'] = custom_results
    
    # 保存结果
    save_results(all_results, str(output_dir))
    
    # 打印摘要
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    
    if 'standard_benchmarks' in all_results:
        print("\nStandard Benchmarks:")
        for dataset, score in all_results['standard_benchmarks'].items():
            print(f"  {dataset}: {score.get('accuracy', 0):.2%}")
    
    if 'custom_eval' in all_results:
        print(f"\nCustom Evaluation:")
        print(f"  Overall Score: {all_results['custom_eval']['overall_score']:.2%}")


if __name__ == '__main__':
    main()
```

运行评测：

```bash
# 运行标准基准评测
python eval_script.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --datasets mmlu gsm8k humaneval

# 运行自定义数据评测（使用LLM-as-Judge）
python eval_script.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --custom-data custom_eval_data.jsonl \
    --use-llm-judge \
    --judge-model Qwen/Qwen2.5-72B-Instruct

# 完整评测
python eval_script.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --datasets mmlu gsm8k \
    --custom-data custom_eval_data.jsonl \
    --use-llm-judge
```

通过本节的实践，读者应能够熟练使用EvalScope进行标准基准评测和自定义评测，并能够根据业务需求构建完整的模型评测流程。
