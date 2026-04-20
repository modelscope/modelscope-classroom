# 自定义Benchmark

通用评测基准虽然提供了标准化的评估框架，但往往难以完全匹配特定业务场景的需求。构建自定义Benchmark能够针对性地评测模型在目标领域的实际表现，是大模型落地应用中的重要环节。

这就像招聘一位专科医生——你不能光看他高考成绩好不好，还得出一套专门的临床技能考核。MMLU的“全科考试”告诉你这位候选人基础不错，但能不能在你的科室里胜任工作，还需要专门的考核方案来验证。

## 自定义评测的必要性

### 业务场景特殊性

不同业务场景对模型能力的要求各不相同。假设你正在为一家医院部署智能问诊系统，你会发现MMLU里的临床医学题目和真实问诊场景差别巨大——真实场景里，患者会用方言描述症状，会同时询问多种不相关的问题，还会反复确认"你确定没问题吗？"——这些通用评测集里可没有。来看一下各场景的具体差异：

| 应用场景 | 核心能力要求 | 通用评测覆盖度 |
|----------|--------------|----------------|
| 客服对话 | 意图识别、情感理解、话术规范 | 低 |
| 法律咨询 | 法条检索、案例分析、专业术语 | 低 |
| 医疗问诊 | 症状分析、用药建议、安全边界 | 低 |
| 代码审查 | 特定语言/框架、代码规范、安全漏洞 | 中 |
| 金融分析 | 数据解读、风险评估、合规要求 | 低 |

### 数据分布差异

预训练数据的分布与目标应用场景可能存在显著差异。举个例子，一个模型在训练时见过大量英语法律文本，但你需要它处理中国《民法典》的咨询，这两者之间的差距可不是翻译一下那么简单。具体差异包括：
- 专业术语和行业黑话（金融行业的"回调"、医疗行业的"阳性"，和日常用语里的含义完全不同）
- 特定格式和结构化输出（法律文书的固定格式、医疗报告的标准模板）
- 领域知识的深度和广度
- 安全合规的特殊要求（医疗场景下不能随便给用药建议，金融场景下不能给出确定性的投资承诺）

### 评测粒度需求

通用评测提供宏观能力评估，而业务落地需要更细粒度的能力诊断。这就好比体检报告告诉你"身体健康"，但你真正需要知道的是"血糖多少、血压多少、肏器功能如何"这样的细节：

```
通用评测: 模型在MMLU上准确率75%
业务需求: 
- 产品问答准确率多少？
- 复杂咨询场景表现如何？
- 边界情况处理是否安全？
- 多轮对话理解是否连贯？
```

## 评测集设计原则

### 代表性原则

评测数据应覆盖目标场景的典型用例分布。想象一下你在出一套客服场景的考试卷，如果80%的真实用户问的是"物流查询"，你的试卷却80%都是"退换货"题目，那考试结果就没有参考价值了。评测集的分布应尽量贴合生产环境的实际分布：

```python
def analyze_production_distribution(logs):
    """分析生产环境查询分布"""
    from collections import Counter
    
    # 提取查询意图
    intents = [extract_intent(log['query']) for log in logs]
    intent_dist = Counter(intents)
    
    # 提取查询复杂度
    complexities = [measure_complexity(log['query']) for log in logs]
    complexity_dist = {
        'simple': sum(1 for c in complexities if c < 0.3),
        'medium': sum(1 for c in complexities if 0.3 <= c < 0.7),
        'complex': sum(1 for c in complexities if c >= 0.7)
    }
    
    return {
        'intent_distribution': dict(intent_dist.most_common()),
        'complexity_distribution': complexity_dist
    }
```

评测集应按照生产环境的分布进行采样或加权。

### 区分度原则

好的评测集应能有效区分不同能力水平的模型。回到招聘的比喻：如果面试题太简单，每个候选人都能拿满分，你就没法筛选出最优秀的那个；反之如果太难，每个人都交白卷，同样无法区分。好的试卷应该让不同水平的候选人拉开差距：

- 避免过于简单的题目（所有模型都能答对，没有区分价值）
- 避免过于困难的题目（所有模型都答错，同样无法区分）
- 设置不同难度梯度（就像高考试卷的“基础题—中等题—压轴题”配比）

```python
def calculate_discrimination(item_scores):
    """计算题目区分度"""
    # 将考生按总分分为高分组和低分组
    n = len(item_scores)
    high_group = item_scores[:n//3]  # 前1/3
    low_group = item_scores[-n//3:]   # 后1/3
    
    # 区分度 = 高分组正确率 - 低分组正确率
    discrimination = np.mean(high_group) - np.mean(low_group)
    
    # 区分度解释
    # > 0.4: 很好
    # 0.3-0.4: 良好
    # 0.2-0.3: 可接受
    # < 0.2: 需修改
    
    return discrimination
```

### 可扩展性原则

评测框架应支持持续迭代。在实际项目中，业务需求是不断变化的——今天可能加了新的产品线，明天可能改了客服话术规范。你的评测集必须能够跟上这种变化，而不是“一锤子买卖”。以下是一个支持版本管理和增量更新的评测集框架：

```python
class EvalDataset:
    def __init__(self, version='1.0'):
        self.version = version
        self.items = []
        self.metadata = {
            'created_at': datetime.now(),
            'version': version,
            'categories': set()
        }
        
    def add_item(self, item):
        """添加评测项"""
        item['id'] = self._generate_id()
        item['added_version'] = self.version
        self.items.append(item)
        self.metadata['categories'].add(item.get('category', 'default'))
        
    def filter_by_version(self, min_version):
        """按版本筛选"""
        return [item for item in self.items 
                if item['added_version'] >= min_version]
        
    def get_subset(self, category=None, difficulty=None, n_samples=None):
        """获取子集"""
        subset = self.items
        if category:
            subset = [x for x in subset if x.get('category') == category]
        if difficulty:
            subset = [x for x in subset if x.get('difficulty') == difficulty]
        if n_samples and len(subset) > n_samples:
            subset = random.sample(subset, n_samples)
        return subset
```

## 数据收集方法

### 从生产日志提取

真实用户查询是最有价值的评测数据来源。这就像一家餐厅想知道菜品好不好吃，与其自己在后厨试吃，不如直接看顧客的真实反馈——哪些菜点的多、哪些菜被退回、哪些菜被反复追问"这个能不能不放辣"。同理，从生产日志里挖掘出的真实用户问题，最能反映模型在实战中的真实表现：

```python
def extract_eval_candidates(logs, min_quality_score=0.7):
    """从生产日志中提取评测候选"""
    candidates = []
    
    for log in logs:
        # 过滤低质量交互
        if log.get('user_rating', 0) < 3:
            continue
            
        # 提取有明确答案的查询
        if log.get('has_ground_truth', False):
            candidates.append({
                'query': log['query'],
                'reference': log['response'],
                'context': log.get('context'),
                'metadata': {
                    'source': 'production',
                    'timestamp': log['timestamp'],
                    'user_rating': log.get('user_rating')
                }
            })
            
    # 去重
    candidates = deduplicate(candidates, key='query')
    
    return candidates
```

### 专家标注

领域专家提供高质量的问答对。如果生产日志是"顾客反馈"，那专家标注就是"米其林评审员"的专业评价。专家能够构造出那些生产环境中不常见、但一旦出现就很关键的边界场景：

```python
# 标注任务模板
annotation_template = {
    'task_id': str,
    'query': str,
    'expected_answer': str,
    'acceptable_variants': list,  # 可接受的答案变体
    'difficulty': ['easy', 'medium', 'hard'],
    'category': str,
    'reasoning_required': bool,
    'annotator_notes': str,
    'quality_check': {
        'reviewed_by': str,
        'review_date': str,
        'approved': bool
    }
}
```

### 合成数据生成

利用规则或模型生成评测数据。当真实数据不够用时，可以让一个更强的模型来帮忙"出题"——就像让一位资深教师根据教学大纲编写练习题。不过要注意，合成数据的质量必须经过人工核查，否则就像让AI自己出题自己答——题目和答案里的错误可能会“共谋”而难以发现：

```python
def generate_synthetic_qa(seed_data, model, n_samples=100):
    """合成问答数据"""
    generated = []
    
    for seed in seed_data:
        prompt = f"""
基于以下信息生成一个问答对：

背景知识：{seed['knowledge']}
主题：{seed['topic']}
难度：{seed['difficulty']}

请生成：
1. 一个自然的用户问题
2. 准确、完整的答案
3. 该问题测试的能力点

输出JSON格式：
{{"question": "...", "answer": "...", "skill_tested": "..."}}
"""
        response = model.generate(prompt)
        qa = json.loads(response)
        qa['source'] = 'synthetic'
        qa['seed_id'] = seed['id']
        generated.append(qa)
        
    return generated
```

### 对抗样本构造

创建挑战模型弱点的测试用例。这是评测集中最“恶意”但也最有价值的部分——专门找模型的软肋下手。就像一个好的临床考试不会只考常见病，还会故意出一些容易混淆的症状描述、带有误导性信息的病例，看你会不会被带偏：

```python
def create_adversarial_samples(original_samples, attack_types):
    """构造对抗样本"""
    adversarial = []
    
    for sample in original_samples:
        for attack in attack_types:
            if attack == 'typo':
                # 引入拼写错误
                perturbed = introduce_typos(sample['query'])
            elif attack == 'paraphrase':
                # 同义改写
                perturbed = paraphrase(sample['query'])
            elif attack == 'negation':
                # 添加否定
                perturbed = add_negation(sample['query'])
            elif attack == 'distractor':
                # 添加干扰信息
                perturbed = add_distractor(sample['query'])
                
            adversarial.append({
                'query': perturbed,
                'original_query': sample['query'],
                'expected_answer': sample['expected_answer'],
                'attack_type': attack
            })
            
    return adversarial
```

## 标注规范设计

### 标注指南示例

```markdown
# 客服对话评测标注指南 v1.0

## 任务描述
评估模型回复是否满足客服场景的质量要求

## 评分维度

### 1. 准确性 (1-5分)
- 5分：完全正确，无事实错误
- 4分：基本正确，有轻微不准确
- 3分：部分正确，有明显错误但核心正确
- 2分：大部分错误，仅少量正确
- 1分：完全错误或无关

### 2. 完整性 (1-5分)
- 5分：完整回答所有问题点
- 4分：回答主要问题，遗漏次要点
- 3分：回答部分问题
- 2分：回答不完整，遗漏重要信息
- 1分：几乎未回答问题

### 3. 话术规范 (1-5分)
- 5分：完全符合客服话术规范
- 4分：基本符合，有轻微偏差
- 3分：部分符合
- 2分：多处不符合
- 1分：严重违反规范

### 4. 安全性 (通过/不通过)
- 通过：无敏感信息泄露、无误导性内容
- 不通过：存在安全风险

## 标注示例

[示例1]
用户：我的订单什么时候到？
模型：根据物流信息，您的订单预计明天下午送达。
准确性：4（假设物流信息正确）
完整性：4（未提供物流单号）
话术规范：5
安全性：通过
```

### 标注质量控制

```python
class AnnotationQualityControl:
    def __init__(self, min_agreement=0.8):
        self.min_agreement = min_agreement
        
    def calculate_inter_annotator_agreement(self, annotations):
        """计算标注者一致性（Cohen's Kappa）"""
        from sklearn.metrics import cohen_kappa_score
        
        annotator_pairs = list(combinations(annotations.keys(), 2))
        kappa_scores = []
        
        for a1, a2 in annotator_pairs:
            labels1 = [annotations[a1][item_id] for item_id in sorted(annotations[a1])]
            labels2 = [annotations[a2][item_id] for item_id in sorted(annotations[a2])]
            kappa = cohen_kappa_score(labels1, labels2)
            kappa_scores.append(kappa)
            
        return np.mean(kappa_scores)
        
    def adjudicate_disagreements(self, annotations, threshold=2):
        """处理标注分歧"""
        adjudicated = {}
        
        for item_id in annotations:
            labels = [ann[item_id] for ann in annotations.values()]
            
            if max(labels) - min(labels) > threshold:
                # 分歧过大，需要专家仲裁
                adjudicated[item_id] = {
                    'status': 'needs_review',
                    'labels': labels
                }
            else:
                # 取众数或均值
                adjudicated[item_id] = {
                    'status': 'resolved',
                    'final_label': statistics.median(labels)
                }
                
        return adjudicated
```

## 评测指标设计

### 任务特定指标

```python
class CustomMetrics:
    @staticmethod
    def intent_accuracy(predictions, references):
        """意图识别准确率"""
        correct = sum(p['intent'] == r['intent'] 
                     for p, r in zip(predictions, references))
        return correct / len(predictions)
        
    @staticmethod
    def slot_f1(predictions, references):
        """槽位填充F1"""
        true_positives = 0
        pred_count = 0
        ref_count = 0
        
        for p, r in zip(predictions, references):
            pred_slots = set(p.get('slots', {}).items())
            ref_slots = set(r.get('slots', {}).items())
            
            true_positives += len(pred_slots & ref_slots)
            pred_count += len(pred_slots)
            ref_count += len(ref_slots)
            
        precision = true_positives / pred_count if pred_count else 0
        recall = true_positives / ref_count if ref_count else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        
        return f1
        
    @staticmethod
    def safety_rate(predictions, safety_checker):
        """安全通过率"""
        safe_count = sum(safety_checker(p['response']) for p in predictions)
        return safe_count / len(predictions)
        
    @staticmethod
    def format_compliance(predictions, format_spec):
        """格式合规率"""
        compliant = sum(validate_format(p['response'], format_spec) 
                       for p in predictions)
        return compliant / len(predictions)
```

### LLM-as-Judge评测

使用大模型作为评判者进行主观维度评测。这是一个非常巧妙的思路：既然人工评分成本高、耗时长，为什么不用一个更强的模型来当"考官"呢？这就像让一位资深主管来评估初级员工的工作成果。不过要注意，LLM评判者自身也有偏见（比如偏好更长的回答、偏好自己风格的回答），因此多评判者共识机制很重要：

```python
def llm_judge_evaluate(model_output, reference, judge_model, criteria):
    """使用LLM作为评判者"""
    prompt = f"""
请作为专业评审员，评估以下回答的质量。

【问题】
{model_output['query']}

【标准答案】
{reference}

【待评估回答】
{model_output['response']}

【评分标准】
{criteria}

请按照以下格式输出评分：
1. 各维度分数（1-5分）
2. 总体评价
3. 改进建议

输出JSON格式：
{{
    "scores": {{"准确性": X, "完整性": X, "流畅性": X}},
    "overall": X,
    "comments": "..."
}}
"""
    
    judge_response = judge_model.generate(prompt, temperature=0)
    return json.loads(judge_response)


def multi_judge_consensus(model_output, reference, judges, criteria):
    """多评判者共识"""
    scores = []
    
    for judge in judges:
        score = llm_judge_evaluate(model_output, reference, judge, criteria)
        scores.append(score)
        
    # 计算平均分并检测异常值
    avg_scores = {}
    for key in scores[0]['scores']:
        values = [s['scores'][key] for s in scores]
        avg_scores[key] = np.mean(values)
        
        # 检测异常（偏离均值超过1.5分）
        if max(values) - min(values) > 1.5:
            print(f"Warning: Large disagreement on {key}")
            
    return avg_scores
```

## 评测集管理

### 版本控制

```python
class BenchmarkVersion:
    def __init__(self, version_id, parent_version=None):
        self.version_id = version_id
        self.parent_version = parent_version
        self.items = []
        self.changelog = []
        
    def add_items(self, items, reason):
        """添加新题目"""
        self.items.extend(items)
        self.changelog.append({
            'action': 'add',
            'count': len(items),
            'reason': reason,
            'timestamp': datetime.now()
        })
        
    def remove_items(self, item_ids, reason):
        """移除题目"""
        self.items = [x for x in self.items if x['id'] not in item_ids]
        self.changelog.append({
            'action': 'remove',
            'count': len(item_ids),
            'reason': reason,
            'timestamp': datetime.now()
        })
        
    def export(self, path):
        """导出评测集"""
        data = {
            'version': self.version_id,
            'parent': self.parent_version,
            'items': self.items,
            'changelog': self.changelog,
            'statistics': self.get_statistics()
        }
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
```

### 数据污染防护

```python
class ContaminationGuard:
    def __init__(self, benchmark_data):
        self.benchmark_hashes = self._compute_hashes(benchmark_data)
        
    def _compute_hashes(self, data):
        """计算评测数据的指纹"""
        import hashlib
        hashes = set()
        
        for item in data:
            # 对问题文本计算hash
            text = item['query'].lower().strip()
            h = hashlib.md5(text.encode()).hexdigest()
            hashes.add(h)
            
            # n-gram hash
            words = text.split()
            for n in [5, 10, 15]:
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    h = hashlib.md5(ngram.encode()).hexdigest()
                    hashes.add(h)
                    
        return hashes
        
    def check_training_data(self, training_texts):
        """检查训练数据是否包含评测数据"""
        contaminated = []
        
        for i, text in enumerate(training_texts):
            text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
            if text_hash in self.benchmark_hashes:
                contaminated.append(i)
                
        contamination_rate = len(contaminated) / len(training_texts)
        return {
            'contaminated_indices': contaminated,
            'contamination_rate': contamination_rate
        }
```

## 完整评测集构建流程

```python
def build_custom_benchmark(config):
    """构建自定义评测集的完整流程"""
    
    # 1. 数据收集
    print("Step 1: Collecting data...")
    production_data = collect_from_logs(config['log_path'])
    expert_data = load_expert_annotations(config['annotation_path'])
    synthetic_data = generate_synthetic(config['seed_data'], config['model'])
    
    # 2. 数据清洗
    print("Step 2: Cleaning data...")
    all_data = production_data + expert_data + synthetic_data
    cleaned_data = clean_and_deduplicate(all_data)
    
    # 3. 质量筛选
    print("Step 3: Quality filtering...")
    filtered_data = filter_by_quality(cleaned_data, min_score=config['min_quality'])
    
    # 4. 难度标注
    print("Step 4: Difficulty annotation...")
    for item in filtered_data:
        item['difficulty'] = estimate_difficulty(item)
        
    # 5. 平衡采样
    print("Step 5: Balanced sampling...")
    balanced_data = stratified_sample(
        filtered_data,
        strata=['category', 'difficulty'],
        target_size=config['target_size']
    )
    
    # 6. 标注验证
    print("Step 6: Annotation verification...")
    verified_data = verify_annotations(balanced_data, config['validators'])
    
    # 7. 创建评测集
    print("Step 7: Creating benchmark...")
    benchmark = BenchmarkVersion(version_id=config['version'])
    benchmark.add_items(verified_data, reason='Initial creation')
    
    # 8. 导出
    print("Step 8: Exporting...")
    benchmark.export(config['output_path'])
    
    # 9. 生成报告
    print("Step 9: Generating report...")
    report = generate_benchmark_report(benchmark)
    
    return benchmark, report
```

总结一下，构建自定义Benchmark的过程很像设计一场高质量的招聘考试：你需要明确岗位能力要求（业务场景分析）、出一套有区分度的试卷（题目设计）、让多个考官交叉评分（质量控制）、并且随着业务发展持续更新题库（版本迭代）。通过这样科学的评测集设计和严格的质量控制，自定义Benchmark能够为特定业务场景提供精准的模型能力评估，有效指导模型的迭代优化和选型决策。
