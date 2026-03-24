# RAG技术

RAG（Retrieval-Augmented Generation，检索增强生成）是将信息检索与语言生成相结合的技术，是构建知识密集型智能体应用的核心技术之一。

## RAG概述

### 为什么需要RAG

| LLM局限 | RAG解决方案 |
|---------|------------|
| 知识截止日期 | 实时检索最新信息 |
| 幻觉问题 | 基于检索到的事实生成 |
| 领域知识不足 | 接入专业知识库 |
| 私有数据无法使用 | 安全地使用企业数据 |

### 基本架构

```
用户查询 → 查询处理 → 检索器 → 知识库
                          ↓
                      相关文档
                          ↓
                      上下文构建
                          ↓
                   LLM生成回答 → 用户
```

## 核心组件

### Embedding模型

Embedding模型将文本转换为向量表示，是实现语义检索的基础：

```python
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: list) -> np.ndarray:
        """批量编码文本"""
        return self.model.encode(texts, normalize_embeddings=True)
        
    def encode_query(self, query: str) -> np.ndarray:
        """编码查询（可能有不同的指令前缀）"""
        # BGE模型对查询有特殊处理
        query_with_instruction = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.model.encode([query_with_instruction], normalize_embeddings=True)[0]
```

常用Embedding模型对比：

| 模型 | 维度 | 中文支持 | 特点 |
|------|------|----------|------|
| BGE-large-zh | 1024 | 优秀 | 中文领域领先 |
| text-embedding-3-large | 3072 | 良好 | OpenAI最新模型 |
| GTE-large | 1024 | 优秀 | 阿里开源 |
| E5-large | 1024 | 一般 | 微软出品 |

### 向量数据库

向量数据库存储和检索文档向量：

```python
import chromadb

class VectorStore:
    def __init__(self, collection_name="documents"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_documents(self, documents: list, embeddings: np.ndarray, ids: list = None):
        """添加文档"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
            
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """检索相似文档"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return [
            {"document": doc, "distance": dist}
            for doc, dist in zip(results["documents"][0], results["distances"][0])
        ]
```

### Reranker模型

Reranker对初步检索结果进行精排：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        """重排序文档"""
        pairs = [[query, doc] for doc in documents]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()
            
        # 按分数排序
        sorted_indices = torch.argsort(scores, descending=True)[:top_k]
        
        return [
            {"document": documents[i], "score": scores[i].item()}
            for i in sorted_indices
        ]
```

## 多路召回

单一检索方式可能遗漏相关文档，多路召回结合多种检索策略：

```python
class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, embedding_model, bm25_index, vector_store):
        self.embedding = embedding_model
        self.bm25 = bm25_index
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k: int = 10) -> list:
        # 向量检索
        query_embedding = self.embedding.encode_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k)
        
        # BM25关键词检索
        bm25_results = self.bm25.search(query, top_k)
        
        # 融合结果（RRF融合）
        fused = self._reciprocal_rank_fusion([vector_results, bm25_results])
        
        return fused[:top_k]
        
    def _reciprocal_rank_fusion(self, result_lists: list, k: int = 60) -> list:
        """RRF融合算法"""
        scores = {}
        
        for results in result_lists:
            for rank, item in enumerate(results):
                doc = item["document"]
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += 1 / (k + rank + 1)
                
        # 按融合分数排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [{"document": doc, "score": score} for doc, score in sorted_docs]
```

## RAG Pipeline

### 完整流程

```python
class RAGPipeline:
    """完整的RAG流程"""
    
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        
    def query(self, question: str) -> str:
        # 1. 查询改写（可选）
        rewritten_query = self._rewrite_query(question)
        
        # 2. 检索
        candidates = self.retriever.retrieve(rewritten_query, top_k=20)
        
        # 3. 重排序
        reranked = self.reranker.rerank(question, 
                                        [c["document"] for c in candidates],
                                        top_k=5)
        
        # 4. 构建上下文
        context = self._build_context(reranked)
        
        # 5. 生成回答
        answer = self._generate(question, context)
        
        return answer
        
    def _rewrite_query(self, query: str) -> str:
        """查询改写：扩展或澄清查询"""
        prompt = f"""请将以下用户查询改写为更适合检索的形式：
        
原始查询：{query}

改写要求：
1. 扩展缩写和专业术语
2. 添加相关的同义词
3. 保持核心语义不变

改写后的查询："""
        
        return self.llm.generate(prompt).strip()
        
    def _build_context(self, documents: list) -> str:
        """构建上下文"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[文档{i}]\n{doc['document']}")
            
        return "\n\n".join(context_parts)
        
    def _generate(self, question: str, context: str) -> str:
        """基于上下文生成回答"""
        prompt = f"""基于以下参考资料回答问题。如果资料中没有相关信息，请如实说明。

参考资料：
{context}

问题：{question}

回答："""
        
        return self.llm.generate(prompt)
```

### 高级技术

#### 递归检索

对于复杂问题，可能需要多轮检索：

```python
class RecursiveRAG:
    def __init__(self, rag_pipeline, max_iterations=3):
        self.rag = rag_pipeline
        self.max_iterations = max_iterations
        
    def query(self, question: str) -> str:
        accumulated_context = []
        current_question = question
        
        for i in range(self.max_iterations):
            # 检索
            results = self.rag.retriever.retrieve(current_question)
            accumulated_context.extend([r["document"] for r in results])
            
            # 检查是否有足够信息
            if self._has_sufficient_info(question, accumulated_context):
                break
                
            # 生成后续问题
            current_question = self._generate_followup(question, accumulated_context)
            
        return self.rag._generate(question, "\n".join(accumulated_context))
```

#### 自适应检索

根据问题类型决定是否需要检索：

```python
class AdaptiveRAG:
    def __init__(self, llm, rag_pipeline):
        self.llm = llm
        self.rag = rag_pipeline
        
    def query(self, question: str) -> str:
        # 判断是否需要检索
        needs_retrieval = self._needs_retrieval(question)
        
        if needs_retrieval:
            return self.rag.query(question)
        else:
            # 直接用LLM回答
            return self.llm.generate(f"请回答：{question}")
            
    def _needs_retrieval(self, question: str) -> bool:
        prompt = f"""判断以下问题是否需要检索外部知识才能回答。

问题：{question}

判断标准：
- 需要具体事实或数据 -> 需要检索
- 需要最新信息 -> 需要检索
- 是常识性问题或推理问题 -> 不需要检索

请只回答"是"或"否"。"""
        
        response = self.llm.generate(prompt).strip()
        return response == "是"
```

## 知识库构建

### 文档处理

```python
class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def load_and_split(self, file_path: str) -> list:
        """加载文档并分块"""
        # 加载文档
        text = self._load_file(file_path)
        
        # 分块
        chunks = self._split_text(text)
        
        # 添加元数据
        return [
            {"content": chunk, "source": file_path, "chunk_id": i}
            for i, chunk in enumerate(chunks)
        ]
        
    def _split_text(self, text: str) -> list:
        """文本分块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在句子边界分割
            if end < len(text):
                # 寻找最近的句子结束符
                for sep in ["。", "！", "？", "\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + 1
                        break
                        
            chunks.append(text[start:end])
            start = end - self.overlap
            
        return chunks
```

### 索引构建

```python
class KnowledgeBase:
    """知识库"""
    
    def __init__(self, embedding_model, vector_store):
        self.processor = DocumentProcessor()
        self.embedding = embedding_model
        self.store = vector_store
        
    def index_documents(self, file_paths: list):
        """索引文档"""
        all_chunks = []
        
        for path in file_paths:
            chunks = self.processor.load_and_split(path)
            all_chunks.extend(chunks)
            
        # 批量生成embeddings
        texts = [c["content"] for c in all_chunks]
        embeddings = self.embedding.encode(texts)
        
        # 存储
        self.store.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"]} for c in all_chunks]
        )
        
        print(f"索引完成，共{len(all_chunks)}个文档块")
```

## 评估与优化

### 检索质量评估

```python
class RetrievalEvaluator:
    def evaluate(self, queries: list, ground_truth: list, retriever) -> dict:
        """评估检索质量"""
        metrics = {
            "recall@5": [],
            "mrr": [],
            "ndcg@5": []
        }
        
        for query, relevant_docs in zip(queries, ground_truth):
            results = retriever.retrieve(query, top_k=5)
            retrieved_docs = [r["document"] for r in results]
            
            # Recall@5
            recall = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)
            metrics["recall@5"].append(recall)
            
            # MRR
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_docs:
                    metrics["mrr"].append(1 / rank)
                    break
            else:
                metrics["mrr"].append(0)
                
        return {k: np.mean(v) for k, v in metrics.items()}
```

RAG技术是构建知识型智能体的核心。通过合理的检索策略、高效的向量存储和精心设计的生成提示，可以显著提升智能体回答的准确性和可靠性。
