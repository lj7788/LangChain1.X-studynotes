# 06_retrieval_compression.py

## 功能说明

`ContextualCompressionRetriever` 可以压缩检索到的文档，只保留与查询相关的部分，减少输入给 LLM 的上下文长度。

## 核心概念

### 为什么需要文档压缩？

在 RAG 应用中，检索到的文档可能包含很多与查询无关的内容：
- 向量检索是基于相似度的，可能返回相关但不精确的结果
- 完整的文档可能很长，浪费 token 成本
- LLM 需要处理更多无关信息，可能影响答案质量

### LLMChainExtractor

使用 LLM 来提取与查询最相关的内容：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

## 其他压缩器选项

| 压缩器 | 说明 |
|--------|------|
| LLMChainExtractor | 使用 LLM 提取相关内容 |
| LLMChainFilter | 使用 LLM 判断文档是否相关 |
| DocumentCompressorPipeline | 组合多个压缩器 |

## 运行示例

```bash
python 阶段3/06_retrieval_compression.py
```

## 输出示例

```
=== 基础检索（未压缩）===

查询: Python 在数据科学和机器学习中的应用

--- 文档 1 ---
内容: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。
Python 设计哲学强调代码的可读性...
来源: python_intro

--- 文档 2 ---
内容: JavaScript 是 Web 的编程语言。JavaScript 可以更新和更改 HTML 和 CSS...
来源: javascript_intro

--- 文档 3 ---
内容: Python 的应用领域非常广泛。在 Web 开发方面，Django 和 Flask 是流行的框架。
在数据科学领域，NumPy、Pandas 和 Matplotlib 是重要的库...
来源: python_applications

=== 使用 ContextualCompression 压缩检索 ===

查询: Python 在数据科学和机器学习中的应用

--- 压缩后的文档 1 ---
内容: 在数据科学领域，NumPy、Pandas 和 Matplotlib 是重要的库。Python 还广泛用于机器学习和人工智能，TensorFlow、PyTorch 和 Scikit-learn 是常用的 ML 框架。
来源: python_applications
```
