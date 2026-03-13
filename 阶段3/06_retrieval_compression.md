# 06_retrieval_compression.py

## 功能说明

`ContextualCompressionRetriever` 可以压缩检索到的文档，只保留与查询相关的部分，减少输入给 LLM 的上下文长度。

## 核心特性

- **智能压缩**: 使用 LLM 提取文档中与查询相关的内容
- **减少 Token 消耗**: 避免将整个文档发送给 LLM
- **保留关键信息**: 只保留与查询最相关的部分

## 关键组件

### ContextualCompressionRetriever

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

### LLMChainExtractor

从文档中提取与查询相关的部分。

## 关键参数

| 参数 | 说明 |
|------|------|
| base_compressor | 压缩器（如 LLMChainExtractor） |
| base_retriever | 基础检索器 |

## 运行示例

```bash
python 阶段3/06_retrieval_compression.py
```

## 输出示例

```
=== 基础检索（未压缩）===

查询: Python 在数据科学和机器学习中的应用

--- 文档 1 ---
内容: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布...
来源: python_intro

--- 文档 2 ---
内容: Python 的应用领域非常广泛。在 Web 开发方面，Django 和 Flask 是流行的框架...
来源: python_applications

=== 使用 ContextualCompression 压缩检索 ===

查询: Python 在数据科学和机器学习中的应用

--- 压缩后的文档 1 ---
内容: 在数据科学领域，NumPy、Pandas 和 Matplotlib 是重要的库。Python 还广泛用于机器学习和人工智能
来源: python_applications
```
