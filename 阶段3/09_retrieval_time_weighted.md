# 09_retrieval_time_weighted.py

## 功能说明

`TimeWeightedVectorStoreRetriever` 根据文档的新鲜度（最后访问时间）和语义相似度进行检索，平衡"新鲜"文档和"相关"文档的重要性。

## 核心特性

- **时间衰减**: 越 recent 访问的文档权重越高
- **语义相似度**: 同时考虑文档与查询的语义相关性
- **灵活配置**: 可调整衰减率控制时间影响力

## 关键组件

### TimeWeightedVectorStoreRetriever

```python
from langchain_classic.retrievers import TimeWeightedVectorStoreRetriever

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,
    k=3
)
```

## 关键参数

| 参数 | 说明 |
|------|------|
| vectorstore | 向量存储 |
| decay_rate | 衰减率，值越小早期文档权重下降越慢 |
| k | 返回的文档数量 |

## 运行示例

```bash
python 阶段3/09_retrieval_time_weighted.py
```

## 输出示例

```
=== TimeWeightedVectorStoreRetriever ===

查询 1: Python 编程语言
首次检索结果:
  1. Python 是一种高级编程语言...
  2. Python 3.12 版本引入了许多新特性...
  3. JavaScript 是 Web 的编程语言...

查询 2: JavaScript
第二次检索结果（部分文档已被访问）:
  1. JavaScript 是 Web 的编程语言...
  2. Python 是一种高级编程语言...
  3. TypeScript 是 JavaScript 的超集...

说明: decay_rate 控制时间衰减速度，值越小，早期访问的文档权重下降越慢。
```
