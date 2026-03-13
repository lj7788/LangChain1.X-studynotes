# 07_retrieval_multi_query.py

## 功能说明

`MultiQueryRetriever` 使用 LLM 为每个查询生成多个变体，然后从所有变体的检索结果中合并去重，提高检索的召回率。

## 核心特性

- **查询扩展**: 自动生成多个语义相似的查询
- **召回率提升**: 从不同角度检索，增加找到相关文档的概率
- **去重合并**: 自动合并重复的检索结果

## 关键组件

### MultiQueryRetriever

```python
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)
```

## 关键参数

| 参数 | 说明 |
|------|------|
| retriever | 基础检索器 |
| llm | 用于生成查询变体的 LLM |

## 运行示例

```bash
python 阶段3/07_retrieval_multi_query.py
```

## 输出示例

```
=== MultiQueryRetriever 多查询检索器 ===

原始查询: 深度学习在哪些领域有应用？

--- 基础检索结果 ---
文档 1: 深度学习是机器学习的一个分支，它使用多层神经网络...
来源: deep_learning

--- MultiQuery 检索结果 ---
文档 1: 深度学习是机器学习的一个分支，它使用多层神经网络...
来源: deep_learning

文档 2: Python 是一种广泛用于数据科学和机器学习的编程语言...
来源: python_ml
```
