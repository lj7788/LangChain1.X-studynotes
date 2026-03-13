# 08_retrieval_ensemble.py

## 功能说明

`EnsembleRetriever` 结合多个检索器的结果，通过加权融合的方式返回最终结果。

## 核心特性

- **多检索器组合**: 结合向量检索、关键词检索等多种检索方式
- **权重配置**: 可以为不同检索器设置不同权重
- **结果融合**: 自动合并和排序不同检索器的结果

## 关键组件

### EnsembleRetriever

```python
from langchain_classic.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)
```

## 关键参数

| 参数 | 说明 |
|------|------|
| retrievers | 基础检索器列表 |
| weights | 对应检索器的权重 |

## 运行示例

```bash
python 阶段3/08_retrieval_ensemble.py
```

## 输出示例

```
=== EnsembleRetriever 集成检索器 ===

查询: Python Web 框架有哪些？

--- Chroma 检索结果 ---
文档 1: Python 是一种高级编程语言...
文档 2: Python 广泛应用于 Web 开发...

--- FAISS 检索结果 ---
文档 1: Python 是一种高级编程语言...
文档 2: Django 是 Python 的高级 Web 框架...

--- Ensemble 检索结果 ---
文档 1: Python 是一种高级编程语言...
来源: doc1

文档 2: Python 广泛应用于 Web 开发...
来源: doc2

文档 3: Django 是 Python 的高级 Web 框架...
来源: doc3
```
