# 07_retrieval_multi_query.py

## 功能说明

`MultiQueryRetriever` 使用 LLM 为每个查询生成多个变体，然后从所有变体的检索结果中合并去重，提高检索的召回率。

## 核心概念

### 为什么需要多查询检索？

1. **查询表达多样性**: 用户用不同的方式表达同一个查询
2. **语义匹配差异**: 不同的查询可能有不同的语义匹配
3. **检索盲区**: 某些关键词可能不在相关文档中

### 工作原理

```
原始查询: "深度学习框架有哪些？"
          ↓
    LLM 生成变体
          ↓
查询1: "深度学习框架有哪些？"
查询2: "常用的深度学习框架是什么？"
查询3: "主流的深度学习库有哪些？"
          ↓
    分别检索文档
          ↓
    合并去重
          ↓
返回最终结果
```

## 关键参数

| 参数 | 说明 |
|------|------|
| retriever | 基础检索器 |
| llm | 用于生成查询变体的 LLM |
| prompt | 自定义生成查询的提示词 |

## 自定义提示词

```python
from langchain.prompts import PromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一个AI助手。你的任务是对于给定的用户问题，生成3个不同的版本。
    目的是为了从向量数据库中检索相关文档。通过生成多个版本的查询，可以增加找到相关文档的可能性。
    
    原问题: {question}
    
    生成3个不同的查询版本，用换行分隔:"""
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=QUERY_PROMPT
)
```

## 运行示例

```bash
python 阶段3/07_retrieval_multi_query.py
```

## 输出示例

```
=== 基础检索 ===

查询: 深度学习框架有哪些？

文档 1: 深度学习是机器学习的一个分支，它使用多层神经网络来学习... (来源: deep_learning)
文档 2: PyTorch 是一个开源的深度学习框架，由 Facebook 开发... (来源: pytorch)

=== MultiQueryRetriever 检索 ===

查询: 深度学习框架有哪些？
返回的文档数量: 3

文档 1: 深度学习是机器学习的一个分支，它使用多层神经网络来学习... (来源: deep_learning)
文档 2: PyTorch 是一个开源的深度学习框架，由 Facebook 开发... (来源: pytorch)
文档 3: 常见的深度学习框架包括 TensorFlow、PyTorch 和 Keras... (来源: deep_learning)

=== 查看生成的查询变体 ===
MultiQueryRetriever 会为原始查询生成多个变体，
例如: '深度学习框架有哪些？' -> ['深度学习框架有哪些？', '常用的深度学习框架是什么？', '主流的深度学习库有哪些？']
```
