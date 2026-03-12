# 10_retrieval_custom.py

## 功能说明

展示如何创建自定义 Retriever，实现特定场景的检索逻辑。

## 自定义 Retriever 基础

要创建自定义 Retriever，需要继承 `BaseRetriever` 并实现 `_get_relevant_documents` 方法：

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.schema import Document

class MyRetriever(BaseRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # 实现检索逻辑
        return documents
```

## 示例：关键词过滤 Retriever

```python
class KeywordFilterRetriever(BaseRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        keywords = query.split()
        
        filtered_docs = []
        for doc in self.documents:
            content = doc.page_content.lower()
            if any(kw.lower() in content for kw in keywords):
                filtered_docs.append(doc)
        
        return filtered_docs[:self.k]
```

## 示例：混合 Retriever

结合向量检索和关键词过滤的优点：

```python
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, keyword_retriever, k: int = 3):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.k = k
    
    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)
        
        # 合并去重
        seen = set()
        combined = []
        for doc in keyword_docs + vector_docs:
            doc_id = doc.page_content[:50]
            if doc_id not in seen:
                seen.add(doc_id)
                combined.append(doc)
        
        return combined[:self.k]
```

## 运行示例

```bash
python 阶段3/10_retrieval_custom.py
```

## 输出示例

```
=== 基础向量检索 ===

查询: Python 编程语言

文档 1: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。 (来源: doc1)
文档 2: Python 广泛应用于 Web 开发、数据科学和机器学习领域。 (来源: doc2)
文档 3: JavaScript 是 Web 的编程语言，最初于 1995 年发布。 (来源: doc3)

=== 关键词过滤检索 ===

查询: Python 编程语言

文档 1: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。 (来源: doc1)
文档 2: Python 广泛应用于 Web 开发、数据科学和机器学习领域。 (来源: doc2)

=== 混合检索（向量 + 关键词）===

查询: Python 编程语言

文档 1: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。 (来源: doc1)
文档 2: Python 广泛应用于 Web 开发、数据科学和机器学习领域。 (来源: doc2)
```
