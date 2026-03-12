# 08_retrieval_ensemble.py

## 功能说明

`EnsembleRetriever` 结合多个检索器的结果，通过加权融合的方式返回最终结果，综合不同检索器的优势。

## 核心概念

### 为什么需要集成检索？

不同的检索器有不同的特点和优势：
- **Chroma**: 基于相似度检索，适合精确匹配
- **FAISS**: 高效的向量检索，适合大规模数据
- **BM25**: 基于关键词的传统检索，适合精确术语匹配
- **自定义检索器**: 可以结合外部知识源

集成检索可以：
- 提高检索的准确性和覆盖率
- 弥补单一检索器的不足
- 平衡精确度和召回率

### 加权融合

```python
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2, retriever3],
    weights=[0.5, 0.3, 0.2]  # 权重可以不同
)
```

权重决定了每个检索器结果的相对重要性。

## 融合算法

EnsembleRetriever 使用 **RRF (Reciprocal Rank Fusion)** 算法：

```
Score(d) = Σ (1 / (k + rank_i(d))) * weight_i
```

其中：
- `rank_i(d)` 是文档 d 在第 i 个检索结果中的排名
- `k` 是常数（通常为 60）
- `weight_i` 是第 i 个检索器的权重

## 运行示例

```bash
python 阶段3/08_retrieval_ensemble.py
```

## 输出示例

```
=== Chroma 检索器结果 ===

查询: Python Web 框架有哪些？

文档 1: Django 是 Python 的高级 Web 框架，鼓励快速开发和简洁实用的设计。 (来源: doc3)
文档 2: Flask 是 Python 的轻量级 Web 框架，易于学习和使用。 (来源: doc4)

=== FAISS 检索器结果 ===

查询: Python Web 框架有哪些？

文档 1: Django 是 Python 的高级 Web 框架，鼓励快速开发和简洁实用的设计。 (来源: doc3)
文档 2: Flask 是 Python 的轻量级 Web 框架，易于学习和使用。 (来源: doc4)

=== EnsembleRetriever 集成检索结果 ===

查询: Python Web 框架有哪些？
返回文档数量: 3

文档 1: Django 是 Python 的高级 Web 框架，鼓励快速开发和简洁实用的设计。 (来源: doc3)
文档 2: Flask 是 Python 的轻量级 Web 框架，易于学习和使用。 (来源: doc4)
文档 3: Python 广泛应用于 Web 开发、数据科学和机器学习领域。 (来源: doc2)
```
