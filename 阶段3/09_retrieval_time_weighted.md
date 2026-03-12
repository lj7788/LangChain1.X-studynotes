# 09_retrieval_time_weighted.py

## 功能说明

`TimeWeightedVectorStoreRetriever` 根据文档的新鲜度（最后访问时间）和语义相似度进行检索，平衡"新鲜"文档和"相关"文档的重要性。

## 核心概念

### 为什么需要时间加权检索？

在很多应用场景中，我们希望：
- 优先展示最近访问/更新的文档
- 同时保留语义相似度的考量
- 让用户更容易找到"最新"的相关信息

### 评分公式

```
Score = (1 - decay_rate) ^ hours_since_access * semantic_score
```

- **decay_rate**: 衰减率，控制新鲜度权重下降的速度
- **hours_since_access**: 距离上次访问的小时数
- **semantic_score**: 向量相似度分数

## 关键参数

| 参数 | 说明 |
|------|------|
| vectorstore | 向量数据库 |
| search_kwargs | 检索参数（如 k） |
| decay_rate | 衰减率（0-1），越大越偏向新文档 |
| k | 返回的文档数量 |

## 使用场景

- 新闻资讯检索
- 文档版本管理
- 推荐系统
- 用户历史浏览记录

## 运行示例

```bash
python 阶段3/09_retrieval_time_weighted.py
```

## 输出示例

```
=== TimeWeightedVectorStoreRetriever ===

查询: Python 编程语言

首次检索结果（按新鲜度加权）:
文档 1: Python 3.12 版本引入了许多新特性，包括更好的性能。
   来源: doc2, 最后访问: 2024-12-01

文档 2: Go 是 Google 开发的编程语言，适合构建高效的服务器应用。
   来源: doc6, 最后访问: 2024-10-01

文档 3: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
   来源: doc1, 最后访问: 2024-01-01

=== 模拟用户访问文档 ===
用户访问了: Python 3.12 版本引入了许多新特性
用户访问了: TypeScript 是 JavaScript 的超集

=== 再次检索相同查询 ===
查询: Python 编程语言

调整后的检索结果（刚才访问的文档排名提升）:
文档 1: Python 3.12 版本引入了许多新特性，包括更好的性能。
   来源: doc2, 最后访问: 2024-12-01

文档 2: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
   来源: doc1, 最后访问: 2024-01-01
```
