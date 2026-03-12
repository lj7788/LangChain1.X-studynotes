# MMR 搜索

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/18_vectorstore_mmr.py` 中的代码。

---

## 完整代码

```python
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("=== MMR 搜索 (最大边际相关性) ===")

from langchain_community.vectorstores import FAISS

texts = [
    "今天天气很好，阳光明媚",
    "天气晴朗，适合外出活动",
    "我喜欢吃苹果和香蕉",
    "苹果是一种营养丰富的水果",
    "Python 是一种编程语言",
    "Java 也是一种流行的编程语言",
    "天气变化很快，要随时关注天气预报",
    "水果富含维生素，对身体有益"
]

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings
)

query = "天气怎么样"

print(f"\n查询: '{query}'")
print("\n--- 标准相似度搜索 (k=3) ---")
results = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(results):
    print(f"  {i+1}. {doc.page_content}")

print("\n--- MMR 搜索 (k=3, fetch_k=5) ---")
mmr_results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=5)
for i, doc in enumerate(mmr_results):
    print(f"  {i+1}. {doc.page_content}")

print("\n说明: MMR 搜索会在相关性和多样性之间取得平衡")
```

---

## MMR (Maximum Marginal Relevance)

- 最大边际相关性搜索
- 在**相关性**和**多样性**之间取得平衡
- 避免返回过于相似的多个结果

### 参数说明

| 参数 | 含义 |
|------|------|
| `k` | 最终返回的结果数量（多样性筛选后的） |
| `fetch_k` | 初始检索的数量（多样性筛选前的） |

### 工作流程

```
查询: "天气怎么样"

Step 1: fetch_k=5
先从数据库检索出 5 个最相似的：
  1. 天气变化很快，要随时关注天气预报  ← 相似度最高
  2. 天气晴朗，适合外出活动
  3. 今天天气很好，阳光明媚
  4. 天气...
  5. ...

Step 2: k=3
从这 5 个中筛选出 3 个最多样的：
  1. 天气变化很快，要随时关注天气预报
  2. 天气晴朗，适合外出活动
  3. 我喜欢吃苹果和香蕉    ← 跳过了相似的，选了不同的
```

---

## MMR vs 普通搜索

| 搜索方式 | 特点 | 场景 |
|---------|------|------|
| **普通相似度搜索** | 只看相似度，可能返回重复/冗余的结果 | 简单场景 |
| **MMR 搜索** | 同时考虑相似度+多样性，返回结果更丰富 | 需要多样化结果的场景 |

### 结果对比

```
查询: '天气怎么样'

--- 标准相似度搜索 (k=3) ---
  1. 天气变化很快，要随时关注天气预报
  2. 天气晴朗，适合外出活动
  3. 今天天气很好，阳光明媚
     ↑ 都是关于天气的，有重复

--- MMR 搜索 (k=3, fetch_k=5) ---
  1. 天气变化很快，要随时关注天气预报
  2. 天气晴朗，适合外出活动
  3. 我喜欢吃苹果和香蕉    ← 换话题了！
     ↑ 增加了多样性
```

---

## 实际项目用哪种？

**90% 用普通相似度搜索**

| 场景 | 推荐方式 |
|------|---------|
| **RAG 问答** | 普通搜索 ✅ |
| **文档问答** | 普通搜索 ✅ |
| **代码问答** | 普通搜索 ✅ |
| **多样化推荐** | MMR |

### 为什么普通搜索更常用？

1. **RAG 场景**：需要精确答案，不需要多样性
2. **简单高效**：计算量小，速度快
3. **LLM 有上下文理解**：即使相似结果有少量重复，LLM 也能处理

### 什么时候用 MMR？

- 推荐系统（需要多样化）
- 创意生成（需要不同角度）
- 探索性搜索（不想只看单一答案）
