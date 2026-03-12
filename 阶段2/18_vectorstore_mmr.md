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

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=api_base
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

## 代码解析

### MMR (Maximum Marginal Relevance)
- 最大边际相关性搜索
- 在相关性和多样性之间取得平衡
- `k`: 返回的结果数量
- `fetch_k`: 从数据库中获取的候选数量
- 避免返回过于相似的多个结果
