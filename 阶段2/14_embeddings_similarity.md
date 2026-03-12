# Embeddings 相似度计算

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/14_embeddings_similarity.py` 中的代码。

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
import numpy as np

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=api_base
)

text1 = "今天天气很好"
text2 = "天气晴朗，适合外出"
text3 = "苹果是一种水果"
text4 = "我喜欢吃苹果"

texts = [text1, text2, text3, text4]

print("=== Embeddings 相似度计算 ===")

embedding_results = embeddings.embed_documents(texts)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n计算文本之间的余弦相似度:")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(embedding_results[i], embedding_results[j])
        print(f"\n文本{i+1} vs 文本{j+1}: {sim:.4f}")
        print(f"  '{texts[i]}' vs '{texts[j]}'")

query = "今天适合出去玩吗"
query_embedding = embeddings.embed_query(query)

print(f"\n\n查询: '{query}'")
print("\n与各文本的相似度:")
for i, (text, emb) in enumerate(zip(texts, embedding_results)):
    sim = cosine_similarity(query_embedding, emb)
    print(f"  {text}: {sim:.4f}")
```

---

## 代码解析

### 余弦相似度
- 用于衡量两个向量之间的相似程度
- 值范围：-1 到 1，1 表示完全相似
- 在文本 embedding 中常用
- 计算公式：`cos(θ) = (A·B) / (||A|| × ||B||)`
