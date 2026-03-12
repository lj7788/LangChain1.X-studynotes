# OpenAI Embeddings

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/12_embeddings_openai.py` 中的代码。

---

## 完整代码

```python
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from 阶段1.tools import make_model

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

texts = [
    "今天天气很好",
    "天气晴朗，适合外出",
    "我喜炊吃苹果",
    "苹果是一种水果"
]

print("=== OpenAI Embeddings ===")
print(f"嵌入维度: {embeddings.embedding_dim}")

results = embeddings.embed_documents(texts)
for i, (text, embedding) in enumerate(zip(texts, results)):
    print(f"\n文本 {i+1}: {text}")
    print(f"向量长度: {len(embedding)}")
    print(f"向量前5位: {embedding[:5]}")

query = "今天适合出去玩吗"
query_embedding = embeddings.embed_query(query)
print(f"\n查询: {query}")
print(f"查询向量长度: {len(query_embedding)}")
print(f"查询向量前5位: {query_embedding[:5]}")
```

---

## 代码解析

### OpenAIEmbeddings
- 使用 OpenAI 的 embedding 模型生成文本向量
- `model`: 指定使用的 embedding 模型
- `embed_documents`: 批量生成文档的 embedding
- `embed_query`: 为查询文本生成 embedding
- 返回的向量可以用于相似度计算
