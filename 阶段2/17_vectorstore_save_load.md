# 向量数据库 - 保存与加载

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/17_vectorstore_save_load.py` 中的代码。

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

print("=== FAISS 向量数据库 - 保存与加载 ===")

from langchain_community.vectorstores import FAISS

texts = [
    "今天天气很好，阳光明媚",
    "天气晴朗，适合外出活动",
    "Python 是一种编程语言"
]

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings
)

save_path = Path(__file__).parent / "faiss_index"
vectorstore.save_local(str(save_path))

print(f"向量数据库已保存到: {save_path}")

loaded_vectorstore = FAISS.load_local(
    str(save_path),
    embeddings,
    allow_dangerous_deserialization=True
)

print("向量数据库加载成功")

query = "天气怎么样"
results = loaded_vectorstore.similarity_search(query, k=2)
print(f"\n查询: '{query}'")
print("相似结果:")
for doc in results:
    print(f"  - {doc.page_content}")
```

---

## 代码解析

### 保存与加载
- `save_local`: 保存向量数据库到本地
- `load_local`: 从本地加载向量数据库
- `allow_dangerous_deserialization`: 允许反序列化（仅在可信来源时使用）
- 可以持久化向量数据库，避免重复计算
