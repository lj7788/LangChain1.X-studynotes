import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
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
