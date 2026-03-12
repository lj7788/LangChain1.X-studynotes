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
