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

print("=== FAISS 向量数据库 ===")

from langchain_community.vectorstores import FAISS

texts = [
    "今天天气很好，阳光明媚",
    "天气晴朗，适合外出活动",
    "我喜欢吃苹果和香蕉",
    "苹果是一种营养丰富的水果",
    "Python 是一种编程语言",
    "Java 也是一种流行的编程语言"
]

print(f"加载 {len(texts)} 个文本到向量数据库...")

vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings
)

print(f"向量数据库创建成功")

query = "今天适合出去玩吗"
print(f"\n查询: '{query}'")

results = vectorstore.similarity_search(query, k=2)
print(f"\nTop 2 相似结果:")
for i, doc in enumerate(results):
    print(f"\n--- 结果 {i+1} ---")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")

scores = vectorstore.similarity_search_with_score(query, k=2)
print(f"\n带分数的相似度结果:")
for doc, score in scores:
    print(f"  分数: {score:.4f}, 内容: {doc.page_content}")
