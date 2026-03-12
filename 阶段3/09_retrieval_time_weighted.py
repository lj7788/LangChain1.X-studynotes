"""
阶段3 - 09_retrieval_time_weighted.py
Retrieval - TimeWeightedRetriever 时间加权检索器

TimeWeightedRetriever 根据文档的新鲜度（最后访问时间）和语义相似度进行检索，
平衡"新鲜"文档和"相关"文档的重要性。
"""

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime, timedelta
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()
embedding = HuggingFaceEmbeddings(model_name="dengcao/Dmeta-embedding-zh:F16")

documents = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        metadata={"source": "doc1", "topic": "python", "last_accessed": "2024-01-01"}
    ),
    Document(
        page_content="Python 3.12 版本引入了许多新特性，包括更好的性能。",
        metadata={"source": "doc2", "topic": "python", "last_accessed": "2024-12-01"}
    ),
    Document(
        page_content="JavaScript 是 Web 的编程语言，最初于 1995 年发布。",
        metadata={"source": "doc3", "topic": "javascript", "last_accessed": "2024-06-01"}
    ),
    Document(
        page_content="TypeScript 是 JavaScript 的超集，添加了类型系统。",
        metadata={"source": "doc4", "topic": "typescript", "last_accessed": "2024-11-15"}
    ),
    Document(
        page_content="Rust 是一种系统编程语言，强调安全性和并发性。",
        metadata={"source": "doc5", "topic": "rust", "last_accessed": "2024-03-01"}
    ),
    Document(
        page_content="Go 是 Google 开发的编程语言，适合构建高效的服务器应用。",
        metadata={"source": "doc6", "topic": "go", "last_accessed": "2024-10-01"}
    ),
]

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(texts, embedding=embedding)

print("=== TimeWeightedVectorStoreRetriever ===\n")

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,
    k=3
)

query = "Python 编程语言"
print(f"查询: {query}\n")

results = retriever.invoke(query)
print(f"首次检索结果（按新鲜度加权）:")
for i, doc in enumerate(results, 1):
    last_accessed = doc.metadata.get("last_accessed", "unknown")
    print(f"文档 {i}: {doc.page_content}")
    print(f"   来源: {doc.metadata['source']}, 最后访问: {last_accessed}\n")

print("\n=== 模拟用户访问文档 ===")
retriever.invoke("Python 3.12 的特性")
print("用户访问了: Python 3.12 版本引入了许多新特性")

retriever.invoke("TypeScript 类型系统")
print("用户访问了: TypeScript 是 JavaScript 的超集\n")

print("=== 再次检索相同查询 ===")
results2 = retriever.invoke(query)
print(f"查询: {query}\n")
print(f"调整后的检索结果（刚才访问的文档排名提升）:")
for i, doc in enumerate(results2, 1):
    last_accessed = doc.metadata.get("last_accessed", "unknown")
    print(f"文档 {i}: {doc.page_content}")
    print(f"   来源: {doc.metadata['source']}, 最后访问: {last_accessed}\n")

print("=== 关键参数说明 ===")
print("- decay_rate: 衰减率，决定新鲜度权重下降的速度")
print("  - 值越大，新文档的权重越高")
print("  - 值为 0 时，只考虑语义相似度")
print("- k: 返回的文档数量")
print("- optional_score_threshold: 可选的分数阈值")
