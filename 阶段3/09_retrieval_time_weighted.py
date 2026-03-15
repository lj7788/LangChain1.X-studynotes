"""
阶段3 - 09_retrieval_time_weighted.py
Retrieval - TimeWeightedRetriever 时间加权检索器

TimeWeightedRetriever 根据文档的新鲜度（最后访问时间）和语义相似度进行检索，
平衡"新鲜"文档和"相关"文档的重要性。
"""

from langchain_classic.retrievers import TimeWeightedVectorStoreRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import sys
sys.path.append("../")
from tools import make_ollama

llm = make_ollama()
embedding = OllamaEmbeddings(model="dengcao/Dmeta-embedding-zh:F16")

documents = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        metadata={"source": "doc1", "topic": "python"}
    ),
    Document(
        page_content="Python 3.12 版本引入了许多新特性，包括更好的性能。",
        metadata={"source": "doc2", "topic": "python"}
    ),
    Document(
        page_content="JavaScript 是 Web 的编程语言，最初于 1995 年发布。",
        metadata={"source": "doc3", "topic": "javascript"}
    ),
    Document(
        page_content="TypeScript 是 JavaScript 的超集，添加了类型系统。",
        metadata={"source": "doc4", "topic": "typescript"}
    ),
    Document(
        page_content="Rust 是一种系统编程语言，强调安全性和并发性。",
        metadata={"source": "doc5", "topic": "rust"}
    ),
    Document(
        page_content="Go 是 Google 开发的编程语言，适合构建高效的服务器应用。",
        metadata={"source": "doc6", "topic": "go"}
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

query1 = "Python 编程语言"
print(f"查询 1: {query1}")
docs1 = retriever.invoke(query1)
print("首次检索结果:")
for i, doc in enumerate(docs1, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

query2 = "JavaScript"
print(f"\n查询 2: {query2}")
docs2 = retriever.invoke(query2)
print("第二次检索结果（部分文档已被访问）:")
for i, doc in enumerate(docs2, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

query3 = "编程语言"
print(f"\n查询 3: {query3}")
docs3 = retriever.invoke(query3)
print("第三次检索结果:")
for i, doc in enumerate(docs3, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

print("\n说明: decay_rate 控制时间衰减速度，值越小，早期访问的文档权重下降越慢。")
