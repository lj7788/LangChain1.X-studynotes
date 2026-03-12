"""
阶段3 - 08_retrieval_ensemble.py
Retrieval - EnsembleRetriever 集成检索器

EnsembleRetriever 结合多个检索器的结果，通过加权融合的方式返回最终结果。
"""

from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()
embedding = HuggingFaceEmbeddings(model_name="dengcao/Dmeta-embedding-zh:F16")

documents = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        metadata={"source": "doc1", "topic": "python"}
    ),
    Document(
        page_content="Python 广泛应用于 Web 开发、数据科学和机器学习领域。",
        metadata={"source": "doc2", "topic": "python"}
    ),
    Document(
        page_content="Django 是 Python 的高级 Web 框架，鼓励快速开发和简洁实用的设计。",
        metadata={"source": "doc3", "topic": "django"}
    ),
    Document(
        page_content="Flask 是 Python 的轻量级 Web 框架，易于学习和使用。",
        metadata={"source": "doc4", "topic": "flask"}
    ),
    Document(
        page_content="NumPy 是 Python 的科学计算库，提供强大的数组和矩阵操作功能。",
        metadata={"source": "doc5", "topic": "numpy"}
    ),
    Document(
        page_content="Pandas 是 Python 的数据分析库，提供高效的数据结构和数据分析工具。",
        metadata={"source": "doc6", "topic": "pandas"}
    ),
]

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

chroma_vectorstore = Chroma.from_documents(texts, embedding=embedding)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

faiss_vectorstore = FAISS.from_documents(texts, embedding=embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

print("=== Chroma 检索器结果 ===\n")
query = "Python Web 框架有哪些？"
chroma_docs = chroma_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(chroma_docs, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")

print("\n=== FAISS 检索器结果 ===\n")
faiss_docs = faiss_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(faiss_docs, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")

print("\n=== EnsembleRetriever 集成检索结果 ===\n")
ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

ensemble_docs = ensemble_retriever.invoke(query)
print(f"查询: {query}\n")
print(f"返回文档数量: {len(ensemble_docs)}\n")
for i, doc in enumerate(ensemble_docs, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")

print("\n=== 不同权重示例 ===\n")
ensemble_retriever_weighted = EnsembleRetriever(
    retrievers=[chroma_retriever, faiss_retriever],
    weights=[0.7, 0.3]
)

ensemble_docs_weighted = ensemble_retriever_weighted.invoke(query)
print(f"查询: {query}")
print(f"权重 [0.7, 0.3]，返回 {len(ensemble_docs_weighted)} 个文档")
for i, doc in enumerate(ensemble_docs_weighted, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")
