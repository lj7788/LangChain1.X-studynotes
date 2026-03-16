"""
阶段3 - 08_retrieval_ensemble.py
Retrieval - EnsembleRetriever 集成检索器

EnsembleRetriever 结合多个检索器的结果，通过加权融合的方式返回最终结果。

核心概念：
- EnsembleRetriever: 集成检索器
- 多检索器融合：结合不同检索器的优势
- 加权融合：根据权重分配不同检索器的重要性

工作流程：
1. 创建多个不同的检索器（如 Chroma、FAISS）
2. 设置每个检索器的权重
3. 对查询进行检索，获取所有检索器的结果
4. 根据权重合并和排序结果
5. 返回最终的检索结果

优点：
- 结合不同检索器的优势
- 提高检索的准确性和鲁棒性
- 可以根据需求调整权重

缺点：
- 增加检索时间（需要多次检索）
- 需要维护多个向量数据库
- 权重设置需要经验

使用场景：
- 需要高准确性的检索
- 不同检索器各有优势的场景
- 需要平衡不同检索策略
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_ollama import OllamaEmbeddings
import sys
sys.path.append("../")
from tools import make_ollama, make_embedding

# 初始化 Ollama LLM 和 embedding 模型
llm = make_ollama()
embedding = make_embedding()    

# 创建示例文档集合（Python相关主题）
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

# 使用文本分割器将文档分割成小块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 创建 Chroma 向量数据库和检索器
chroma_vectorstore = Chroma.from_documents(texts, embedding=embedding)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建 FAISS 向量数据库和检索器
faiss_vectorstore = FAISS.from_documents(texts, embedding=embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建集成检索器
# retrievers: 检索器列表
# weights: 每个检索器的权重，总和为1
ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

print("=== EnsembleRetriever 集成检索器 ===\n")

# 测试查询
query = "Python Web 框架有哪些？"
print(f"查询: {query}\n")

# 使用 Chroma 检索器检索
print("--- Chroma 检索结果 ---")
chroma_docs = chroma_retriever.invoke(query)
for i, doc in enumerate(chroma_docs, 1):
    print(f"文档 {i}: {doc.page_content}")

# 使用 FAISS 检索器检索
print("\n--- FAISS 检索结果 ---")
faiss_docs = faiss_retriever.invoke(query)
for i, doc in enumerate(faiss_docs, 1):
    print(f"文档 {i}: {doc.page_content}")

# 使用集成检索器检索
print("\n--- Ensemble 检索结果 ---")
ensemble_docs = ensemble_retriever.invoke(query)
for i, doc in enumerate(ensemble_docs, 1):
    print(f"文档 {i}: {doc.page_content}")
    print(f"来源: {doc.metadata['source']}\n")
