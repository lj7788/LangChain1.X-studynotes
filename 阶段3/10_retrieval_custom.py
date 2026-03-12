"""
阶段3 - 10_retrieval_custom.py
Retrieval - 自定义 Retriever

展示如何创建自定义 Retriever，实现特定场景的检索逻辑。
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()
embedding = HuggingFaceEmbeddings(model_name="dengcao/Dmeta-embedding-zh:F16")


class KeywordFilterRetriever(BaseRetriever):
    """关键词过滤 Retriever：只返回包含指定关键词的文档"""
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        keywords = query.split()
        
        filtered_docs = []
        for doc in self.documents:
            content = doc.page_content.lower()
            if any(kw.lower() in content for kw in keywords):
                filtered_docs.append(doc)
        
        return filtered_docs[:self.k]
    
    def __init__(self, documents: list[Document], k: int = 3):
        super().__init__()
        self.documents = documents
        self.k = k


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
        page_content="JavaScript 是 Web 的编程语言，最初于 1995 年发布。",
        metadata={"source": "doc3", "topic": "javascript"}
    ),
    Document(
        page_content="TypeScript 是 JavaScript 的超集，添加了类型系统。",
        metadata={"source": "doc4", "topic": "typescript"}
    ),
    Document(
        page_content="Go 是 Google 开发的编程语言，适合构建高效的服务器应用。",
        metadata={"source": "doc5", "topic": "go"}
    ),
]

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(texts, embedding=embedding)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

keyword_retriever = KeywordFilterRetriever(documents=texts, k=3)


class HybridRetriever(BaseRetriever):
    """混合 Retriever：结合向量检索和关键词过滤"""
    
    def __init__(self, vector_retriever, keyword_retriever, k: int = 3):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.k = k
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        keyword_docs = self.keyword_retriever.invoke(query)
        
        seen = set()
        combined = []
        
        for doc in keyword_docs:
            doc_id = doc.page_content[:50]
            if doc_id not in seen:
                seen.add(doc_id)
                combined.append(doc)
        
        for doc in vector_docs:
            doc_id = doc.page_content[:50]
            if doc_id not in seen:
                seen.add(doc_id)
                combined.append(doc)
        
        return combined[:self.k]


hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever, k=3)

print("=== 基础向量检索 ===\n")
query = "Python 编程语言"
results = vector_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")

print("\n=== 关键词过滤检索 ===\n")
results = keyword_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")

print("\n=== 混合检索（向量 + 关键词）===\n")
results = hybrid_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")

print("\n=== 另一种查询 ===\n")
query2 = "Web 开发"
results = hybrid_retriever.invoke(query2)
print(f"查询: {query2}\n")
for i, doc in enumerate(results, 1):
    print(f"文档 {i}: {doc.page_content} (来源: {doc.metadata['source']})")
