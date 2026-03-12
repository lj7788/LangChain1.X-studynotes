"""
阶段3 - 06_retrieval_compression.py
Retrieval - ContextualCompression 上下文压缩检索器

ContextualCompressionRetriever 可以压缩检索到的文档，只保留与查询相关的部分，
减少输入给 LLM 的上下文长度。
"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()

embedding = HuggingFaceEmbeddings(model_name="dengcao/Dmeta-embedding-zh:F16")

documents = [
    Document(
        page_content="""Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。
Python 设计哲学强调代码的可读性，其语法允许程序员用更少的代码表达想法。
Python 支持多种编程范式，包括结构化、过程式、反射式、面向对象和函数式编程。""",
        metadata={"source": "python_intro", "topic": "编程语言"}
    ),
    Document(
        page_content="""JavaScript 是 Web 的编程语言。JavaScript 可以更新和更改 HTML 和 CSS。
JavaScript 可以计算、操作和验证数据。JavaScript 最初于 1995 年发布，用于为 Netscape Navigator 提供脚本支持。
现在，JavaScript 被所有现代 Web 浏览器支持。""",
        metadata={"source": "javascript_intro", "topic": "Web开发"}
    ),
    Document(
        page_content="""Python 的应用领域非常广泛。在 Web 开发方面，Django 和 Flask 是流行的框架。
在数据科学领域，NumPy、Pandas 和 Matplotlib 是重要的库。Python 还广泛用于机器学习和人工智能，
TensorFlow、PyTorch 和 Scikit-learn 是常用的 ML 框架。""",
        metadata={"source": "python_applications", "topic": "编程语言"}
    ),
    Document(
        page_content="""JavaScript 不仅可以用于前端开发，还可以通过 Node.js 用于后端开发。
Node.js 是基于 Chrome V8 引擎的 JavaScript 运行时。Express.js 是 Node.js 上最流行的 Web 框架。
JavaScript 生态系统非常丰富，有大量的 npm 包可供使用。""",
        metadata={"source": "javascript_applications", "topic": "Web开发"}
    ),
    Document(
        page_content="""学习编程需要大量的实践。建议从简单的项目开始，逐步增加难度。
不要害怕犯错误，每个错误都是学习的机会。参与开源项目也是提高编程技能的好方法。
此外，阅读优秀的代码可以帮助你学习新的编程技术和最佳实践。""",
        metadata={"source": "learning_programming", "topic": "学习方法"}
    ),
]

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(texts, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("=== 基础检索（未压缩）===\n")
query = "Python 在数据科学和机器学习中的应用"
docs = retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(docs, 1):
    print(f"--- 文档 {i} ---")
    print(f"内容: {doc.page_content[:100]}...")
    print(f"来源: {doc.metadata['source']}\n")

print("\n=== 使用 ContextualCompression 压缩检索 ===\n")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(compressed_docs, 1):
    print(f"--- 压缩后的文档 {i} ---")
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata['source']}\n")
