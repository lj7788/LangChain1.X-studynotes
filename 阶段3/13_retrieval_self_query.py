"""
阶段3 - 13_retrieval_self_query.py
Retrieval - SelfQueryRetriever 自查询检索器

SelfQueryRetriever 使用 LLM 将自然语言查询转换为结构化查询条件，
适用于带有元数据的文档检索场景。

核心概念：
- SelfQueryRetriever: 自查询检索器
- 结构化查询：将自然语言转换为过滤条件
- 元数据过滤：基于文档元数据进行精确筛选

工作流程：
1. 用户输入自然语言查询
2. LLM 分析查询，提取语义部分和过滤条件
3. 语义部分用于向量检索
4. 过滤条件用于元数据筛选
5. 返回同时满足语义和过滤条件的文档

元数据字段定义：
- source: 编程语言的名称或来源
- language: 编程语言使用的自然语言
- year: 编程语言发布的年份

优点：
- 支持复杂的查询条件
- 结合语义检索和精确过滤
- 用户可以使用自然语言查询

缺点：
- 依赖 LLM 的理解能力
- 需要定义清晰的元数据字段
- 可能产生错误的过滤条件

使用场景：
- 带有丰富元数据的文档库
- 需要精确筛选的检索场景
- 用户习惯自然语言查询
"""

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import sys
sys.path.append("../")
from tools import make_ollama, make_embedding

# 初始化 Ollama LLM 和 embedding 模型
llm = make_ollama()
embedding = make_embedding()

# 创建示例文档集合（编程语言相关，带元数据）
documents = [
    Document(
        page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
        metadata={"source": "python", "language": "en", "year": 1991}
    ),
    Document(
        page_content="JavaScript 是由 Brendan Eich 于 1995 年创建的网页脚本语言。",
        metadata={"source": "javascript", "language": "en", "year": 1995}
    ),
    Document(
        page_content="Java 是由 Sun Microsystems 于 1995 年发布的面向对象编程语言。",
        metadata={"source": "java", "language": "en", "year": 1995}
    ),
    Document(
        page_content="Go 是由 Google 于 2009 年发布的现代编程语言。",
        metadata={"source": "go", "language": "en", "year": 2009}
    ),
    Document(
        page_content="Rust 是由 Mozilla 于 2010 年发布的系统编程语言，强调安全性。",
        metadata={"source": "rust", "language": "en", "year": 2010}
    ),
    Document(
        page_content="TypeScript 是 Microsoft 于 2012 年发布的 JavaScript 超集。",
        metadata={"source": "typescript", "language": "en", "year": 2012}
    ),
]

# 使用文本分割器将文档分割成小块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 创建向量数据库并初始化检索器
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# 定义元数据字段信息
# 用于告诉 LLM 文档有哪些元数据字段及其含义
field_info = [
    AttributeInfo(
        name="source",
        description="编程语言的名称或来源",
        type="string"
    ),
    AttributeInfo(
        name="language",
        description="编程语言使用的自然语言",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="编程语言发布的年份",
        type="integer"
    ),
]

# 定义查询示例
# 用于帮助 LLM 理解如何构建查询
examples = [
    ("2010年之前发布的语言", {"query": "2010年之前发布的语言", "filter": 'lt("year", 2010)'}),
    ("Python 或 JavaScript", {"query": "Python 或 JavaScript", "filter": 'or(eq("source", "python"), eq("source", "javascript"))'}),
]

# 创建自查询检索器
self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_contents="编程语言的描述和元数据",
    metadata_field_info=field_info,
    chain_kwargs={"examples": examples}
)

print("=== SelfQueryRetriever 示例 ===\n")

# 查询 1：2010年之前发布的语言
print("查询 1：2010年之前发布的语言")
results1 = self_query_retriever.invoke("2010年之前发布的语言")
for i, doc in enumerate(results1):
    print(f"  文档 {i+1}: {doc.page_content[:40]}... | 元数据: {doc.metadata}")

# 查询 2：Python 或 JavaScript
print("\n查询 2：Python 或 JavaScript")
results2 = self_query_retriever.invoke("Python 或 JavaScript")
for i, doc in enumerate(results2):
    print(f"  文档 {i+1}: {doc.page_content[:40]}... | 元数据: {doc.metadata}")
