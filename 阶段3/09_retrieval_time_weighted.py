"""
阶段3 - 09_retrieval_time_weighted.py
Retrieval - TimeWeightedRetriever 时间加权检索器

TimeWeightedRetriever 根据文档的新鲜度（最后访问时间）和语义相似度进行检索，
平衡"新鲜"文档和"相关"文档的重要性。

核心概念：
- TimeWeightedVectorStoreRetriever: 时间加权向量检索器
- 时间衰减：文档的权重随时间衰减
- 新鲜度优先：最近访问的文档权重更高

工作流程：
1. 记录每个文档的最后访问时间
2. 计算文档的语义相似度分数
3. 根据时间衰减公式计算最终分数
4. 返回综合评分最高的文档

衰减公式：
score = similarity * e^(-decay_rate * time_elapsed)

参数说明：
- decay_rate: 衰减率，控制时间影响的强度
  - 值越大，时间影响越大
  - 值为0时，不考虑时间因素
- k: 返回的文档数量

优点：
- 平衡相关性和新鲜度
- 适合需要关注最新信息的场景
- 可以调整衰减率控制时间影响

缺点：
- 需要维护访问时间戳
- 可能过度偏向新文档
- 参数调整需要经验

使用场景：
- 新闻、资讯类检索
- 需要关注最新信息的知识库
- 时效性重要的应用
"""

from langchain_classic.retrievers import TimeWeightedVectorStoreRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import sys
sys.path.append("../")
from tools import make_ollama, make_embedding

# 初始化 Ollama LLM 和 embedding 模型
llm = make_ollama()
embedding = make_embedding()

# 创建示例文档集合（编程语言相关）
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

# 使用文本分割器将文档分割成小块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 创建向量数据库
vectorstore = Chroma.from_documents(texts, embedding=embedding)

print("=== TimeWeightedVectorStoreRetriever ===\n")

# 创建时间加权检索器
# decay_rate: 衰减率，值越小，早期访问的文档权重下降越慢
# k: 返回的文档数量
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,
    k=3
)

# 第一次查询：Python 编程语言
query1 = "Python 编程语言"
print(f"查询 1: {query1}")
docs1 = retriever.invoke(query1)
print("首次检索结果:")
for i, doc in enumerate(docs1, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

# 第二次查询：JavaScript
# 注意：这次查询会更新某些文档的访问时间
query2 = "JavaScript"
print(f"\n查询 2: {query2}")
docs2 = retriever.invoke(query2)
print("第二次检索结果（部分文档已被访问）:")
for i, doc in enumerate(docs2, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

# 第三次查询：编程语言
# 由于时间衰减，之前访问过的文档权重会降低
query3 = "编程语言"
print(f"\n查询 3: {query3}")
docs3 = retriever.invoke(query3)
print("第三次检索结果:")
for i, doc in enumerate(docs3, 1):
    print(f"  {i}. {doc.page_content[:50]}...")

print("\n说明: decay_rate 控制时间衰减速度，值越小，早期访问的文档权重下降越慢。")
