"""
阶段3 - 07_retrieval_multi_query.py
Retrieval - MultiQueryRetriever 多查询检索器

MultiQueryRetriever 使用 LLM 为每个查询生成多个变体，
然后从所有变体的检索结果中合并去重，提高检索的召回率。

核心概念：
- MultiQueryRetriever: 多查询检索器
- 查询变体：使用 LLM 生成多个相似但不同的查询
- 合并去重：合并所有查询的结果，去除重复文档

工作流程：
1. 用户输入原始查询
2. LLM 生成多个查询变体（通常3-5个）
3. 对每个查询进行检索
4. 合并所有结果并去重
5. 返回最相关的文档

优点：
- 提高召回率，找到更多相关文档
- 解决查询表述不准确的问题
- 适合语义相近但表述不同的查询

缺点：
- 增加检索时间（需要多次检索）
- 可能返回一些不太相关的文档
- 需要额外的 LLM 调用

使用场景：
- 查询表述不确定
- 需要高召回率的场景
- 专业知识库检索
"""

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings 
import sys
sys.path.append("../")
from tools import make_ollama,make_embedding

# 初始化 Ollama LLM 和 embedding 模型
llm = make_ollama()
embedding = make_embedding()

# 创建示例文档集合（AI相关主题）
documents = [
    Document(
        page_content="""深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。
深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。
常见的深度学习框架包括 TensorFlow、PyTorch 和 Keras。""",
        metadata={"source": "deep_learning", "topic": "AI"}
    ),
    Document(
        page_content="""机器学习是人工智能的一个分支，专注于开发能够从数据中学习的算法。
机器学习有三种主要类型：监督学习、无监督学习和强化学习。
监督学习需要标注数据，无监督学习从无标注数据中发现模式。""",
        metadata={"source": "machine_learning", "topic": "AI"}
    ),
    Document(
        page_content="""Python 是一种广泛用于数据科学和机器学习的编程语言。
Python 拥有丰富的库支持，如 NumPy、Pandas、Scikit-learn 等。
Python 的简洁语法使其成为学习和实现机器学习算法的理想选择。""",
        metadata={"source": "python_ml", "topic": "编程"}
    ),
    Document(
        page_content="""神经网络是受生物神经系统启发的计算模型。神经网络由多层神经元组成，
包括输入层、隐藏层和输出层。深度学习通过增加网络层数来提高模型的表达能力。
卷积神经网络（CNN）主要用于图像处理，循环神经网络（RNN）主要用于序列数据。""",
        metadata={"source": "neural_network", "topic": "AI"}
    ),
    Document(
        page_content="""自然语言处理（NLP）是人工智能和语言学的交叉领域。
NLP 涉及文本分析、文本生成、机器翻译、情感分析等任务。
近年来，基于 Transformer 的预训练语言模型（如 BERT、GPT）大幅提升了 NLP 任务的性能。""",
        metadata={"source": "nlp", "topic": "AI"}
    ),
]

# 使用文本分割器将文档分割成小块
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 创建向量数据库并初始化基础检索器
vectorstore = Chroma.from_documents(texts, embedding=embedding)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("=== MultiQueryRetriever 多查询检索器 ===\n")

# 创建多查询检索器
# from_llm: 使用 LLM 生成查询变体
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# 测试查询
query = "深度学习在哪些领域有应用？"
print(f"原始查询: {query}\n")

# 使用基础检索器检索
print("--- 基础检索结果 ---")
basic_docs = base_retriever.invoke(query)
for i, doc in enumerate(basic_docs, 1):
    print(f"文档 {i}: {doc.page_content[:60]}...")
    print(f"来源: {doc.metadata['source']}\n")

# 使用多查询检索器检索
print("--- MultiQuery 检索结果 ---")
unique_docs = retriever.invoke(query)
for i, doc in enumerate(unique_docs, 1):
    print(f"文档 {i}: {doc.page_content[:60]}...")
    print(f"来源: {doc.metadata['source']}\n")
