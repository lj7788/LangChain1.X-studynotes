"""
阶段3 - 07_retrieval_multi_query.py
Retrieval - MultiQueryRetriever 多查询检索器

MultiQueryRetriever 使用 LLM 为每个查询生成多个变体，
然后从所有变体的检索结果中合并去重，提高检索的召回率。
"""

from langchain.retrievers.multi_query import MultiQueryRetriever
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
    Document(
        page_content="""PyTorch 是一个开源的深度学习框架，由 Facebook 开发。
PyTorch 以其动态计算图和易用性而受到研究者欢迎。
PyTorch 提供了丰富的神经网络模块和优化算法。""",
        metadata={"source": "pytorch", "topic": "深度学习框架"}
    ),
]

text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(texts, embedding=embedding)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("=== 基础检索 ===\n")
query = "深度学习框架有哪些？"
docs = base_retriever.invoke(query)
print(f"查询: {query}\n")
for i, doc in enumerate(docs, 1):
    print(f"文档 {i}: {doc.page_content[:60]}... (来源: {doc.metadata['source']})")

print("\n=== MultiQueryRetriever 检索 ===\n")
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

unique_docs = multi_query_retriever.invoke(query)
print(f"查询: {query}\n")
print(f"返回的文档数量: {len(unique_docs)}\n")
for i, doc in enumerate(unique_docs, 1):
    print(f"文档 {i}: {doc.page_content[:60]}... (来源: {doc.metadata['source']})")

print("\n=== 查看生成的查询变体 ===")
print("MultiQueryRetriever 会为原始查询生成多个变体，")
print("例如: '深度学习框架有哪些？' -> ['深度学习框架有哪些？', '常用的深度学习框架是什么？', '主流的深度学习库有哪些？']")
print("这样可以从不同角度检索，增加找到相关文档的可能性。")
