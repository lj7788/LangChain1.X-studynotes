import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings



env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)




embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("=== 完整 RAG 流程示例 ===")

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

sample_file = data_dir / "knowledge.txt"
sample_file.write_text("""LangChain 是一个用于构建 LLM 应用的框架。

它提供了丰富的组件，包括：
- Model I/O: 与各种 LLM 模型交互
- Chains: 构建多步骤的工作流
- Agents: 让 LLM 自主决策
- Memory: 为对话添加记忆功能
- Retrieval: 增强 LLM 的知识库

使用 LangChain，你可以快速构建：
- 问答系统
- 聊天机器人
- 文档摘要
- 代码助手等应用
""", encoding="utf-8")

print("\n1. 加载文档...")
loader = TextLoader(str(sample_file), encoding="utf-8")
documents = loader.load()
print(f"   加载了 {len(documents)} 个文档")

print("\n2. 分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
splits = text_splitter.split_documents(documents)
print(f"   分割为 {len(splits)} 个文本块")

print("\n3. 创建向量数据库...")
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)
print(f"   向量数据库创建成功，包含 {vectorstore.index.ntotal} 个文档")

print("\n4. 相似度搜索...")
query = "LangChain 可以做什么？"
results = vectorstore.similarity_search(query, k=2)
print(f"   查询: '{query}'")
print("   搜索结果:")
for i, doc in enumerate(results):
    print(f"   {i+1}. {doc.page_content}")

print("\n5. 基于检索的问答...")
from tools import make_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""基于以下上下文回答问题。

上下文:
{context}

问题: {question}

回答:""")

model = make_model()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke(query)
print(f"\n问题: {query}")
print(f"回答: {answer}")
