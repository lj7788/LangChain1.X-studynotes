"""
阶段3 - 15_memory_vectorstore.py
Memory - VectorStoreRetrieverMemory 向量存储记忆（LangChain 1.x）

VectorStoreRetrieverMemory 使用向量存储来保存和检索记忆，
支持大规模实体记忆的持久化和语义相似度检索。
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

llm = make_ollama()

embedding = HuggingFaceEmbeddings(model_name="dengcao/Dmeta-embedding-zh:F16")

initial_memory = [
    Document(
        page_content="用户叫张三，是一名软件工程师，喜欢 Python 编程。",
        metadata={"type": "user_info", "name": "张三"}
    ),
    Document(
        page_content="用户的公司叫科技未来有限公司，位于北京。",
        metadata={"type": "company_info"}
    ),
    Document(
        page_content="用户的项目经验：电商平台后端开发、RAG 知识库系统、智能客服机器人。",
        metadata={"type": "project_experience"}
    ),
]

vectorstore = Chroma.from_documents(
    documents=initial_memory,
    embedding=embedding,
    collection_name="user_memory"
)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(k=3),
    memory_key="context",
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个友好的助手。根据用户的背景信息回答问题。

用户背景：
{context}"""),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}")
])

conversation = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== VectorStoreRetrieverMemory 示例 ===\n")

print("查询 1：用户是做什么工作的？")
q1 = "用户是做什么工作的？"
r1 = conversation.invoke({"question": q1}, config={"configurable": {"session_id": "test"}})
print(f"问题: {q1}\n回答: {r1.content}\n")

print("查询 2：用户有什么项目经验？")
q2 = "用户有什么项目经验？"
r2 = conversation.invoke({"question": q2}, config={"configurable": {"session_id": "test"}})
print(f"问题: {q2}\n回答: {r2.content}\n")

print("查询 3：添加新记忆后再查询")
memory.save_context(
    {"input": "我最近在学习 LangGraph 和 Agent 开发"},
    {"output": "太棒了！这些是非常热门的技术方向。"}
)
q3 = "用户最近在学习什么？"
r3 = conversation.invoke({"question": q3}, config={"configurable": {"session_id": "test"}})
print(f"问题: {q3}\n回答: {r3.content}\n")

print("=== 查看向量存储中的记忆 ===")
docs = vectorstore.similarity_search("用户背景 项目", k=5)
for i, doc in enumerate(docs):
    print(f"{i+1}. {doc.page_content}")
    print(f"   元数据: {doc.metadata}")
