"""
阶段3 - 15_memory_vectorstore.py
Memory - VectorStoreRetrieverMemory 向量存储记忆（LangChain 1.x）

VectorStoreRetrieverMemory 使用向量存储来保存和检索记忆，
支持大规模实体记忆的持久化和语义相似度检索。

注意：在 LangChain 1.x 中，VectorStoreRetrieverMemory 需要手动管理，
不能直接与 RunnableWithMessageHistory 一起使用。

核心概念：
- 向量存储：使用向量数据库存储记忆
- 语义检索：基于语义相似度检索相关记忆
- 持久化：记忆可以持久化到磁盘

工作流程：
1. 将记忆转换为文档并存储到向量数据库
2. 根据查询检索相关的记忆
3. 将检索到的记忆作为上下文传递给 LLM
4. LLM 基于记忆上下文回答问题

记忆类型：
- 用户信息：姓名、职业、喜好等
- 公司信息：公司名称、地点等
- 项目经验：参与的项目、技术栈等

优点：
- 支持大规模记忆存储
- 语义检索，智能匹配
- 记忆可持久化
- 适合长期记忆场景

缺点：
- 需要手动管理记忆的添加
- 不能直接与 RunnableWithMessageHistory 集成
- 需要额外的向量数据库

使用场景：
- 需要长期记忆的应用
- 大规模实体记忆
- 需要语义检索的场景
"""

import sys
sys.path.append("../")
from tools import make_ollama, make_embedding
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory

# 初始化 Ollama LLM 模型
llm = make_ollama()

# 初始化 embedding 模型
embedding = make_embedding()

# 初始记忆文档
# 这些是预先定义的用户信息
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

# 创建向量数据库并初始化记忆
# collection_name: 集合名称，用于区分不同的记忆库
vectorstore = Chroma.from_documents(
    documents=initial_memory,
    embedding=embedding,
    collection_name="user_memory"
)

# 创建向量存储记忆
# retriever: 检索器，用于检索相关记忆
# memory_key: 记忆的键名
# return_messages: 是否返回消息对象
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(k=3),
    memory_key="context",
    return_messages=True
)

# 使用字典存储不同会话的历史记录
store = {}

def get_session_history(session_id: str):
    """
    获取会话历史，如果不存在则创建新的

    参数:
        session_id: 会话唯一标识符

    返回:
        ChatMessageHistory: 包含该会话所有历史消息的对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_relevant_context(question: str):
    """
    从向量存储中检索相关的上下文

    参数:
        question: 用户问题

    返回:
        list: 相关的记忆文档列表
    """
    # 使用记忆的 load_memory_variables 方法检索相关上下文
    relevant_docs = memory.load_memory_variables({"question": question})
    return relevant_docs.get("context", [])

# 定义提示模板
# 包含用户背景信息和对话历史
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个友好的助手。根据用户的背景信息回答问题。

用户背景：
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 创建对话链
conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== VectorStoreRetrieverMemory 示例 ===\n")

print("查询 1：用户是做什么工作的？")
# 查询 1：测试用户信息检索
q1 = "用户是做什么工作的？"
context1 = get_relevant_context(q1)
r1 = conversation.invoke(
    {"question": q1, "context": "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in context1])}, 
    config={"configurable": {"session_id": "test"}}
)
print(f"问题: {q1}\n回答: {r1.content}\n")

print("查询 2：用户有什么项目经验？")
# 查询 2：测试项目经验检索
q2 = "用户有什么项目经验？"
context2 = get_relevant_context(q2)
r2 = conversation.invoke(
    {"question": q2, "context": "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in context2])}, 
    config={"configurable": {"session_id": "test"}}
)
print(f"问题: {q2}\n回答: {r2.content}\n")

print("查询 3：添加新记忆后再查询")
# 查询 3：添加新记忆并测试
memory.save_context(
    {"input": "我最近在学习 LangGraph 和 Agent 开发"},
    {"output": "太棒了！这些是非常热门的技术方向。"}
)
q3 = "用户最近在学习什么？"
context3 = get_relevant_context(q3)
r3 = conversation.invoke(
    {"question": q3, "context": "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in context3])}, 
    config={"configurable": {"session_id": "test"}}
)
print(f"问题: {q3}\n回答: {r3.content}\n")

print("=== 查看向量存储中的记忆 ===")
# 查看向量存储中的所有记忆
docs = vectorstore.similarity_search("用户背景 项目", k=5)
for i, doc in enumerate(docs):
    print(f"{i+1}. {doc.page_content}")
    print(f"   元数据: {doc.metadata}")
