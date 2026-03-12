from langchain_core.prompts import HumanMessagePromptTemplate
"""
阶段3 - 01_memory_buffer.py
Memory 基础 - ChatMessageHistory + RunnableWithMessageHistory

LangChain 1.x 推荐使用 ChatMessageHistory 配合 RunnableWithMessageHistory 来实现对话记忆。
"""


import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = make_ollama()

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

store = {}

# 定义会话历史获取函数
def get_session_history(session_id: str):
    """获取会话历史，如果不存在则创建新的"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory() # 创建新的会话历史
    return store[session_id]

# 创建 RunnableWithMessageHistory 实例
conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== 对话 1 ===")
response1 = conversation.invoke(
    {"question": "你好，我叫张三，请记住我的名字"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你好，我叫张三，请记住我的名字")
print(f"助手: {response1.content}")

print("\n=== 对话 2 ===")
response2 = conversation.invoke(
    {"question": "我刚才告诉你我的名字是什么？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我刚才告诉你我的名字是什么？")
print(f"助手: {response2.content}")

print("\n=== 查看记忆内容 ===")
history = get_session_history("session_001")
print(f"记忆中的消息数量: {len(history.messages)}")
for i, msg in enumerate(history.messages, 1):
    print(f"消息 {i}: {type(msg).__name__} - {msg.content[:50]}...")

print("\n=== 对话 3 ===")
response3 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response3.content}")

print("\n=== 清除记忆后（新会话）===")
response4 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_002"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response4.content}")
