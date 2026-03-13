"""
阶段3 - 05_memory_lcel.py
Memory - 在LCEL链中使用Memory（LangChain 0.3+ 新版 API）

展示如何在 LCEL (LangChain Expression Language) 中使用 Memory。
使用 ChatMessageHistory + RunnableWithMessageHistory。
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = make_ollama()

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== LCEL + Memory 对话示例 ===\n")

print("--- 对话 1 ---")
response1 = conversation.invoke(
    {"question": "你好，我叫李明"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你好，我叫李明")
print(f"助手: {response1.content}")

print("\n--- 对话 2 ---")
response2 = conversation.invoke(
    {"question": "我刚才告诉你我的名字是什么？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我刚才告诉你我的名字是什么？")
print(f"助手: {response2.content}")

print("\n--- 对话 3 ---")
response3 = conversation.invoke(
    {"question": "你知道我喜欢什么吗？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你知道我喜欢什么吗？")
print(f"助手: {response3.content}")

print("\n=== 另一个会话（新用户）===\n")

print("--- 会话 A ---")
responseA = conversation.invoke(
    {"question": "我叫王芳，是一名教师"},
    config={"configurable": {"session_id": "session_002"}}
)
print(f"用户: 我叫王芳，是一名教师")
print(f"助手: {responseA.content}")

print("\n--- 会话 B（同一会话）---")
responseB = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_002"}}
)
print(f"用户: 我是谁？")
print(f"助手: {responseB.content}")

print("\n--- 会话 C（新会话，无历史）---")
responseC = conversation.invoke(
    {"question": "还记得我叫什么吗？"},
    config={"configurable": {"session_id": "session_003"}}
)
print(f"用户: 还记得我叫什么吗？")
print(f"助手: {responseC.content}")

print("\n=== 查看所有会话历史 ===")
for session_id, history in store.items():
    print(f"\n会话 {session_id}:")
    for msg in history.messages:
        role = "用户" if msg.type == "human" else "助手"
        print(f"  {role}: {msg.content[:50]}...")
