"""
阶段3 - 12_memory_token_buffer.py
Memory - ConversationTokenBufferMemory Token计数记忆（LangChain 1.x）

ConversationTokenBufferMemory 根据 Token 数量来管理对话历史，
当超过指定 token 数时自动清理旧消息。
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = make_ollama()

memory = ConversationTokenBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=100
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{question}")
])

conversation = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

print("=== 对话 1：发送较长消息 ===")
q1 = "我叫张三，我是一名软件工程师，已经工作了5年。我擅长 Python 编程和后端开发。我之前在一家互联网公司负责架构设计和系统优化工作。"
r1 = conversation.invoke({"question": q1}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q1[:50]}...\n助手: {r1.content}")
print(f"当前 token 数: {memory.token_count}")

print("\n=== 对话 2：继续发送消息 ===")
q2 = "我最近在学习大模型应用开发，包括 RAG、Agent 等技术方向。希望能够将传统软件开发经验与 AI 技术结合。"
r2 = conversation.invoke({"question": q2}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q2[:50]}...\n助手: {r2.content}")
print(f"当前 token 数: {memory.token_count}")

print("\n=== 对话 3：检查是否超过限制 ===")
q3 = "今天天气真不错！"
r3 = conversation.invoke({"question": q3}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q3}\n助手: {r3.content}")
print(f"当前 token 数: {memory.token_count}")

print("\n=== 对话 4：继续对话 ===")
q4 = "你喜欢编程吗？"
r4 = conversation.invoke({"question": q4}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q4}\n助手: {r4.content}")
print(f"当前 token 数: {memory.token_count}")

print("\n=== 查看保留的记忆 ===")
history = memory.load_memory_variables({})["chat_history"]
print(f"保留消息数: {len(history)}")
for i, msg in enumerate(history):
    print(f"  {i+1}. {type(msg).__name__}: {msg.content[:40]}...")
