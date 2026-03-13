"""
阶段3 - 11_memory_buffer_window.py
Memory - ConversationBufferWindowMemory 窗口记忆（LangChain 1.x）

ConversationBufferWindowMemory 只保留最近 k 轮对话，避免记忆无限增长。
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = make_ollama()

memory = ConversationBufferWindowMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    k=2
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

print("=== 对话 1：第1轮 ===")
q1 = "我叫张三，喜欢编程。"
r1 = conversation.invoke({"question": q1}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q1}\n助手: {r1.content}")

print("\n=== 对话 2：第2轮 ===")
q2 = "我喜欢 Python 语言。"
r2 = conversation.invoke({"question": q2}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q2}\n助手: {r2.content}")

print("\n=== 对话 3：第3轮（窗口k=2，只会保留最近2轮）===")
q3 = "我今天天气很好。"
r3 = conversation.invoke({"question": q3}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q3}\n助手: {r3.content}")

print("\n=== 查看记忆（应该只保留最近2轮）===")
history = memory.load_memory_variables({})["chat_history"]
print(f"记忆轮数: {len(history)}")
for i, msg in enumerate(history):
    print(f"  {i+1}. {type(msg).__name__}: {msg.content[:30]}...")

print("\n=== 对话 4：第4轮 ===")
q4 = "你还记得我叫什么吗？"
r4 = conversation.invoke({"question": q4}, config={"configurable": {"session_id": "test"}})
print(f"用户: {q4}\n助手: {r4.content}")
