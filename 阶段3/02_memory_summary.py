"""
阶段3 - 02_memory_summary.py
Memory - ConversationSummaryMemory 对话摘要（LangChain 1.x）

ConversationSummaryMemory 使用 LLM 自动总结对话历史，节省 token 消耗。
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = make_ollama()

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
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

print("=== 对话 1 ===")
q1 = "我叫张三，是一名软件工程师。我喜欢编程和读书。"
r1 = conversation.invoke(
    {"question": q1},
    config={"configurable": {"session_id": "test"}}
)
print(f"用户: {q1}\n助手: {r1.content}")

print("\n=== 查看摘要记忆 ===")
print(f"记忆内容:\n{memory.load_memory_variables({})['chat_history']}\n")

print("=== 对话 2 ===")
q2 = "你喜欢什么运动？我喜欢打篮球。"
r2 = conversation.invoke(
    {"question": q2},
    config={"configurable": {"session_id": "test"}}
)
print(f"用户: {q2}\n助手: {r2.content}")

print("\n=== 查看更新后的摘要 ===")
print(f"记忆内容:\n{memory.load_memory_variables({})['chat_history']}\n")

print("=== 对话 3 ===")
q3 = "总结一下我告诉你的关于我自己的信息"
r3 = conversation.invoke(
    {"question": q3},
    config={"configurable": {"session_id": "test"}}
)
print(f"用户: {q3}\n助手: {r3.content}")
