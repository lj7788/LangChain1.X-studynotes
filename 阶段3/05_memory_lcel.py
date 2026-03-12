"""
阶段3 - 05_memory_lcel.py
Memory - 在LCEL链中使用Memory

展示如何在 LCEL (LangChain Expression Language) 中使用 Memory。
"""

from langchain_community.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

用户问题: {question}

回答:"""
)

chain = prompt | llm

print("=== LCEL + Memory 对话示例 ===\n")

print("--- 对话 1 ---")
question1 = "你好，我叫李明"
inputs1 = {"question": question1}
output1 = chain.invoke(inputs1, config={"memory": memory})
print(f"用户: {question1}")
print(f"助手: {output1.content}")

print("\n--- 对话 2 ---")
question2 = "我刚才告诉你我的名字是什么？"
inputs2 = {"question": question2}
output2 = chain.invoke(inputs2, config={"memory": memory})
print(f"用户: {question2}")
print(f"助手: {output2.content}")

print("\n--- 对话 3 ---")
question3 = "你知道我喜欢什么吗？"
inputs3 = {"question": question3}
output3 = chain.invoke(inputs3, config={"memory": memory})
print(f"用户: {question3}")
print(f"助手: {output3.content}")

print("\n=== 使用 RunnableWithMessageHistory ===")
print("使用 RunnableWithMessageHistory 实现更方便的记忆管理\n")

from langchain.runnables.history import RunnableWithMessageHistory

chat_prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

当前问题: {question}

回答:"""
)

chat_chain = chat_prompt | llm

chat_with_history = RunnableWithMessageHistory(
    chat_chain,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

print("--- 会话 1 ---")
response1 = chat_with_history.invoke(
    {"question": "我叫王芳，是一名教师"},
    config={"configurable": {"session_id": "user_001"}}
)
print(f"用户: 我叫王芳，是一名教师")
print(f"助手: {response1.content}")

print("\n--- 会话 2（同一会话）---")
response2 = chat_with_history.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "user_001"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response2.content}")

print("\n--- 会话 3（新会话，无历史）---")
response3 = chat_with_history.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "user_002"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response3.content}")
