"""
阶段3 - 02_memory_summary.py
Memory - ConversationSummaryMemory 对话摘要

ConversationSummaryMemory 使用 LLM 生成对话摘要，适合长对话场景。
"""

from langchain_community.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()

memory = ConversationSummaryMemory(
    llm=llm,
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

print("=== 对话 1 ===")
question1 = "我叫张三，是一名软件工程师。我喜欢编程和读书。"
response1 = chain.invoke(
    {"question": question1}, 
    config={"memory": memory}
)
print(f"用户: {question1}")
print(f"助手: {response1.content}")

print("\n=== 对话 2 ===")
question2 = "你喜欢什么运动？我喜欢打篮球。"
response2 = chain.invoke(
    {"question": question2}, 
    config={"memory": memory}
)
print(f"用户: {question2}")
print(f"助手: {response2.content}")

print("\n=== 查看摘要记忆 ===")
print(f"记忆内容:\n{memory.buffer}")

print("\n=== 对话 3 ===")
question3 = "总结一下我告诉你的关于我自己的信息"
response3 = chain.invoke(
    {"question": question3}, 
    config={"memory": memory}
)
print(f"用户: {question3}")
print(f"助手: {response3.content}")

print("\n=== 查看更新后的摘要 ===")
print(f"记忆内容:\n{memory.buffer}")

print("\n=== 对话 4 ===")
question4 = "我是谁？喜欢什么？"
response4 = chain.invoke(
    {"question": question4}, 
    config={"memory": memory}
)
print(f"用户: {question4}")
print(f"助手: {response4.content}")
