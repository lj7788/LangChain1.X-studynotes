"""
阶段3 - 03_memory_entity.py
Memory - EntityMemory 实体记忆

EntityMemory 自动从对话中提取和记忆实体信息（如人物、地点、组织等）。
"""

from langchain_community.memory import EntityMemory
from langchain_core.prompts import PromptTemplate
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()

memory = EntityMemory(
    llm=llm,
    memory_key="entities",
    return_messages=True
)

prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据已知的实体信息回答用户的问题。

已知实体:
{entities}

用户问题: {question}

回答:"""
)

chain = prompt | llm

print("=== 对话 1：提到人物 ===")
question1 = "我的朋友李明在北京工作，他是一名医生。"
response1 = chain.invoke(
    {"question": question1}, 
    config={"memory": memory}
)
print(f"用户: {question1}")
print(f"助手: {response1.content}")

print("\n=== 查看实体记忆 ===")
print(f"实体信息: {memory.buffer}")

print("\n=== 对话 2：提到更多人物 ===")
question2 = "李明的妻子叫王芳，她是一名老师。他们住在上海。"
response2 = chain.invoke(
    {"question": question2}, 
    config={"memory": memory}
)
print(f"用户: {question2}")
print(f"助手: {response2.content}")

print("\n=== 查看更新后的实体 ===")
print(f"实体信息: {memory.buffer}")

print("\n=== 对话 3：询问实体信息 ===")
question3 = "告诉我你知道的关于李明的所有信息"
response3 = chain.invoke(
    {"question": question3}, 
    config={"memory": memory}
)
print(f"用户: {question3}")
print(f"助手: {response3.content}")

print("\n=== 对话 4：询问另一个实体 ===")
question4 = "王芳是做什么工作的？"
response4 = chain.invoke(
    {"question": question4}, 
    config={"memory": memory}
)
print(f"用户: {question4}")
print(f"助手: {response4.content}")
