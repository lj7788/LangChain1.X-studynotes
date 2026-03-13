"""
阶段3 - 16_memory_entity_kg.py
Memory - 手动实体关系提取（替代 KnowledgeGraphMemory）

由于 KnowledgeGraphMemory 需要外部图数据库（如 Neo4j）支持，
本示例展示如何使用 ChatMessageHistory + LLM 手动实现实体关系提取和管理。
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import json

llm = make_ollama()

chat_history = ChatMessageHistory()

entities = {}
relationships = []

EXTRACT_PROMPT = ChatPromptTemplate.from_template("""请从以下对话中提取实体和关系。

对话：
{conversation}

请以 JSON 格式返回，格式如下：
{{
    "entities": [
        {{"name": "实体名", "type": "类型", "description": "描述"}}
    ],
    "relationships": [
        {{"from": "实体1", "relation": "关系", "to": "实体2"}}
    ]
}}

只返回 JSON，不要其他内容：""")

QUERY_PROMPT = ChatPromptTemplate.from_template("""根据以下实体和关系信息回答用户的问题。

实体信息：
{entities}

关系信息：
{relationships}

用户问题：{question}

请根据以上信息回答问题。如果信息不足，请说明无法回答。""")

def extract_entities_and_relations():
    """从对话历史中提取实体和关系"""
    if not chat_history.messages:
        return
    
    conversation = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in chat_history.messages
    ])
    
    response = EXTRACT_PROMPT | llm | StrOutputParser()
    try:
        result = json.loads(response.invoke({"conversation": conversation}))
        return result.get("entities", []), result.get("relationships", [])
    except:
        return [], []

def add_message(user_input: str, ai_output: str):
    """添加对话消息并提取实体关系"""
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(ai_output)
    
    new_entities, new_relations = extract_entities_and_relations()
    
    global entities, relationships
    for e in new_entities:
        if e["name"] not in entities:
            entities[e["name"]] = e
    
    for r in new_relations:
        if r not in relationships:
            relationships.append(r)

def query_with_entities(question: str):
    """基于实体关系回答问题"""
    entities_str = "\n".join([
        f"- {name}: {info['type']} - {info['description']}"
        for name, info in entities.items()
    ])
    relations_str = "\n".join([
        f"- {r['from']} --[{r['relation']}]--> {r['to']}"
        for r in relationships
    ])
    
    response = QUERY_PROMPT | llm | StrOutputParser()
    return response.invoke({
        "entities": entities_str,
        "relationships": relations_str,
        "question": question
    })

print("=== 手动实体关系提取示例（替代 KnowledgeGraphMemory）===\n")

print("对话 1：用户自我介绍")
add_message(
    "我叫李明，是一名软件工程师，在北京工作。我朋友王芳是一名设计师，在上海工作。",
    "很高兴认识你，李明！我是你的 AI 助手。"
)
print(f"当前实体: {list(entities.keys())}")
print(f"当前关系: {relationships}")

print("\n对话 2：添加更多关系")
add_message(
    "王芳是我的妻子，她是一名设计师。",
    "了解，你们是一对设计师和工程师的组合！"
)
print(f"当前实体: {list(entities.keys())}")
print(f"当前关系: {relationships}")

print("\n对话 3：查询关系")
answer = query_with_entities("李明和王芳是什么关系？")
print(f"问题: 李明和王芳是什么关系？")
print(f"回答: {answer}")

print("\n对话 4：查询工作地点")
answer = query_with_entities("李明在哪里工作？")
print(f"问题: 李明在哪里工作？")
print(f"回答: {answer}")

print("\n=== 完整实体信息 ===")
print("实体:")
for name, info in entities.items():
    print(f"  - {name}: {info}")

print("\n关系:")
for r in relationships:
    print(f"  - {r['from']} --[{r['relation']}]--> {r['to']}")
