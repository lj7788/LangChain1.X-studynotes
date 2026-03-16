"""
阶段3 - 16_memory_entity_kg.py
Memory - 手动实体关系提取（替代 KnowledgeGraphMemory）

由于 KnowledgeGraphMemory 需要外部图数据库（如 Neo4j）支持，
本示例展示如何使用 ChatMessageHistory + LLM 手动实现实体关系提取和管理。

核心概念：
- 实体：对话中提到的重要对象（如人名、公司名等）
- 关系：实体之间的连接（如"妻子"、"同事"等）
- 知识图谱：实体和关系的网络结构
- 手动提取：使用 LLM 从对话中提取实体和关系

工作流程：
1. 用户输入对话
2. LLM 从对话中提取实体和关系
3. 将实体和关系存储到内存
4. 查询时，基于实体和关系回答问题

数据结构：
- entities: 字典，存储实体信息
  - key: 实体名称
  - value: 实体详细信息（类型、描述）
- relationships: 列表，存储关系信息
  - from: 起始实体
  - relation: 关系类型
  - to: 目标实体

优点：
- 不需要外部图数据库
- 灵活控制实体和关系的提取
- 可以自定义提取逻辑

缺点：
- 需要手动管理实体和关系
- 依赖 LLM 的提取能力
- 不支持复杂的图查询

使用场景：
- 需要跟踪实体关系的对话
- 不想使用外部图数据库
- 简单的实体关系管理
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

# 初始化 Ollama LLM 模型
llm = make_ollama()

# 创建对话历史对象
chat_history = ChatMessageHistory()

# 实体和关系存储
entities = {}
relationships = []

# 定义实体提取提示模板
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

# 定义查询提示模板
QUERY_PROMPT = ChatPromptTemplate.from_template("""根据以下实体和关系信息回答用户的问题。

实体信息：
{entities}

关系信息：
{relationships}

用户问题：{question}

请根据以上信息回答问题。如果信息不足，请说明无法回答。""")

def extract_entities_and_relations():
    """
    从对话历史中提取实体和关系

    返回:
        tuple: (实体列表, 关系列表)
    """
    # 如果没有对话历史，返回空列表
    if not chat_history.messages:
        return [], []
    
    # 将对话历史格式化为文本
    conversation = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in chat_history.messages
    ])
    
    # 调用 LLM 提取实体和关系
    response = EXTRACT_PROMPT | llm | StrOutputParser()
    try:
        result = json.loads(response.invoke({"conversation": conversation}))
        return result.get("entities", []), result.get("relationships", [])
    except:
        return [], []

def add_message(user_input: str, ai_output: str):
    """
    添加对话消息并提取实体关系

    参数:
        user_input: 用户输入
        ai_output: AI 输出
    """
    # 添加消息到对话历史
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(ai_output)
    
    # 提取新的实体和关系
    new_entities, new_relations = extract_entities_and_relations()
    
    # 更新全局实体和关系
    global entities, relationships
    for e in new_entities:
        if e["name"] not in entities:
            entities[e["name"]] = e
    
    for r in new_relations:
        if r not in relationships:
            relationships.append(r)

def query_with_entities(question: str):
    """
    基于实体关系回答问题

    参数:
        question: 用户问题

    返回:
        str: AI 的回答
    """
    # 格式化实体信息
    entities_str = "\n".join([
        f"- {name}: {info['type']} - {info['description']}"
        for name, info in entities.items()
    ])
    # 格式化关系信息
    relations_str = "\n".join([
        f"- {r['from']} --[{r['relation']}]--> {r['to']}"
        for r in relationships
    ])
    
    # 调用 LLM 回答问题
    response = QUERY_PROMPT | llm | StrOutputParser()
    return response.invoke({
        "entities": entities_str,
        "relationships": relations_str,
        "question": question
    })

print("=== 手动实体关系提取示例（替代 KnowledgeGraphMemory）===\n")

print("对话 1：用户自我介绍")
# 第一轮对话：用户介绍自己和朋友
add_message(
    "我叫李明，是一名软件工程师，在北京工作。我朋友王芳是一名设计师，在上海工作。",
    "很高兴认识你，李明！我是你的 AI 助手。"
)
print(f"当前实体: {list(entities.keys())}")
print(f"当前关系: {relationships}")

print("\n对话 2：添加更多关系")
# 第二轮对话：添加更多关系
add_message(
    "王芳是我的妻子，她是一名设计师。",
    "了解，你们是一对设计师和工程师的组合！"
)
print(f"当前实体: {list(entities.keys())}")
print(f"当前关系: {relationships}")

print("\n对话 3：查询关系")
# 第三轮对话：查询实体关系
answer = query_with_entities("李明和王芳是什么关系？")
print(f"问题: 李明和王芳是什么关系？")
print(f"回答: {answer}")

print("\n对话 4：查询工作地点")
# 第四轮对话：查询工作地点
answer = query_with_entities("李明在哪里工作？")
print(f"问题: 李明在哪里工作？")
print(f"回答: {answer}")

print("\n=== 完整实体信息 ===")
# 打印所有实体信息
print("实体:")
for name, info in entities.items():
    print(f"  - {name}: {info}")

# 打印所有关系信息
print("\n关系:")
for r in relationships:
    print(f"  - {r['from']} --[{r['relation']}]--> {r['to']}")
