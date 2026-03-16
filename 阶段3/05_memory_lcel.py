"""
阶段3 - 05_memory_lcel.py
Memory - 在LCEL链中使用Memory（LangChain 0.3+ 新版 API）

展示如何在 LCEL (LangChain Expression Language) 中使用 Memory。
使用 ChatMessageHistory + RunnableWithMessageHistory。

核心概念：
- LCEL (LangChain Expression Language): LangChain 的表达式语言，用于构建链式操作
- RunnableWithMessageHistory: 包装 LCEL 链，自动管理对话历史
- 会话隔离：不同的 session_id 对应不同的对话历史

工作流程：
1. 创建 LCEL 链：prompt | llm
2. 用 RunnableWithMessageHistory 包装链
3. 通过 session_id 区分不同会话
4. 自动管理每个会话的对话历史

LCEL 优势：
- 声明式语法，代码简洁
- 易于组合和扩展
- 支持异步操作
- 类型安全

使用场景：
- 需要在链式操作中集成对话记忆
- 多用户、多会话的场景
- 需要灵活组合不同组件的应用
"""

import sys
sys.path.append("../")
from tools import make_ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 初始化 Ollama LLM 模型
llm = make_ollama()

# 使用字典存储不同会话的历史记录
store = {}


def get_session_history(session_id: str):
    """
    获取会话历史，如果不存在则创建新的

    参数:
        session_id: 会话唯一标识符，用于区分不同用户或对话

    返回:
        ChatMessageHistory: 包含该会话所有历史消息的对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 定义提示模板
# SystemMessage: 设置系统角色和行为
# MessagesPlaceholder: 动态插入对话历史
# HumanMessage: 用户的问题
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

# 创建 RunnableWithMessageHistory 实例
# prompt | llm: LCEL 链式操作，先格式化提示，再调用 LLM
# get_session_history: 获取会话历史的函数
# input_messages_key: 输入消息的键名
# history_messages_key: 历史消息的键名
conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== LCEL + Memory 对话示例 ===\n")

print("--- 对话 1 ---")
# 第一轮对话：用户介绍自己
# 使用 session_001 作为会话 ID
response1 = conversation.invoke(
    {"question": "你好，我叫李明"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你好，我叫李明")
print(f"助手: {response1.content}")

print("\n--- 对话 2 ---")
# 第二轮对话：测试记忆功能
# 使用相同的 session_id，可以访问之前的对话历史
response2 = conversation.invoke(
    {"question": "我刚才告诉你我的名字是什么？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我刚才告诉你我的名字是什么？")
print(f"助手: {response2.content}")

print("\n--- 对话 3 ---")
# 第三轮对话：继续测试
response3 = conversation.invoke(
    {"question": "你知道我喜欢什么吗？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你知道我喜欢什么吗？")
print(f"助手: {response3.content}")

print("\n=== 另一个会话（新用户）===\n")

print("--- 会话 A ---")
# 创建新会话：session_002
# 这个会话与 session_001 完全独立
responseA = conversation.invoke(
    {"question": "我叫王芳，是一名教师"},
    config={"configurable": {"session_id": "session_002"}}
)
print(f"用户: 我叫王芳，是一名教师")
print(f"助手: {responseA.content}")

print("\n--- 会话 B（同一会话）---")
# 继续使用 session_002
responseB = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_002"}}
)
print(f"用户: 我是谁？")
print(f"助手: {responseB.content}")

print("\n--- 会话 C（新会话，无历史）===")
# 创建第三个会话：session_003
# 这个会话没有历史记录
responseC = conversation.invoke(
    {"question": "还记得我叫什么吗？"},
    config={"configurable": {"session_id": "session_003"}}
)
print(f"用户: 还记得我叫什么吗？")
print(f"助手: {responseC.content}")

print("\n=== 查看所有会话历史 ===")
# 遍历所有会话，打印每个会话的历史消息
for session_id, history in store.items():
    print(f"\n会话 {session_id}:")
    for msg in history.messages:
        role = "用户" if msg.type == "human" else "助手"
        print(f"  {role}: {msg.content[:50]}...")
