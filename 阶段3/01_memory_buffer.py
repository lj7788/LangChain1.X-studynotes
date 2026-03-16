from langchain_core.prompts import HumanMessagePromptTemplate

"""
阶段3 - 01_memory_buffer.py
Memory 基础 - ChatMessageHistory + RunnableWithMessageHistory

LangChain 1.x 推荐使用 ChatMessageHistory 配合 RunnableWithMessageHistory 来实现对话记忆。

核心概念：
- ChatMessageHistory: 用于存储对话历史消息的内存存储
- RunnableWithMessageHistory: 包装 LLM 链，自动管理对话历史
- session_id: 用于区分不同会话的唯一标识符

使用场景：
- 需要记住用户在对话中提到的信息
- 多轮对话中保持上下文连贯性
- 支持多个独立的对话会话
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

# 定义提示模板
# SystemMessage: 设置系统角色和行为
# MessagesPlaceholder: 动态插入对话历史
# HumanMessage: 用户的问题
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

# 使用字典存储不同会话的历史记录
# key: session_id, value: ChatMessageHistory 对象
store = {}

# 定义会话历史获取函数
def get_session_history(session_id: str):
    """
    获取会话历史，如果不存在则创建新的

    参数:
        session_id: 会话唯一标识符，用于区分不同用户或对话

    返回:
        ChatMessageHistory: 包含该会话所有历史消息的对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory() # 创建新的会话历史
    return store[session_id]

# 创建 RunnableWithMessageHistory 实例
# 参数说明：
# - prompt | llm: 组合提示模板和 LLM 的链
# - get_session_history: 获取会话历史的函数
# - input_messages_key: 输入消息的键名（对应 prompt 中的 {question}）
# - history_messages_key: 历史消息的键名（对应 prompt 中的 {history}）
conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== 对话 1 ===")
# 第一次对话：用户介绍自己的名字
# session_id 为 "session_001"，表示这是第一个会话
response1 = conversation.invoke(
    {"question": "你好，我叫张三，请记住我的名字"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你好，我叫张三，请记住我的名字")
print(f"助手: {response1.content}")

print("\n=== 对话 2 ===")
# 第二次对话：测试助手是否记住了用户的名字
# 使用相同的 session_id，所以助手可以访问之前的对话历史
response2 = conversation.invoke(
    {"question": "我刚才告诉你我的名字是什么？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我刚才告诉你我的名字是什么？")
print(f"助手: {response2.content}")

print("\n=== 查看记忆内容 ===")
# 获取并打印会话历史中的所有消息
history = get_session_history("session_001")
print(f"记忆中的消息数量: {len(history.messages)}")
for i, msg in enumerate(history.messages, 1):
    # 打印每条消息的类型和内容（前50个字符）
    print(f"消息 {i}: {type(msg).__name__} - {msg.content[:50]}...")

print("\n=== 对话 3 ===")
# 第三次对话：再次测试记忆功能
response3 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response3.content}")

print("\n=== 清除记忆后（新会话）===")
# 使用新的 session_id，创建一个新的会话
# 这个会话没有之前的历史记录
response4 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_002"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response4.content}")
