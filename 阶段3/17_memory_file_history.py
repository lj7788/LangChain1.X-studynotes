"""
阶段3 - 17_memory_file_history.py
Memory - FileChatMessageHistory 文件历史记录（LangChain 1.x）

使用 FileChatMessageHistory 内置组件实现对话历史的持久化存储。

核心概念：
- FileChatMessageHistory: 文件存储的对话历史
- 持久化存储：将对话历史保存到 JSON 文件
- 自动管理：自动处理文件的读写

工作流程：
1. 创建 FileChatMessageHistory 对象，指定文件路径
2. 使用 RunnableWithMessageHistory 包装 LLM 链
3. 进行对话，消息自动保存到文件
4. 下次启动时自动加载历史记录

文件格式：
- JSON 格式存储
- 包含所有消息的类型、内容和元数据
- 自动处理文件不存在的情况

优点：
- 内置组件，使用简单
- 自动处理文件的读写
- 支持持久化存储

缺点：
- 只支持单文件存储
- 不支持多会话管理
- 文件较大时性能可能下降

使用场景：
- 需要持久化对话历史
- 单会话应用
- 不需要复杂的多用户管理
"""

import os
from pathlib import Path
import sys

sys.path.append("../")

from tools import make_ollama
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

# 初始化 Ollama LLM 模型
llm = make_ollama()

# 定义数据目录和记忆文件路径
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "chat_history.json"

# 创建文件历史对象
# FileChatMessageHistory 会自动处理文件的读写
history = FileChatMessageHistory(MEMORY_FILE)

# 定义提示模板
# SystemMessage: 设置系统角色和行为
# MessagesPlaceholder: 动态插入对话历史
# HumanMessage: 用户的问题
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

# 创建对话链
# lambda session_id: history: 所有会话共享同一个历史对象
conversation = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== FileChatMessageHistory 示例 ===\n")

print("对话 1：用户自我介绍")
# 第一轮对话：用户介绍自己
r1 = conversation.invoke(
    {"question": "你好，我叫张三，是一名软件工程师。"},
    config={"configurable": {"session_id": "default"}}
)
print(f"用户: 你好，我叫张三，是一名软件工程师。")
print(f"助手: {r1.content}\n")

print("对话 2：继续对话")
# 第二轮对话：用户添加更多信息
r2 = conversation.invoke(
    {"question": "我喜欢 Python 编程。"},
    config={"configurable": {"session_id": "default"}}
)
print(f"用户: 我喜欢 Python 编程。")
print(f"助手: {r2.content}\n")

print("对话 3：查询之前的信息")
# 第三轮对话：测试记忆是否有效
r3 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "default"}}
)
print(f"用户: 我是谁？")
print(f"助手: {r3.content}\n")

print(f"=== 查看保存的文件 ===")
# 打印文件路径和消息数量
print(f"文件路径: {MEMORY_FILE}")
print(f"消息数量: {len(history.messages)}")

# 如果文件存在，打印文件内容的前500个字符
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"\n文件内容:\n{content[:500]}...")
