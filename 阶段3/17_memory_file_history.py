"""
阶段3 - 17_memory_file_history.py
Memory - FileChatMessageHistory 文件历史记录（LangChain 1.x）

使用 FileChatMessageHistory 内置组件实现对话历史的持久化存储。
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

llm = make_ollama()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "chat_history.json"

history = FileChatMessageHistory(MEMORY_FILE)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

conversation = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== FileChatMessageHistory 示例 ===\n")

print("对话 1：用户自我介绍")
r1 = conversation.invoke(
    {"question": "你好，我叫张三，是一名软件工程师。"},
    config={"configurable": {"session_id": "default"}}
)
print(f"用户: 你好，我叫张三，是一名软件工程师。")
print(f"助手: {r1.content}\n")

print("对话 2：继续对话")
r2 = conversation.invoke(
    {"question": "我喜欢 Python 编程。"},
    config={"configurable": {"session_id": "default"}}
)
print(f"用户: 我喜欢 Python 编程。")
print(f"助手: {r2.content}\n")

print("对话 3：查询之前的信息")
r3 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "default"}}
)
print(f"用户: 我是谁？")
print(f"助手: {r3.content}\n")

print(f"=== 查看保存的文件 ===")
print(f"文件路径: {MEMORY_FILE}")
print(f"消息数量: {len(history.messages)}")

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"\n文件内容:\n{content[:500]}...")
