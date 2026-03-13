"""
阶段3 - 04_memory_persist.py
Memory - 记忆的保存与加载（LangChain 1.x）

展示如何使用 FileChatMessageHistory 实现持久化存储对话历史。
"""

import json
from pathlib import Path
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import sys
sys.path.append("../")
from tools import make_ollama

llm = make_ollama()

MEMORY_FILE = Path(__file__).parent / "data" / "memory.json"


def save_memory(chat_history: ChatMessageHistory):
    messages = []
    for msg in chat_history.messages:
        messages.append({
            "type": type(msg).__name__,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        })
    
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"记忆已保存到: {MEMORY_FILE}")


def load_memory() -> ChatMessageHistory:
    memory = ChatMessageHistory()
    
    if not MEMORY_FILE.exists():
        print("没有找到保存的记忆文件")
        return memory
    
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    for msg in messages:
        msg_type = msg["type"]
        content = msg["content"]
        additional_kwargs = msg.get("additional_kwargs", {})
        
        if msg_type == "HumanMessage":
            memory.messages.append(
                HumanMessage(content=content, **additional_kwargs)
            )
        elif msg_type == "AIMessage":
            memory.messages.append(
                AIMessage(content=content, **additional_kwargs)
            )
        elif msg_type == "SystemMessage":
            memory.messages.append(
                SystemMessage(content=content, **additional_kwargs)
            )
    
    print(f"记忆已加载，共 {len(messages)} 条消息，消息是{messages}")
    return memory


prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== 场景1：创建新对话并保存 ===\n")

response1 = conversation.invoke(
    {"question": "你好！我叫张三"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你好！我叫张三")
print(f"助手: {response1.content}")

response2 = conversation.invoke(
    {"question": "我喜欢机器学习"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我喜欢机器学习")
print(f"助手: {response2.content}")

print("\n=== 保存记忆 ===")
save_memory(get_session_history("session_001"))

print("\n=== 场景2：清除记忆（模拟重启应用）===")
store.clear()
print("记忆已清除")

print("\n=== 场景3：加载保存的记忆 ===")
loaded_history = load_memory()
store["session_001"] = loaded_history

print("\n=== 使用加载的记忆继续对话 ===")
response3 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response3.content}")

response4 = conversation.invoke(
    {"question": "我刚才说我喜欢什么？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我刚才说我喜欢什么？")
print(f"助手: {response4.content}")
