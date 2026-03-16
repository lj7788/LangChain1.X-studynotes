"""
阶段3 - 04_memory_persist.py
Memory - 记忆的保存与加载（LangChain 1.x）

展示如何使用 FileChatMessageHistory 实现持久化存储对话历史。

核心概念：
- 持久化存储：将对话历史保存到文件，避免程序重启后丢失
- JSON 格式：使用 JSON 格式序列化和反序列化消息
- 手动管理：需要手动实现保存和加载逻辑

工作流程：
1. 创建对话并进行交流
2. 将对话历史保存到 JSON 文件
3. 清除内存中的对话历史（模拟程序重启）
4. 从文件加载对话历史
5. 继续对话，验证记忆是否恢复

使用场景：
- 需要长期保存对话历史
- 程序重启后需要恢复之前的对话
- 跨会话保持用户记忆

注意事项：
- 需要处理文件不存在的情况
- 需要正确处理不同类型的消息（HumanMessage、AIMessage、SystemMessage）
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

# 初始化 Ollama LLM 模型
llm = make_ollama()

# 定义记忆文件的保存路径
MEMORY_FILE = Path(__file__).parent / "data" / "memory.json"


def save_memory(chat_history: ChatMessageHistory):
    """
    将对话历史保存到 JSON 文件

    参数:
        chat_history: ChatMessageHistory 对象，包含所有对话消息
    """
    # 将消息对象转换为可序列化的字典
    messages = []
    for msg in chat_history.messages:
        messages.append({
            "type": type(msg).__name__,  # 消息类型（HumanMessage、AIMessage、SystemMessage）
            "content": msg.content,  # 消息内容
            "additional_kwargs": msg.additional_kwargs  # 额外的关键字参数
        })
    
    # 确保目录存在
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 将消息列表保存到 JSON 文件
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"记忆已保存到: {MEMORY_FILE}")


def load_memory() -> ChatMessageHistory:
    """
    从 JSON 文件加载对话历史

    返回:
        ChatMessageHistory: 包含加载的消息的对话历史对象
    """
    # 创建空的对话历史对象
    memory = ChatMessageHistory()
    
    # 如果文件不存在，返回空的历史
    if not MEMORY_FILE.exists():
        print("没有找到保存的记忆文件")
        return memory
    
    # 从 JSON 文件读取消息
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    # 根据消息类型创建对应的消息对象
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


# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

# 使用字典存储不同会话的历史记录
store = {}


def get_session_history(session_id: str):
    """
    获取会话历史，如果不存在则创建新的

    参数:
        session_id: 会话唯一标识符

    返回:
        ChatMessageHistory: 包含该会话所有历史消息的对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建 RunnableWithMessageHistory 实例
conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("=== 场景1：创建新对话并保存 ===\n")

# 第一次对话：用户介绍自己
response1 = conversation.invoke(
    {"question": "你好！我叫张三"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 你好！我叫张三")
print(f"助手: {response1.content}")

# 第二次对话：用户添加更多信息
response2 = conversation.invoke(
    {"question": "我喜欢机器学习"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我喜欢机器学习")
print(f"助手: {response2.content}")

print("\n=== 保存记忆 ===")
# 将对话历史保存到文件
save_memory(get_session_history("session_001"))

print("\n=== 场景2：清除记忆（模拟重启应用）===")
# 清空内存中的对话历史，模拟程序重启
store.clear()
print("记忆已清除")

print("\n=== 场景3：加载保存的记忆 ===")
# 从文件加载对话历史
loaded_history = load_memory()
# 将加载的历史放入 store
store["session_001"] = loaded_history

print("\n=== 使用加载的记忆继续对话 ===")
# 测试记忆是否恢复成功
response3 = conversation.invoke(
    {"question": "我是谁？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我是谁？")
print(f"助手: {response3.content}")

# 继续测试记忆
response4 = conversation.invoke(
    {"question": "我刚才说我喜欢什么？"},
    config={"configurable": {"session_id": "session_001"}}
)
print(f"用户: 我刚才说我喜欢什么？")
print(f"助手: {response4.content}")
