"""
阶段3 - 04_memory_persist.py
Memory - 记忆的保存与加载

展示如何保存和加载 ConversationBufferMemory 的对话历史。
"""

import json
from pathlib import Path
from langchain_community.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
sys.path.append("/Volumes/data/code/me/2026/03/LangChain1.X-")
from tools import make_model

llm = make_model()

MEMORY_FILE = Path(__file__).parent / "data" / "memory.json"


def save_memory(memory: ConversationBufferMemory):
    messages = []
    for msg in memory.chat_memory.messages:
        messages.append({
            "type": type(msg).__name__,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        })
    
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"记忆已保存到: {MEMORY_FILE}")


def load_memory(memory: ConversationBufferMemory) -> bool:
    if not MEMORY_FILE.exists():
        print("没有找到保存的记忆文件")
        return False
    
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    
    memory.chat_memory.messages.clear()
    for msg in messages:
        msg_type = msg["type"]
        content = msg["content"]
        additional_kwargs = msg.get("additional_kwargs", {})
        
        if msg_type == "HumanMessage":
            memory.chat_memory.messages.append(
                HumanMessage(content=content, **additional_kwargs)
            )
        elif msg_type == "AIMessage":
            memory.chat_memory.messages.append(
                AIMessage(content=content, **additional_kwargs)
            )
        elif msg_type == "SystemMessage":
            memory.chat_memory.messages.append(
                SystemMessage(content=content, **additional_kwargs)
            )
    
    print(f"记忆已加载，共 {len(messages)} 条消息")
    return True


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

template = """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

用户问题: {question}

回答:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question", "chat_history"]
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

print("=== 场景1：创建新对话并保存 ===\n")

response1 = chain.invoke({"question": "你好！我叫张三"})
print(f"用户: 你好！我叫张三")
print(f"助手: {response1['text']}")

response2 = chain.invoke({"question": "我喜欢机器学习"})
print(f"用户: 我喜欢机器学习")
print(f"助手: {response2['text']}")

print("\n=== 保存记忆 ===")
save_memory(memory)

print("\n=== 场景2：清除记忆（模拟重启应用）===")
memory.clear()
print("记忆已清除")

print("\n=== 场景3：加载保存的记忆 ===")
load_memory(memory)

print("\n=== 使用加载的记忆继续对话 ===")
response3 = chain.invoke({"question": "我是谁？"})
print(f"用户: 我是谁？")
print(f"助手: {response3['text']}")

response4 = chain.invoke({"question": "我刚才说我喜欢什么？"})
print(f"用户: 我刚才说我喜欢什么？")
print(f"助手: {response4['text']}")
