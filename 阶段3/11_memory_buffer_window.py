"""
阶段3 - 11_memory_buffer_window.py
Memory - 窗口记忆（LangChain 1.2.11 新版 API）

只保留最近 k 轮对话，避免记忆无限增长。
使用 ChatMessageHistory + 手动窗口控制实现。
"""

import sys
sys.path.append("../")
from tools import make_ollama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

llm = make_ollama()

WINDOW_SIZE = 2
store = {}

def get_session_history(session_id: str = "default") -> BaseChatMessageHistory:
    """获取/创建会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def trim_history(history: ChatMessageHistory, k: int = WINDOW_SIZE):
    """只保留最近 k 轮对话"""
    if len(history.messages) > k * 2:
        history.messages = history.messages[-(k * 2):]

prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

用户问题: {question}

回答:"""
)

def load_history(inputs):
    session_id = inputs.get("session_id", "default")
    history = get_session_history(session_id)
    messages = history.messages[-WINDOW_SIZE * 2:] if history.messages else []
    history_text = "\n".join([
        f"用户: {m.content}" if m.type == "human" else f"助手: {m.content}"
        for m in messages
    ])
    return {
        "chat_history": history_text,
        "question": inputs["question"]
    }

chain = (
    RunnablePassthrough.assign(chat_history=load_history)
    | prompt
    | llm
    | StrOutputParser()
)

def chat_with_window(question, session_id="default"):
    response = chain.invoke({"question": question, "session_id": session_id})
    history = get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(response)
    trim_history(history, WINDOW_SIZE)
    return response

print("=== 对话 1：第1轮 ===")
q1 = "我叫张三，喜欢编程。"
r1 = chat_with_window(q1)
print(f"用户: {q1}\n助手: {r1}")

print("\n=== 对话 2：第2轮 ===")
q2 = "我喜欢 Python 语言。"
r2 = chat_with_window(q2)
print(f"用户: {q2}\n助手: {r2}")

print("\n=== 对话 3：第3轮（窗口k=2，只会保留最近2轮）===")
q3 = "我今天天气很好。"
r3 = chat_with_window(q3)
print(f"用户: {q3}\n助手: {r3}")

print("\n=== 查看记忆（应该只保留最近2轮）===")
history = get_session_history()
print(f"记忆轮数: {len(history.messages) // 2}")
for i, msg in enumerate(history.messages):
    print(f"  {i+1}. {type(msg).__name__}: {msg.content}")

print("\n=== 对话 4：第4轮 ===")
q4 = "你还记得我叫什么吗？"
r4 = chat_with_window(q4)
print(f"用户: {q4}\n助手: {r4}")
