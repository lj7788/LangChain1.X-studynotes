"""
阶段3 - 12_memory_token_buffer.py
Memory - Token计数记忆（LangChain 1.2.11 新版 API）

根据 Token 数量来管理对话历史，当超过指定 token 数时自动清理旧消息。
使用 ChatMessageHistory + 手动 Token 控制实现。
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

MAX_TOKENS = 100
store = {}

def get_session_history(session_id: str = "default") -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def count_tokens(text: str) -> int:
    return len(text) // 4

def trim_by_tokens(history: ChatMessageHistory, max_tokens: int = MAX_TOKENS):
    total_tokens = 0
    keep_messages = []
    for msg in reversed(history.messages):
        msg_tokens = count_tokens(msg.content)
        if total_tokens + msg_tokens <= max_tokens:
            keep_messages.insert(0, msg)
            total_tokens += msg_tokens
        else:
            break
    history.messages = keep_messages

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
    history_text = "\n".join([
        f"用户: {m.content}" if m.type == "human" else f"助手: {m.content}"
        for m in history.messages
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

def chat_with_token_limit(question, session_id="default"):
    response = chain.invoke({"question": question, "session_id": session_id})
    history = get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(response)
    trim_by_tokens(history, MAX_TOKENS)
    return response

print("=== 对话 1：发送较长消息 ===")
q1 = "我叫张三，我是一名软件工程师，已经工作了5年。我擅长 Python 编程和后端开发。我之前在一家互联网公司负责架构设计和系统优化工作。"
r1 = chat_with_token_limit(q1)
print(f"用户: {q1[:50]}...\n助手: {r1}")

history = get_session_history()
total = sum(count_tokens(m.content) for m in history.messages)
print(f"当前 token 数: {total}")

print("\n=== 对话 2：继续发送消息 ===")
q2 = "我最近在学习大模型应用开发，包括 RAG、Agent 等技术方向。希望能够将传统软件开发经验与 AI 技术结合。"
r2 = chat_with_token_limit(q2)
print(f"用户: {q2[:50]}...\n助手: {r2}")

history = get_session_history()
total = sum(count_tokens(m.content) for m in history.messages)
print(f"当前 token 数: {total}")

print("\n=== 对话 3：检查是否超过限制 ===")
q3 = "今天天气真不错！"
r3 = chat_with_token_limit(q3)
print(f"用户: {q3}\n助手: {r3}")

history = get_session_history()
total = sum(count_tokens(m.content) for m in history.messages)
print(f"当前 token 数: {total}")

print("\n=== 对话 4：继续对话 ===")
q4 = "你喜欢编程吗？"
r4 = chat_with_token_limit(q4)
print(f"用户: {q4}\n助手: {r4}")
