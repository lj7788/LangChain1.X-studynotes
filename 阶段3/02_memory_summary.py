"""
阶段3 - 02_memory_summary.py
Memory - 对话摘要记忆（LangChain 1.2.11 新版 API）
完全消除弃用警告，使用官方推荐方案
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

store = {}

def get_session_history(session_id: str = "default") -> BaseChatMessageHistory:
    """获取/创建会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_summary(history: BaseChatMessageHistory) -> str:
    """生成对话历史的简洁摘要"""
    if not history.messages:
        return "暂无对话历史"
    
    msg_text = "\n".join([
        f"用户: {msg.content}" if msg.type == "human" else f"助手: {msg.content}"
        for msg in history.messages
    ])
    
    summary_prompt = PromptTemplate.from_template("""
    请简洁总结以下对话，保留用户的关键信息（姓名、职业、喜好等）：
    {messages}
    总结：
    """)
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"messages": msg_text}).strip()

prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话摘要回答用户问题。

对话摘要:
{chat_summary}

用户问题: {question}

回答:"""
)

def load_summary(inputs):
    history = get_session_history(inputs.get("session_id", "default"))
    return {
        "chat_summary": generate_summary(history),
        "question": inputs["question"]
    }

chain = (
    RunnablePassthrough.assign(chat_summary=load_summary)
    | prompt
    | llm
    | StrOutputParser()
)

def chat_with_summary(question, session_id="default"):
    response = chain.invoke({"question": question, "session_id": session_id})
    history = get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(response)
    return response, generate_summary(history)

SESSION_ID = "test_chat"

print("=== 对话 1 ===")
q1 = "我叫张三，是一名软件工程师。我喜欢编程和读书。"
r1, s1 = chat_with_summary(q1, SESSION_ID)
print(f"用户: {q1}\n助手: {r1}")

print("\n=== 对话 2 ===")
q2 = "你喜欢什么运动？我喜欢打篮球。"
r2, s2 = chat_with_summary(q2, SESSION_ID)
print(f"用户: {q2}\n助手: {r2}")

print("\n=== 查看摘要记忆 ===")
print(f"记忆内容:\n{s2}")

print("\n=== 对话 3 ===")
q3 = "总结一下我告诉你的关于我自己的信息"
r3, s3 = chat_with_summary(q3, SESSION_ID)
print(f"用户: {q3}\n助手: {r3}")

print("\n=== 查看更新后的摘要 ===")
print(f"记忆内容:\n{s3}")

print("\n=== 对话 4 ===")
q4 = "我是谁？喜欢什么？"
r4, s4 = chat_with_summary(q4, SESSION_ID)
print(f"用户: {q4}\n助手: {r4}")
