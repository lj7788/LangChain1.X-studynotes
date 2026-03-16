"""
阶段3 - 12_memory_token_buffer.py
Memory - Token计数记忆（LangChain 1.2.11 新版 API）

根据 Token 数量来管理对话历史，当超过指定 token 数时自动清理旧消息。
使用 ChatMessageHistory + 手动 Token 控制实现。

核心概念：
- Token 计数：根据文本长度估算 token 数量
- Token 限制：设置最大 token 数量
- 智能裁剪：从旧消息开始删除，直到满足限制

工作流程：
1. 添加新对话到历史
2. 计算当前历史的总 token 数
3. 如果超过限制，从旧消息开始删除
4. 保留最新的对话，满足 token 限制

Token 计算方法：
- 简单方法：字符数 / 4（近似值）
- 精确方法：使用 tokenizer 计算（更准确）

优点：
- 精确控制 token 消耗
- 适应不同长度的对话
- 避免 token 超限错误

缺点：
- 需要准确的 token 计数
- 可能丢失重要的早期信息
- 裁剪策略可能不够智能

使用场景：
- Token 预算严格的应用
- 对话长度变化较大的场景
- 需要精确控制成本的系统
"""

import sys
sys.path.append("../")
from tools import make_ollama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 初始化 Ollama LLM 模型
llm = make_ollama()

# 定义最大 token 数量
MAX_TOKENS = 100
store = {}

def get_session_history(session_id: str = "default") -> BaseChatMessageHistory:
    """
    获取/创建会话历史

    参数:
        session_id: 会话唯一标识符，默认为 "default"

    返回:
        BaseChatMessageHistory: 包含该会话所有历史消息的对象
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def count_tokens(text: str) -> int:
    """
    估算文本的 token 数量

    参数:
        text: 待计算的文本

    返回:
        int: 估算的 token 数量（字符数 / 4）
    """
    return len(text) // 4

def trim_by_tokens(history: ChatMessageHistory, max_tokens: int = MAX_TOKENS):
    """
    根据 token 数量裁剪对话历史

    参数:
        history: 对话历史对象
        max_tokens: 最大允许的 token 数量
    """
    total_tokens = 0
    keep_messages = []
    # 从最新消息开始倒序遍历
    for msg in reversed(history.messages):
        msg_tokens = count_tokens(msg.content)
        if total_tokens + msg_tokens <= max_tokens:
            # 如果添加这条消息不会超过限制，就保留
            keep_messages.insert(0, msg)
            total_tokens += msg_tokens
        else:
            # 否则停止添加
            break
    history.messages = keep_messages

# 定义提示模板
prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

用户问题: {question}

回答:"""
)

def load_history(inputs):
    """
    加载对话历史并准备输入

    参数:
        inputs: 包含 session_id 和 question 的字典

    返回:
        dict: 包含 chat_history 和 question 的字典
    """
    session_id = inputs.get("session_id", "default")
    history = get_session_history(session_id)
    # 将所有消息格式化为文本
    history_text = "\n".join([
        f"用户: {m.content}" if m.type == "human" else f"助手: {m.content}"
        for m in history.messages
    ])
    return {
        "chat_history": history_text,
        "question": inputs["question"]
    }

# 创建对话链
chain = (
    RunnablePassthrough.assign(chat_history=load_history)
    | prompt
    | llm
    | StrOutputParser()
)

def chat_with_token_limit(question, session_id="default"):
    """
    带 token 限制的对话函数

    参数:
        question: 用户问题
        session_id: 会话 ID

    返回:
        str: 助手的回复
    """
    # 调用对话链获取回复
    response = chain.invoke({"question": question, "session_id": session_id})
    # 获取会话历史
    history = get_session_history(session_id)
    # 添加用户消息和助手回复到历史
    history.add_user_message(question)
    history.add_ai_message(response)
    # 根据 token 限制裁剪历史
    trim_by_tokens(history, MAX_TOKENS)
    return response

print("=== 对话 1：发送较长消息 ===")
# 第一轮对话：发送较长的消息
q1 = "我叫张三，我是一名软件工程师，已经工作了5年。我擅长 Python 编程和后端开发。我之前在一家互联网公司负责架构设计和系统优化工作。"
r1 = chat_with_token_limit(q1)
print(f"用户: {q1[:50]}...\n助手: {r1}")

# 计算当前 token 数
history = get_session_history()
total = sum(count_tokens(m.content) for m in history.messages)
print(f"当前 token 数: {total}")

print("\n=== 对话 2：继续发送消息 ===")
# 第二轮对话：继续发送消息
q2 = "我最近在学习大模型应用开发，包括 RAG、Agent 等技术方向。希望能够将传统软件开发经验与 AI 技术结合。"
r2 = chat_with_token_limit(q2)
print(f"用户: {q2[:50]}...\n助手: {r2}")

# 计算当前 token 数
history = get_session_history()
total = sum(count_tokens(m.content) for m in history.messages)
print(f"当前 token 数: {total}")

print("\n=== 对话 3：检查是否超过限制 ===")
# 第三轮对话：发送短消息
q3 = "今天天气真不错！"
r3 = chat_with_token_limit(q3)
print(f"用户: {q3}\n助手: {r3}")

# 计算当前 token 数
history = get_session_history()
total = sum(count_tokens(m.content) for m in history.messages)
print(f"当前 token 数: {total}")

print("\n=== 对话 4：继续对话 ===")
# 第四轮对话：继续对话
q4 = "你喜欢编程吗？"
r4 = chat_with_token_limit(q4)
print(f"用户: {q4}\n助手: {r4}")
