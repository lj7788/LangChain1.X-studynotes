"""
阶段3 - 02_memory_summary.py
Memory - 对话摘要记忆（LangChain 1.2.11 新版 API）
完全消除弃用警告，使用官方推荐方案

核心概念：
- 对话摘要：将完整的对话历史压缩成简短的摘要
- 动态摘要生成：每次对话后自动更新摘要
- 摘要记忆：用摘要代替完整历史，减少 token 消耗

工作流程：
1. 保存完整的对话历史
2. 每次对话后，用 LLM 生成对话摘要
3. 将摘要作为上下文传递给 LLM
4. 摘要包含用户的关键信息（姓名、职业、喜好等）

优点：
- 减少 token 消耗，降低成本
- 保留关键信息，过滤无关细节
- 适合长期对话场景

缺点：
- 可能丢失一些细节信息
- 需要额外的 LLM 调用生成摘要
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

# 使用字典存储不同会话的历史记录
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

def generate_summary(history: BaseChatMessageHistory) -> str:
    """
    生成对话历史的简洁摘要

    参数:
        history: 对话历史对象

    返回:
        str: 对话摘要字符串
    """
    # 如果没有历史消息，返回默认文本
    if not history.messages:
        return "暂无对话历史"
    
    # 将所有消息格式化为文本
    msg_text = "\n".join([
        f"用户: {msg.content}" if msg.type == "human" else f"助手: {msg.content}"
        for msg in history.messages
    ])
    
    # 定义摘要生成的提示模板
    summary_prompt = PromptTemplate.from_template("""
    请简洁总结以下对话，保留用户的关键信息（姓名、职业、喜好等）：
    {messages}
    总结：
    """)
    
    # 创建摘要生成链：提示模板 -> LLM -> 字符串输出解析器
    summary_chain = summary_prompt | llm | StrOutputParser()
    # 调用链生成摘要
    return summary_chain.invoke({"messages": msg_text}).strip()

# 定义主对话的提示模板
# 包含对话摘要和用户问题
prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话摘要回答用户问题。

对话摘要:
{chat_summary}

用户问题: {question}

回答:"""
)

def load_summary(inputs):
    """
    加载对话摘要并准备输入

    参数:
        inputs: 包含 session_id 和 question 的字典

    返回:
        dict: 包含 chat_summary 和 question 的字典
    """
    # 获取会话历史
    history = get_session_history(inputs.get("session_id", "default"))
    # 生成摘要
    return {
        "chat_summary": generate_summary(history),
        "question": inputs["question"]
    }

# 创建对话链：
# 1. RunnablePassthrough.assign: 保留原始输入，添加 chat_summary 字段
# 2. prompt: 格式化提示
# 3. llm: 调用语言模型
# 4. StrOutputParser: 解析输出为字符串
chain = (
    RunnablePassthrough.assign(chat_summary=load_summary)
    | prompt
    | llm
    | StrOutputParser()
)

def chat_with_summary(question, session_id="default"):
    """
    带摘要记忆的对话函数

    参数:
        question: 用户问题
        session_id: 会话 ID

    返回:
        tuple: (助手回复, 更新后的摘要)
    """
    # 调用对话链获取回复
    response = chain.invoke({"question": question, "session_id": session_id})
    # 获取会话历史
    history = get_session_history(session_id)
    # 添加用户消息和助手回复到历史
    history.add_user_message(question)
    history.add_ai_message(response)
    # 返回回复和生成的摘要
    return response, generate_summary(history)

# 定义测试用的会话 ID
SESSION_ID = "test_chat"

print("=== 对话 1 ===")
# 第一轮对话：用户介绍自己
q1 = "我叫张三，是一名软件工程师。我喜欢编程和读书。"
r1, s1 = chat_with_summary(q1, SESSION_ID)
print(f"用户: {q1}\n助手: {r1}")

print("\n=== 对话 2 ===")
# 第二轮对话：用户添加更多信息
q2 = "你喜欢什么运动？我喜欢打篮球。"
r2, s2 = chat_with_summary(q2, SESSION_ID)
print(f"用户: {q2}\n助手: {r2}")

print("\n=== 查看摘要记忆 ===")
# 打印当前摘要内容
print(f"记忆内容:\n{s2}")

print("\n=== 对话 3 ===")
# 第三轮对话：让助手总结用户信息
q3 = "总结一下我告诉你的关于我自己的信息"
r3, s3 = chat_with_summary(q3, SESSION_ID)
print(f"用户: {q3}\n助手: {r3}")

print("\n=== 查看更新后的摘要 ===")
# 打印更新后的摘要
print(f"记忆内容:\n{s3}")

print("\n=== 对话 4 ===")
# 第四轮对话：测试记忆是否有效
q4 = "我是谁？喜欢什么？"
r4, s4 = chat_with_summary(q4, SESSION_ID)
print(f"用户: {q4}\n助手: {r4}")
