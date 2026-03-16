"""
阶段3 - 11_memory_buffer_window.py
Memory - 窗口记忆（LangChain 1.2.11 新版 API）

只保留最近 k 轮对话，避免记忆无限增长。
使用 ChatMessageHistory + 手动窗口控制实现。

核心概念：
- 窗口记忆：只保留最近 k 轮对话
- 滑动窗口：新对话进入，旧对话移出
- 固定大小：记忆大小固定，不会无限增长

工作流程：
1. 添加新对话到历史
2. 检查历史长度是否超过窗口大小
3. 如果超过，移除最旧的对话
4. 保留最近的 k 轮对话

优点：
- 控制记忆大小，避免 token 超限
- 保持对话的连贯性
- 适合短期对话场景

缺点：
- 丢失早期对话信息
- 无法回忆很久之前的内容
- 需要设置合适的窗口大小

使用场景：
- 短期对话场景
- Token 预算有限的应用
- 不需要长期记忆的聊天机器人
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

# 定义窗口大小：保留最近 2 轮对话
WINDOW_SIZE = 2
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

def trim_history(history: ChatMessageHistory, k: int = WINDOW_SIZE):
    """
    只保留最近 k 轮对话

    参数:
        history: 对话历史对象
        k: 保留的对话轮数
    """
    if len(history.messages) > k * 2:
        history.messages = history.messages[-(k * 2):]

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
    # 只保留最近 WINDOW_SIZE 轮对话
    messages = history.messages[-WINDOW_SIZE * 2:] if history.messages else []
    # 将消息格式化为文本
    history_text = "\n".join([
        f"用户: {m.content}" if m.type == "human" else f"助手: {m.content}"
        for m in messages
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

def chat_with_window(question, session_id="default"):
    """
    带窗口记忆的对话函数

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
    # 裁剪历史，只保留最近 WINDOW_SIZE 轮
    trim_history(history, WINDOW_SIZE)
    return response

print("=== 对话 1：第1轮 ===")
# 第一轮对话：用户介绍自己
q1 = "我叫张三，喜欢编程。"
r1 = chat_with_window(q1)
print(f"用户: {q1}\n助手: {r1}")

print("\n=== 对话 2：第2轮 ===")
# 第二轮对话：用户添加更多信息
q2 = "我喜欢 Python 语言。"
r2 = chat_with_window(q2)
print(f"用户: {q2}\n助手: {r2}")

print("\n=== 对话 3：第3轮（窗口k=2，只会保留最近2轮）===")
# 第三轮对话：由于窗口大小为2，第1轮对话会被移除
q3 = "我今天天气很好。"
r3 = chat_with_window(q3)
print(f"用户: {q3}\n助手: {r3}")

print("\n=== 查看记忆（应该只保留最近2轮）===")
# 查看当前记忆中的消息
history = get_session_history()
print(f"记忆轮数: {len(history.messages) // 2}")
for i, msg in enumerate(history.messages):
    print(f"  {i+1}. {type(msg).__name__}: {msg.content}")

print("\n=== 对话 4：第4轮 ===")
# 第四轮对话：由于窗口大小为2，第2轮对话会被移除
q4 = "你还记得我叫什么吗？"
r4 = chat_with_window(q4)
print(f"用户: {q4}\n助手: {r4}")
