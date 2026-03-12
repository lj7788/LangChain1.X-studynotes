"""
阶段3 - 02_memory_summary.py
Memory - ConversationSummaryMemory 对话摘要
适配 LangChain 1.2.11 版本（修正 Memory 导入路径）
"""
import sys
sys.path.append("../")
from tools import make_ollama  # 保持你的 Ollama 初始化函数

# ========== 关键修正：1.2.11 正确的导入路径 ==========
from langchain_classic.memory import ConversationSummaryMemory  # 核心修正
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化 LLM（保持你的逻辑）
llm = make_ollama()

# 2. 初始化对话摘要记忆（参数不变，导入路径修正）
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

# 3. 定义 Prompt（保持原有模板）
prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

用户问题: {question}

回答:"""
)

# 4. 构建带记忆的链（逻辑不变，1.2 标准方式）
def load_memory(inputs):
    """加载对话记忆的函数"""
    return memory.load_memory_variables(inputs)["chat_history"]

# 构建 Runnable 链
chain = (
    RunnablePassthrough.assign(chat_history=load_memory)  # 先加载记忆
    | prompt  # 传入 Prompt
    | llm  # 调用 LLM
    | StrOutputParser()  # 解析输出为字符串
)

# 5. 封装对话逻辑（手动保存记忆）
def chat_with_memory(question):
    """调用链 + 手动保存记忆"""
    response = chain.invoke({"question": question})
    # 手动保存对话到摘要记忆中（1.2 必须手动做）
    memory.save_context(
        inputs={"question": question},
        outputs={"output": response}
    )
    return response

# ========== 测试对话逻辑 ==========
print("=== 对话 1 ===")
question1 = "我叫张三，是一名软件工程师。我喜欢编程和读书。"
response1 = chat_with_memory(question1)
print(f"用户: {question1}")
print(f"助手: {response1}")

print("\n=== 对话 2 ===")
question2 = "你喜欢什么运动？我喜欢打篮球。"
response2 = chat_with_memory(question2)
print(f"用户: {question2}")
print(f"助手: {response2}")

print("\n=== 查看摘要记忆 ===")
print(f"记忆内容:\n{memory.buffer}")

print("\n=== 对话 3 ===")
question3 = "总结一下我告诉你的关于我自己的信息"
response3 = chat_with_memory(question3)
print(f"用户: {question3}")
print(f"助手: {response3}")

print("\n=== 查看更新后的摘要 ===")
print(f"记忆内容:\n{memory.buffer}")

print("\n=== 对话 4 ===")
question4 = "我是谁？喜欢什么？"
response4 = chat_with_memory(question4)
print(f"用户: {question4}")
print(f"助手: {response4}")