"""
阶段4 - 01_agent_basic.py
Agent 基础 - 使用 create_agent 创建工具调用智能体

LangChain 1.2.x 新API：
- create_agent(model, tools, system_prompt) → 直接返回 CompiledStateGraph
- 不再需要单独的 AgentExecutor 和 prompt 模板
- 输入格式: {"messages": [...]}
- 输出格式: {"messages": [...]} （包含完整对话历史）

核心概念：
- Agent: 能自主决定何时使用工具的 LLM
- Tools: Agent 可调用的外部函数
- System Prompt: 指导 Agent 的行为规则

工作流程：
1. 定义工具函数（@tool 装饰器）
2. 创建 LLM 实例
3. 用 create_agent() 组装
4. invoke() 执行查询
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_ollama
from langchain.agents import create_agent
from langchain_core.tools import tool


# ========== 定义工具 ==========
@tool
def search(query: str) -> str:
    """搜索信息。当需要查找事实、定义或知识时使用此工具。"""
    mock_results = {
        "LangChain": "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。",
        "Agent": "Agent 是能够感知环境、做出决策并执行行动的实体。",
        "Tools": "Tools 是代理可以使用的外部函数或服务，用于扩展其能力。",
    }
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    return f"未找到关于 '{query}' 的具体信息。"


@tool
def calculator(expression: str) -> str:
    """执行数学计算。输入数学表达式如 '2+2' 或 '(10*5)/2'。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        return f"结果: {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


# ========== 创建 Agent ==========
llm = make_ollama()
tools = [search, calculator]

agent_app = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一个能够使用工具解决问题的智能助手。\n"
        "你有搜索和计算两个工具可用。\n"
        "请根据用户问题自主决定是否需要使用工具，然后给出最终回答。"
    ),
)

print("=" * 60)
print("阶段4 - 基础 Agent 示例")
print("=" * 60)
print(f"\n已注册工具: {[t.name for t in tools]}")

test_cases = [
    "LangChain 是什么？",
    "计算 (25 * 4) + 10",
    "先搜索 Agent 的定义，再算一下 100 除以 5",
]

for i, question in enumerate(test_cases):
    print(f"\n--- 示例{i + 1}: {question} ---")
    result = agent_app.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # 从结果中提取最终 AI 回复
    ai_msg = result["messages"][-1]
    print(f"最终回答: {ai_msg.content}")

print("\n=== 使用说明 ===")
print("1. create_agent() 自动处理 工具调用→观察→再思考 的循环")
print("2. 输入必须是 {'messages': [{'role': 'user', 'content': ...}]} 格式")
print("3. 结果的 messages 包含完整对话历史（含工具调用记录）")
