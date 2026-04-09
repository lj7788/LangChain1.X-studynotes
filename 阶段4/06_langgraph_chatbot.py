"""
阶段4 - 06_langgraph_chatbot.py
基于 LangGraph 的带记忆聊天机器人实战

核心概念：
- MessageGraph vs StateGraph：消息序列专用图
- SystemMessage / HumanMessage / AIMessage：标准消息格式
- AddMessages：消息历史追加 reducer（内置）
- 工具调用集成：在图中嵌入 LLM + Tools 循环

图结构：

    ┌──────────┐
    │  START   │
    └────┬─────┘
         ▼
    ┌──────────┐
    │  call_llm │ ◀────┐
    └────┬─────┘      │
         ▼            │
   ┌─────────────┐    │
   │ 有工具调用？  │───┤ Yes: 执行工具后回 call_llm
   └──────┬──────┘    │
          │ No        │
          ▼           │
         END ◀────────┘
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from tools import make_ollama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ========== 定义工具 ==========
@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户询问现在几点/今天几号时使用。"""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S 星期%w")


@tool
def calculate(expression: str) -> str:
    """执行数学计算。当用户需要算术运算时使用。

    Args:
        expression: 数学表达式，如 "(100 + 200) * 3"
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：包含非法字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


TOOLS = [get_current_time, calculate]
tool_map = {t.name: t for t in TOOLS}

# ========== 状态定义（使用 MessagesState）==========
class ChatState(TypedDict):
    """聊天机器人状态"""
    messages: Annotated[list, add_messages]


# ========== 节点函数 ==========

def call_llm(state: ChatState) -> dict:
    """调用 LLM，绑定工具"""
    llm = make_ollama()
    bound_llm = llm.bind_tools(TOOLS)

    messages = state["messages"]
    response = bound_llm.invoke(messages)
    print(f"  🤖 LLM 回复: {response.content[:80] if response.content else '(工具调用)'}...")

    return {"messages": [response]}


def execute_tools(state: ChatState) -> dict:
    """执行 LLM 请求的工具调用"""
    messages = list(state["messages"])
    last_msg = messages[-1]

    if not last_msg.tool_calls:
        return {}

    tool_results = []
    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        print(f"  🔧 执行工具: {name}({args})")

        if name in tool_map:
            result = tool_map[name].invoke(args)
        else:
            result = f"未知工具: {name}"

        tool_results.append(result)

    # 将工具结果作为 ToolMessage 追加
    from langchain_core.messages import ToolMessage
    new_messages = []
    for tc, result in zip(last_msg.tool_calls, tool_results):
        new_messages.append(
            ToolMessage(content=result, tool_call_id=tc["id"])
        )

    return {"messages": new_messages}


def should_continue(state: ChatState) -> str:
    """判断是否还有工具需要执行"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return END


# ========== 构建图 ==========
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", execute_tools)

    graph.add_edge(START, "call_llm")# 条件边：LLM 回复后检查是否需要执行工具
    graph.add_conditional_edges("call_llm", should_continue, {
        "tools": "tools",
        END: END,
    })

    # 工具执行完后继续调 LLM
    graph.add_edge("tools", "call_llm")

    return graph.compile()


def main():
    print("=" * 60)
    print("阶段4 - LangGraph 聊天机器人（含工具 + 记忆）")
    print("=" * 60)

    app = build_graph()

    # 打印图结构
    try:
        print("\n【图结构 Mermaid】\n")
        print(app.get_graph().draw_mermaid())
    except Exception:
        pass

    # 初始系统消息
    system_msg = SystemMessage(content=(
        "你是一个有用的AI助手。你可以使用工具来获取当前时间或进行数学计算。"
        "请用中文回答用户的问题。"
    ))

    config = {"configurable": {"thread_id": "chat-001"}}

    print("\n" + "=" * 60)
    print("开始对话！（输入 'quit' 退出）\n")

    # 多轮对话
    conversations = [
        "你好！你是谁？",
        "现在几点了？",
        "帮我计算 (256 + 128) * 4",
        "刚才的计算结果再乘以 2 是多少？",
    ]

    state = {"messages": [system_msg]}
    for user_input in conversations:
        print(f"👤 用户: {user_input}")

        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state, config=config)

        ai_message = result["messages"][-1]
        if isinstance(ai_message, AIMessage):
            print(f"🤖 助手: {ai_message.content}")

        # 更新状态（保留完整对话历史）
        state = result
        print()

    # 显示完整的对话记忆
    print("=" * 60)
    print(f"【对话统计】共 {len(state['messages'])} 条消息")
    for i, msg in enumerate(state["messages"]):
        role = type(msg).__name__.replace("Message", "")
        content_preview = (msg.content or "")[:50]
        print(f"  [{i:>2}] {role}: {content_preview}")


if __name__ == "__main__":
    main()
