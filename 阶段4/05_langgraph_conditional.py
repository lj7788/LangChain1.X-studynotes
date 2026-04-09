"""
阶段4 - 05_langgraph_conditional.py
LangGraph 条件边与循环

核心概念：
- 条件边（Conditional Edge）：根据 State 动态决定下一个节点
- 循环：通过条件边回到之前的节点，形成循环
- add_conditional_edges：注册条件路由函数

图结构（本例：智能客服路由）：

                    ┌──────────────┐
              ┌────▶│  human_agent │ (人工客服)
              │     └──────────────┘
    START     │
      │       │     ┌──────────────┐
      ▼       └────▶│   end_call   │ (结束)
┌──────────┐        └──────────────┘
│ classify │
│  (分类)  │──?──▶  ┌──────────────┐
└──────────┘        │  bot_agent   │ (机器人)
                    └──────┬───────┘
                           │
                     ┌─────▼──────┐
                     │  resolve   │ (解决问题)
                     └──────┬─────┘
                            │
                      (是否解决?)
                       /          \
                     是             否
                      │              │
                      ▼              ▼
                   END           human_agent
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Literal, Annotated
import operator
from langgraph.graph import StateGraph, START, END


# ========== 状态定义 ==========
class SupportState(TypedDict):
    """智能客服状态"""
    query: str                          # 用户问题
    category: str                       # 问题类别
    bot_answer: str                     # 机器人的回答
    resolved: bool                      # 是否已解决
    messages: Annotated[list, operator.add]  # 过程记录


# ========== 节点函数 ==========

def node_classify(state: SupportState) -> dict:
    """节点1：问题分类 - 判断是技术/账单还是其他"""
    q = state["query"].lower()
    if any(kw in q for kw in ["登录", "密码", "无法访问", "报错", "error", "bug"]):
        cat = "技术问题"
    elif any(kw in q for kw in ["账单", "费用", "退款", "扣款", "price", "money"]):
        cat = "账单问题"
    else:
        cat = "一般咨询"

    msg = f"[分类] 问题归类为: {cat}"
    print(f"  ▶ {msg}")
    return {"category": cat, "messages": [msg]}


def node_bot(state: SupportState) -> dict:
    """节点2：机器人自动回复"""
    cat = state["category"]
    answers = {
        "技术问题": (
            "您好！针对技术问题，请尝试以下步骤：\n"
            "1. 清除浏览器缓存后重试\n"
            "2. 更新到最新版本\n"
            "3. 如果仍存在问题，我们将为您转接人工服务。"
        ),
        "账单问题": (
            "您好！关于账单问题：\n"
            "1. 请登录「账户中心」查看详细账单\n"
            "2. 如有疑问可在7天内申请复核\n"
            "3. 需要人工协助请告知。"
        ),
        "一般咨询": (
            "您好！感谢您的咨询。\n"
            "常见问题FAQ：https://help.example.com\n"
            "如需更多帮助，请描述具体需求。"
        ),
    }
    answer = answers.get(cat, "抱歉，我暂时无法回答这个问题。")
    msg = f"[机器人] 已回复 ({cat})"
    print(f"  ▶ {msg}")
    return {"bot_answer": answer, "messages": [msg]}


def node_human(state: SupportState) -> dict:
    """节点3：转人工客服"""
    msg = "[人工] 已转接到人工客服，工号: CS-8866"
    print(f"  ▶ {msg}")
    return {"resolved": True, "messages": [msg]}


def node_resolve_check(state: SupportState) -> dict:
    """节点4：确认用户是否满意（模拟）"""
    cat = state["category"]
    # 模拟：一般咨询通常能被机器人解决，其他可能需要人工
    is_resolved = cat == "一般咨询"
    status = "✅ 已解决" if is_resolved else "❌ 需要人工介入"
    msg = f"[确认] 用户反馈: {status}"
    print(f"  ▶ {msg}")
    return {"resolved": is_resolved, "messages": [msg]}


# ========== 条件路由函数 ==========

def route_after_classify(state: SupportState) -> Literal["bot_agent", "human_agent", "end_call"]:
    """分类后的路由：简单问题直接结束，复杂问题走不同流程"""
    cat = state["category"]
    q = state["query"].lower()

    # 投诉类直接转人工
    if any(kw in q for kw in ["投诉", "举报", "投诉"]):
        return "human_agent"
    # 退订/注销也直接处理
    if any(kw in q for kw in ["退订", "注销", "删除账号"]):
        return "end_call"

    return "bot_agent"


def route_after_resolve(state: SupportState) -> Literal[END, "human_agent"]:
    """解决确认后的路由"""
    if state.get("resolved"):
        return END
    return "human_agent"


# ========== 构建图 ==========
def build_graph():
    graph = StateGraph(SupportState)

    # 注册节点
    graph.add_node("classify", node_classify)
    graph.add_node("bot_agent", node_bot)
    graph.add_node("human_agent", node_human)
    graph.add_node("resolve_check", node_resolve_check)

    # 注册普通边和条件边
    graph.add_edge(START, "classify")

    # 条件边：分类 → 根据类别路由到不同处理节点
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"bot_agent": "bot_agent", "human_agent": "human_agent", "end_call": END},
    )

    # 机器人回复 → 确认是否解决
    graph.add_edge("bot_agent", "resolve_check")

    # 条件边：确认结果 → 解决则结束，否则转人工
    graph.add_conditional_edges(
        "resolve_check",
        route_after_resolve,
        {END: END, "human_agent": "human_agent"},
    )

    # 人工客服 → 结束
    graph.add_edge("human_agent", END)

    return graph.compile()


def main():
    print("=" * 60)
    print("阶段4 - LangGraph 条件边与循环")
    print("=" * 60)
    print("\n【智能客服路由系统】")
    print("  流程: 分类 → (条件路由) → 机器人/人工 → 结束\n")

    app = build_graph()

    # 打印 Mermaid 图结构
    try:
        mermaid = app.get_graph().draw_mermaid()
        print("【Mermaid 图】\n")
        print(mermaid)
        print()
    except Exception as e:
        print(f"  （Mermaid 渲染失败: {e}）\n")

    test_cases = [
        "我登录不了，一直报错 500",
        "我想查询上个月的账单",
        "你们公司的主营业务是什么？",
        "我要投诉你们的服务态度！",
    ]

    for i, query in enumerate(test_cases):
        print(f"\n{'─' * 50}")
        print(f"测试{i+1}: {query}\n")
        result = app.invoke({
            "query": query,
            "category": "",
            "bot_answer": "",
            "resolved": False,
            "messages": [],
        })
        print(f"\n  结果:")
        print(f"    类别: {result['category']}")
        print(f"    已解决: {'是' if result['resolved'] else '否'}")
        print(f"    执行步数: {len(result['messages'])}")
        for m in result['messages']:
            print(f"    • {m}")


if __name__ == "__main__":
    main()
