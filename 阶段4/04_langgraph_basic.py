"""
阶段4 - 04_langgraph_basic.py
LangGraph 基础：State / Node / Edge

核心概念：
- State（状态）：定义图的状态结构（TypedDict）
- Node（节点）：处理状态的函数，接收并返回 State
- Edge（边）：连接节点的路径，决定执行流向

LangGraph 是 LangChain 官方推荐的有状态图框架，
适合构建复杂的 Agent 工作流（循环、分支、条件路由）。

基本架构：

    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  Start  │────▶│ Node A  │────▶│ Node B  │────▶ End
    └─────────┘     └─────────┘     └─────────┘

本示例构建一个简单的「文章生成流水线」：
    选题 → 大纲生成 → 正文撰写 → 最终输出
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END


# ========== Step 1: 定义状态 ==========
class ArticleState(TypedDict):
    """文章生成的全局状态"""
    topic: str                    # 选题
    outline: str                  # 大纲
    content: str                  # 正文内容
    messages: Annotated[list, operator.add]  # 过程消息（累加）


# ========== Step 2: 定义节点函数 ==========
def node_topic(state: ArticleState) -> dict:
    """节点1：确定选题和写作方向"""
    topic = state.get("topic", "人工智能")
    msg = f"[选题] 确定写作主题: {topic}"
    print(f"  ▶ {msg}")
    return {
        "topic": topic,
        "messages": [msg],
    }


def node_outline(state: ArticleState) -> dict:
    """节点2：根据选题生成大纲"""
    topic = state["topic"]
    outline = (
        f"# {topic} 文章大纲\n\n"
        f"## 一、引言\n- 背景介绍\n- 研究意义\n\n"
        f"## 二、核心概念\n- 定义与分类\n- 发展历程\n\n"
        f"## 三、应用场景\n- 行业案例\n- 实践经验\n\n"
        f"## 四、总结与展望\n"
    )
    msg = f"[大纲] 已为《{topic}》生成大纲 ({len(outline)} 字符)"
    print(f"  ▶ {msg}")
    return {
        "outline": outline,
        "messages": [msg],
    }


def node_content(state: ArticleState) -> dict:
    """节点3：根据大纲撰写正文"""
    topic = state["topic"]
    content = (
        f"{topic} 正在改变我们的世界。\n\n"
        f"从学术研究到工业应用，{topic} 展现出了巨大的潜力。\n"
        f"本文将从多个维度深入探讨这一话题。\n\n"
        f"（此处省略 3000 字正文...）\n\n"
        f"—— 本文完 ——"
    )
    msg = f"[正文] 已完成《{topic}》的正文撰写 ({len(content)} 字符)"
    print(f"  ▶ {msg}")
    return {
        "content": content,
        "messages": [msg],
    }


# ========== Step 3: 构建图 ==========
def build_graph():
    """构建文章生成流水线图"""

    # 创建有向图
    graph = StateGraph(ArticleState)

    # 添加节点
    graph.add_node("topic_node", node_topic)
    graph.add_node("outline_node", node_outline)
    graph.add_node("content_node", node_content)

    # 添加边（定义执行顺序）
    graph.add_edge(START, "topic_node")      # 开始 → 选题
    graph.add_edge("topic_node", "outline_node")  # 选题 → 大纲
    graph.add_edge("outline_node", "content_node") # 大纲 → 正文
    graph.add_edge("content_node", END)       # 正文 → 结束

    # 编译图（生成可运行的应用）
    app = graph.compile()
    return app


def main():
    print("=" * 60)
    print("阶段4 - LangGraph 基础（State / Node / Edge）")
    print("=" * 60)
    print("\n构建的文章生成流水线:")
    print("  START → 选题 → 大纲 → 正文 → END\n")

    # 编译图
    app = build_graph()

    # 打印图结构（Mermaid 格式）
    print("【图结构】Mermaid 格式:\n")
    try:
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f"  （无法渲染 Mermaid: {e}）")
    print()

    # 运行
    print("-" * 40)
    print("【运行】输入: {'topic': 'LangChain'}\n")
    result = app.invoke({
        "topic": "LangChain",
        "outline": "",
        "content": "",
        "messages": [],
    })

    print("\n" + "-" * 40)
    print("【最终状态】")
    print(f"  选题:   {result['topic']}")
    print(f"  大纲长度: {len(result['outline'])} 字符")
    print(f"  正文长度: {len(result['content'])} 字符")
    print(f"  执行步骤: {len(result['messages'])} 步")
    print("\n  过程消息:")
    for m in result["messages"]:
        print(f"    • {m}")

    # 再次运行不同主题
    print(f"\n{'=' * 60}")
    print("【再运行一次】输入: {'topic': '量子计算'}\n")
    result2 = app.invoke({
        "topic": "量子计算",
        "outline": "",
        "content": "",
        "messages": [],
    })
    print(f"\n  选题: {result2['topic']}, 正文: {len(result2['content'])} 字符")


if __name__ == "__main__":
    main()
