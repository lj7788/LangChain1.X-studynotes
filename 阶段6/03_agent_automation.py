"""
阶段6 - 03_agent_automation.py
Agent 自动化任务实战 - 使用工具自动完成任务

综合运用阶段1-5的知识：
- LangGraph 状态图 + 多步循环（阶段4）
- 多工具协作 Agent（阶段4）
- 结构化输出 - 任务规划（阶段5）
- 错误处理与重试（阶段5）
- 回调监控（阶段5）

项目架构：
1. 任务规划 Agent - 将复杂任务分解为子任务
2. 任务执行 Agent - 逐步执行子任务
3. 结果汇总 - 汇总所有子任务结果

场景：数据分析助手
- 用户输入一个数据分析需求
- Agent 自动规划、搜索、计算、汇总

图结构：

    ┌──────────────┐
    │    START      │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  任务规划     │  ← 结构化输出: 分解子任务
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  执行子任务   │◀──────┐
    │  (Agent+Tools)│       │
    └──────┬───────┘       │
           ▼               │
    ┌──────────────┐       │
    │  还有子任务？  │──Yes──┘
    └──────┬───────┘
           │ No
           ▼
    ┌──────────────┐
    │  结果汇总     │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │     END       │
    └──────────────┘
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import make_ollama


# ========== Part 1: 工具定义 ==========

@tool
def query_sales_data(product: str = "", region: str = "", period: str = "") -> str:
    """查询销售数据。支持按产品、地区、时间段筛选。

    Args:
        product: 产品名称（如"手机"、"笔记本"），留空查全部
        region: 地区（如"华东"、"华南"），留空查全部
        period: 时间段（如"2024-Q1"、"2024-Q2"），留空查最近一季度
    """
    # 模拟销售数据
    sales_data = {
        ("手机", "华东", "2024-Q1"): {"revenue": 5280000, "units": 8800, "growth": "+15%"},
        ("手机", "华东", "2024-Q2"): {"revenue": 6120000, "units": 10200, "growth": "+16%"},
        ("手机", "华南", "2024-Q1"): {"revenue": 3960000, "units": 6600, "growth": "+8%"},
        ("手机", "华南", "2024-Q2"): {"revenue": 4500000, "units": 7500, "growth": "+14%"},
        ("笔记本", "华东", "2024-Q1"): {"revenue": 7200000, "units": 3600, "growth": "+12%"},
        ("笔记本", "华东", "2024-Q2"): {"revenue": 8100000, "units": 4050, "growth": "+13%"},
        ("笔记本", "华南", "2024-Q1"): {"revenue": 5400000, "units": 2700, "growth": "+5%"},
        ("笔记本", "华南", "2024-Q2"): {"revenue": 6300000, "units": 3150, "growth": "+17%"},
    }

    results = []
    for (p, r, t), data in sales_data.items():
        if product and product not in p:
            continue
        if region and region not in r:
            continue
        if period and period != t:
            continue
        results.append(f"{p}|{r}|{t}: 营收¥{data['revenue']:,}, 销量{data['units']}台, 同比{data['growth']}")

    if not results:
        return f"未找到匹配数据（产品={product}, 地区={region}, 时间={period}）"

    return "\n".join(results)


@tool
def calculate(expression: str) -> str:
    """执行数学计算。支持基本算术运算。

    Args:
        expression: 数学表达式，如 "5280000 + 6120000"
    """
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "错误：包含非法字符"
    try:
        result = eval(expression)
        return f"{expression} = {result:,}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def generate_chart(data: str, chart_type: str = "bar") -> str:
    """生成数据可视化图表（模拟）。

    Args:
        data: 数据描述，如 "手机Q1:528万, Q2:612万; 笔记本Q1:720万, Q2:810万"
        chart_type: 图表类型（bar/line/pie）
    """
    chart_types = {"bar": "柱状图", "line": "折线图", "pie": "饼图"}
    chart_name = chart_types.get(chart_type, "柱状图")
    return (
        f"📊 [{chart_name}] 已生成\n"
        f"  数据: {data}\n"
        f"  文件: /output/chart_{hash(data) % 10000:04d}.png"
    )


@tool
def write_report(title: str, content: str) -> str:
    """生成分析报告（模拟）。

    Args:
        title: 报告标题
        content: 报告内容（Markdown 格式）
    """
    filename = f"report_{hash(title) % 10000:04d}.md"
    return (
        f"📝 报告已生成\n"
        f"  标题: {title}\n"
        f"  文件: /output/{filename}\n"
        f"  内容长度: {len(content)} 字符"
    )


TOOLS = [query_sales_data, calculate, generate_chart, write_report]
tool_map = {t.name: t for t in TOOLS}


# ========== Part 2: 任务规划（结构化输出）==========

class SubTask(BaseModel):
    """子任务"""
    id: int = Field(description="子任务编号")
    description: str = Field(description="子任务描述")
    tool_name: str = Field(description="需要使用的工具名称")
    tool_args: str = Field(description="工具参数描述")


class TaskPlan(BaseModel):
    """任务规划结果"""
    subtasks: List[SubTask] = Field(description="分解的子任务列表")
    summary: str = Field(description="任务规划摘要")


# ========== Part 3: 状态定义 ==========

class AutomationState(TypedDict):
    """自动化任务状态"""
    user_request: str                          # 用户原始需求
    plan: str                                    # 任务规划（JSON 字符串）
    current_step: int                            # 当前执行步骤
    total_steps: int                             # 总步骤数
    step_results: Annotated[list, lambda a, b: a + b]  # 每步的执行结果
    messages: Annotated[list, lambda a, b: a + b]      # Agent 对话历史
    final_report: str                            # 最终报告


# ========== Part 4: 节点函数 ==========

def node_plan(state: AutomationState) -> dict:
    """任务规划节点 - 将复杂需求分解为子任务"""
    llm = make_ollama()
    structured_llm = llm.with_structured_output(TaskPlan)

    prompt = (
        f"你是一个数据分析任务规划师。请将以下用户需求分解为具体的子任务，"
        f"每个子任务指定需要使用的工具。\n\n"
        f"可用工具：\n"
        f"- query_sales_data: 查询销售数据（参数：product, region, period）\n"
        f"- calculate: 数学计算（参数：expression）\n"
        f"- generate_chart: 生成图表（参数：data, chart_type）\n"
        f"- write_report: 生成报告（参数：title, content）\n\n"
        f"用户需求: {state['user_request']}"
    )

    plan: TaskPlan = structured_llm.invoke(prompt)

    print(f"  [规划] 共 {len(plan.subtasks)} 个子任务:")
    for st in plan.subtasks:
        print(f"    {st.id}. {st.description} → {st.tool_name}({st.tool_args})")
    print(f"  [规划] 摘要: {plan.summary}")

    return {
        "plan": plan.model_dump_json(),
        "current_step": 0,
        "total_steps": len(plan.subtasks),
        "step_results": [],
    }


def node_execute(state: AutomationState) -> dict:
    """任务执行节点 - 使用 Agent + Tools 执行当前子任务"""
    import json

    llm = make_ollama()
    bound_llm = llm.bind_tools(TOOLS)

    # 解析计划
    plan_data = json.loads(state["plan"])
    subtasks = plan_data["subtasks"]
    current = state["current_step"]

    if current >= len(subtasks):
        return {"step_results": []}

    subtask = subtasks[current]
    print(f"\n  [执行] 子任务 {current + 1}/{len(subtasks)}: {subtask['description']}")

    # 构建执行消息
    messages = [
        SystemMessage(content=(
            "你是一个数据分析执行助手。请执行以下子任务，使用合适的工具。\n"
            "执行完毕后，用简洁的文字总结执行结果。"
        )),
        HumanMessage(content=(
            f"原始需求: {state['user_request']}\n"
            f"当前子任务: {subtask['description']}\n"
            f"建议工具: {subtask['tool_name']}\n"
            f"工具参数: {subtask['tool_args']}\n\n"
            f"之前的执行结果:\n"
            + "\n".join(f"- {r}" for r in state.get("step_results", []))
            + "\n\n请执行子任务并总结结果。"
        )),
    ]

    # 调用 LLM（可能触发工具）
    response = bound_llm.invoke(messages)
    new_messages = [response]

    # 执行工具调用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            print(f"    [工具] {name}({args})")

            if name in tool_map:
                result = tool_map[name].invoke(args)
            else:
                result = f"未知工具: {name}"

            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )
            print(f"    [结果] {str(result)[:80]}...")

        # 再次调用 LLM 总结
        all_messages = messages + new_messages
        final_response = llm.invoke(all_messages)
        step_result = final_response.content
    else:
        step_result = response.content

    print(f"  [执行] 结果: {step_result[:80]}...")

    return {
        "current_step": current + 1,
        "step_results": [step_result],
        "messages": new_messages,
    }


def node_should_continue(state: AutomationState) -> str:
    """判断是否还有子任务需要执行"""
    if state["current_step"] < state["total_steps"]:
        return "execute"
    return "summarize"


def node_summarize(state: AutomationState) -> dict:
    """结果汇总节点 - 生成最终报告"""
    llm = make_ollama()

    results_text = "\n".join(
        f"{i+1}. {r}" for i, r in enumerate(state["step_results"])
    )

    prompt = ChatPromptTemplate.from_template(
        "你是一个数据分析报告撰写专家。请基于以下分析结果，撰写一份简洁专业的分析报告。\n\n"
        "用户原始需求: {request}\n\n"
        "分析步骤与结果:\n{results}\n\n"
        "请撰写报告，包含：\n"
        "1. 核心发现（3条以内）\n"
        "2. 数据摘要\n"
        "3. 建议与展望"
    )

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke({
        "request": state["user_request"],
        "results": results_text,
    })

    print(f"\n  [汇总] 最终报告已生成 ({len(report)} 字符)")

    return {"final_report": report}


# ========== Part 5: 构建图 ==========

def build_automation_graph():
    """构建自动化任务 LangGraph"""
    graph = StateGraph(AutomationState)

    # 添加节点
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("summarize", node_summarize)

    # 添加边
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        node_should_continue,
        {"execute": "execute", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


# ========== Part 6: 主函数 ==========

def main():
    print("=" * 60)
    print("阶段6 - Agent 自动化任务实战")
    print("=" * 60)

    app = build_automation_graph()

    # 打印图结构
    try:
        print("\n【图结构 Mermaid】\n")
        print(app.get_graph().draw_mermaid())
    except Exception:
        pass

    # 测试任务
    tasks = [
        "分析华东地区2024年Q1和Q2手机和笔记本的销售情况，对比增长趋势，生成图表和报告",
    ]

    for i, task in enumerate(tasks):
        print(f"\n{'═' * 60}")
        print(f"【任务{i + 1}】{task}")
        print(f"{'═' * 60}")

        result = app.invoke({
            "user_request": task,
            "plan": "",
            "current_step": 0,
            "total_steps": 0,
            "step_results": [],
            "messages": [],
            "final_report": "",
        })

        # 打印最终报告
        print(f"\n{'─' * 60}")
        print("📋 最终分析报告:")
        print(f"{'─' * 60}")
        print(result["final_report"])

        # 执行统计
        print(f"\n{'─' * 60}")
        print(f"【执行统计】")
        print(f"  子任务数: {result['total_steps']}")
        print(f"  执行结果数: {len(result['step_results'])}")
        print(f"  消息数: {len(result['messages'])}")


if __name__ == "__main__":
    main()
