"""
阶段4 - 02_tools_custom.py
自定义工具（@tool 装饰器）与工具绑定

核心概念：
- @tool 装饰器：将普通函数转换为 LangChain Tool
- StructuredTool：带参数校验的结构化工具
- 工具绑定：使用 bind_tools() 将工具绑定到 LLM

LangChain 1.x 提供了多种定义工具的方式：
1. @tool 装饰器（推荐）：最简洁，自动推断 schema
2. @tool("name", args_schema=...)：自定义名称和参数 schema
3. StructuredTool.from_function()：函数式创建
4. Tool 类：完整控制

工作流程：
1. 使用 @tool 定义工具函数
2. 创建 LLM 实例
3. 使用 llm.bind_tools() 绑定工具
4. LLM 返回包含 tool_calls 的响应
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_ollama
from langchain_core.tools import tool


# ========== 方式1：基础 @tool 装饰器 ==========
@tool
def search(query: str) -> str:
    """搜索信息。当需要查找事实、定义或知识时使用此工具。

    Args:
        query: 要搜索的问题或关键词
    """
    mock_db = {
        "python": "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统著称。",
        "langchain": "LangChain 是一个用于构建 LLM 应用的框架，提供链式调用、Agent 等能力。",
        "langgraph": "LangGraph 是基于有状态图的 Agent 框架，支持循环和条件边。",
    }
    for key, val in mock_db.items():
        if key in query.lower():
            return val
    return f"未找到关于 '{query}' 的信息。"


# ========== 方式2：结构化工具（自定义参数） ==========
@tool
def calculator(expression: str) -> str:
    """执行数学计算。当用户需要进行算术运算时使用此工具。

    Args:
        expression: 数学表达式，如 "2+2" 或 "(10*5)/2"
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# ========== 方式3：多参数工具 ==========
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气信息。

    Args:
        city: 城市名称，如 "北京"、"上海"、"广州"
        unit: 温度单位，支持 "celsius"（摄氏度）或 "fahrenheit"（华氏度）
    """
    weather_data = {
        "北京": {"temp": 22, "condition": "晴朗", "humidity": 35},
        "上海": {"temp": 26, "condition": "多云", "humidity": 65},
        "广州": {"temp": 30, "condition": "阴天", "humidity": 80},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "未知", "humidity": 50})
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32
    return (
        f"{city} 当前天气:\n"
        f"- 温度: {temp}{'°F' if unit == 'fahrenheit' else '°C'}\n"
        f"- 天气: {data['condition']}\n"
        f"- 湿度: {data['humidity']}%"
    )


# ========== 方式4：返回列表/复杂结构的工具 ==========
@tool
def list_files(directory: str) -> str:
    """列出指定目录下的文件和子目录。当需要查看目录内容时使用。

    Args:
        directory: 要列出的目录路径
    """
    from pathlib import Path as P
    p = Path(directory)
    if not p.exists():
        return f"错误：路径 '{directory}' 不存在"
    items = list(p.iterdir())
    if not items:
        return f"'{directory}' 是空目录"

    lines = [f"📁 {directory} 内容 ({len(items)} 项):"]
    for item in sorted(items, key=lambda x: (not x.is_dir(), x.name)):
        prefix = "📂 " if item.is_dir() else "📄 "
        lines.append(f"  {prefix}{item.name}")
    return "\n".join(lines)


def main():
    # 初始化模型
    llm = make_ollama()

    # 收集所有工具
    tools = [search, calculator, get_weather, list_files]

    print("=" * 60)
    print("阶段4 - 自定义工具（@tool 装饰器）")
    print("=" * 60)

    # ---- Part 1: 查看工具信息 ----
    print("\n【Part 1】已注册的工具\n")
    for t in tools:
        print(f"  工具名: {t.name}")
        print(f"  描述:   {t.description[:60]}...")
        print(f"  参数 Schema:")
        import json
        print(f"    {json.dumps(t.args_schema.model_json_schema(), ensure_ascii=False, indent=6)}")
        print()

    # ---- Part 2: 直接调用工具（不经过 LLM）----
    print("\n【Part 2】直接调用工具\n")

    print(f"> search('python'):")
    print(f"  {search.invoke('python')}")

    print(f"\n> calculator('123 * 456'):")
    print(f"  {calculator.invoke('123 * 456')}")

    print(f"\n> get_weather('北京', 'celsius'):")
    print(get_weather.invoke({"city": "北京", "unit": "celsius"}))

    print(f"\n> list_files('.'):")
    print(list_files.invoke("."))

    # ---- Part 3: 使用 bind_tools 绑定到 LLM ----
    print("\n" + "=" * 60)
    print("【Part 3】LLM 工具绑定 (bind_tools)")
    print("=" * 60 + "\n")

    bound_llm = llm.bind_tools(tools)

    queries = [
        "今天北京天气怎么样？",
        "帮我计算 256 * 128",
        "LangGraph 是什么？",
    ]

    for q in queries:
        print(f"用户: {q}")
        response = bound_llm.invoke(q)
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  → 工具调用: {tc['name']}")
                print(f"    参数: {tc['args']}")
        else:
            print(f"  → 直接回答: {response.content[:100]}")
        print()


if __name__ == "__main__":
    main()
