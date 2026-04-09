"""
阶段4 - 07_agent_toolkits.py
常用工具集（ToolKit）与 Agent 集成

核心概念：
- Toolkit：将一组相关的工具打包为一个集合，方便复用和管理
- LangChain 内置多种 Toolkit：文件操作、JSON、SQL、Shell 等
- 动态构建：根据需求动态选择和组合工具

本示例展示：
1. 内置 Toolkit 的使用方式
2. 自定义 Toolkit 的创建方法
3. 如何将 Toolkit 与 Agent / LangGraph 集成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.tools import tool

# ============================================================
# Part 1: 使用内置 Toolkit
# ============================================================

def demo_json_tools():
    """演示 JSON 工具集"""
    print("=" * 50)
    print("Part 1: JSON Toolkit")
    print("=" * 50)

    from langchain_community.tools.json.tool import JsonSpec, JsonListKeysTool, JsonGetValueTool

    sample_data = {
        "公司": {
            "名称": "TechCorp",
            "员工": [
                {"姓名": "张三", "部门": "技术", "级别": "P7"},
                {"姓名": "李四", "部门": "产品", "级别": "P6"},
                {"姓名": "王五", "部门": "市场", "级别": "P5"},
            ],
        },
    }

    spec = JsonSpec(dict_=sample_data, max_value_length=100)
    json_list_tool = JsonListKeysTool(spec=spec)
    json_get_tool = JsonGetValueTool(spec=spec)

    print("\n  可用工具:")
    print(f"    • {json_list_tool.name}: {json_list_tool.description[:40]}...")
    print(f"    • {json_get_tool.name}: {json_get_tool.description[:40]}...")

    # 直接使用（参数名为 tool_input，值为 JSON 路径字符串）
    keys = json_list_tool.invoke("")
    print(f"\n  根路径键列表: {keys}")

    value = json_get_tool.invoke("/公司/员工/0")
    print(f"  获取值: {value}")


def demo_file_tools():
    """演示文件操作工具集"""
    print("\n" + "=" * 50)
    print("Part 2: File Toolkit (自定义)")
    print("=" * 50)

    from langchain_core.tools import BaseTool
    from typing import Optional

    @tool
    def read_file(path: str) -> str:
        """读取文本文件内容。当需要查看文件内容时使用。

        Args:
            path: 文件的绝对或相对路径
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"[{path}] ({len(content)} 字符)\n前200字符:\n{content[:200]}"
        except FileNotFoundError:
            return f"错误：文件 '{path}' 不存在"
        except Exception as e:
            return f"读取错误: {e}"

    @tool
    def write_file(path: str, content: str) -> str:
        """将内容写入文本文件。当需要创建或修改文件时使用。

        Args:
            path: 文件路径
            content: 要写入的内容
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"成功写入 {path} ({len(content)} 字符)"
        except Exception as e:
            return f"写入错误: {e}"

    @tool
    def list_directory(path: str = ".") -> str:
        """列出目录中的文件和子目录。当需要浏览目录结构时使用。

        Args:
            path: 目录路径（默认当前目录）
        """
        p = Path(path)
        if not p.exists():
            return f"错误：'{path}' 不存在"
        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = [f"📁 {path}:"]
        for item in items[:15]:
            prefix = "📂 " if item.is_dir() else "📄 "
            lines.append(f"  {prefix}{item.name}")
        if len(items) > 15:
            lines.append(f"  ... 还有 {len(items)-15} 项")
        return "\n".join(lines)

    file_tools = [read_file, write_file, list_directory]

    print("\n  文件工具集:")
    for t in file_tools:
        print(f"    • {t.name}")
    
    # 演示调用
    print(f"\n  > list_directory('.'):")
    print(f"    {list_directory.invoke('.')}")


def demo_custom_toolkit():
    """演示自定义 Toolkit 类"""
    print("\n" + "=" * 50)
    print("Part 3: 自定义 Toolkit 类")
    print("=" * 50)

    from pydantic import BaseModel, Field

    class MathToolkit:
        """数学计算工具集"""
        
        @property
        def tools(self):
            """返回该 Toolkit 包含的所有工具"""
            return self._tools()

        def _tools(self):
            @tool
            def add(a: float, b: float) -> str:
                """两数相加。Args: a: 第一个数, b: 第二个数"""
                return str(a + b)

            @tool
            def multiply(a: float, b: float) -> str:
                """两数相乘。Args: a: 第一个数, b: 第二个数"""
                return str(a * b)

            @tool
            def power(base: float, exponent: float) -> str:
                """幂运算。Args: base: 底数, exponent: 指数"""
                return str(base ** exponent)

            @tool
            def factorial(n: int) -> str:
                """阶乘运算。Args: n: 非负整数"""
                if n < 0:
                    return "错误：n 必须为非负整数"
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return str(result)

            return [add, multiply, power, factorial]

    toolkit = MathToolkit()
    tools = toolkit.tools

    # 演示调用
    calculation = toolkit.tools[0]
    print(f"\n  Toolkit 包含 {len(tools)} 个工具:")
    for t in tools:
        print(f"    • {t.name}")

    print(f"\n  > add(10, 25):")
    print(f"    {tools[0].invoke({'a': 10, 'b': 25})}")
    print(f"\n  > factorial(6):")
    print(f"    {tools[3].invoke({'n': 6})}")


def demo_integration():
    """演示 Toolkit 与 Agent 的集成"""
    print("\n" + "=" * 50)
    print("Part 4: Toolkit + Agent 集成")
    print("=" * 50)

    from tools import make_ollama
    from langchain.agents import create_agent

    # 组合多个 Toolkit 的工具
    all_tools = []
    
    # 来自自定义的数学工具集
    class QuickTools:
        @property
        def tools(self):
            @tool
            def get_length(text: str) -> str:
                """返回字符串长度。当用户询问字符数量时使用。
                Args: text: 要测量长度的字符串
                """
                return str(len(text))

            @tool
            def reverse_text(text: str) -> str:
                """反转字符串。当用户要求反转文字时使用。
                Args: text: 要反转的字符串
                """
                return text[::-1]

            return [get_length, reverse_text]

    all_tools.extend(QuickTools().tools)

    llm = make_ollama()
    app = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt="你是一个助手，可以计算字符串长度和反转文字。",
    )

    print(f"\n  已注册 {len(all_tools)} 个工具:")
    for t in all_tools:
        print(f"    • {t.name}: {t.description[:45]}...")

    queries = ["'LangChain' 这个词有几个字符？", "把 'Hello World' 反转一下"]
    for q in queries:
        print(f"\n  👤 用户: {q}")
        result = app.invoke({"messages": [{"role": "user", "content": q}]})
        ai_msg = result["messages"][-1]
        print(f"  🤖 回答: {ai_msg.content}")


if __name__ == "__main__":
    demo_json_tools()
    demo_file_tools()
    demo_custom_toolkit()
    demo_integration()

    print("\n" + "=" * 60)
    print("总结：Toolkit 的三种组织方式")
    print("=" * 60)
    print("""
  方式1: 直接用内置 Toolkit（如 JsonToolkit）
    → 适合通用场景，开箱即用
  
  方式2: 用 @tool 定义后手动组装为 list
    → 最灵活，推荐日常使用
  
  方式3: 封装为 Toolkit 类（含 .tools 属性）
    → 适合需要跨项目复用的工具集合""")
