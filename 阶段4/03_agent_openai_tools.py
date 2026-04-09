"""
阶段4 - 03_agent_openai_tools.py
OpenAI Tool Calling 范式 - 金融助手 Agent

核心概念：
- create_agent: LangChain 1.2.x 统一的 Agent 构建入口
- Function Calling: LLM 原生支持输出结构化 tool_calls
- 多参数工具: 支持多个必选和可选参数的工具

本示例构建一个金融助手 Agent，具备以下能力：
- 查询股票价格
- 发送邮件
- 数据库 SQL 查询
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_ollama
from langchain.agents import create_agent
from langchain_core.tools import tool


@tool
def get_stock_price(symbol: str) -> str:
    """获取股票当前价格。当用户询问股票价格时使用。

    Args:
        symbol: 股票代码，如 AAPL、TSLA、000001、600519
    """
    prices = {
        "AAPL": {"price": 178.50, "change": "+2.3%"},
        "TSLA": {"price": 245.80, "change": "-1.5%"},
        "000001": {"price": 3256.78, "change": "+0.8%"},
        "600519": {"price": 1680.00, "change": "+1.2%"},
    }
    data = prices.get(symbol.upper())
    if not data:
        return f"未找到股票代码 '{symbol}'"
    return f"📈 {symbol.upper()} | 价格: ${data['price']} | 涨跌: {data['change']}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。当用户要求发邮件时使用。

    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文
    """
    print(f"  📧 [模拟] → {to} | 主题: {subject}")
    return f"邮件已成功发送至 {to}"


@tool
def query_database(sql: str) -> str:
    """执行数据库 SELECT 查询。仅允许 SELECT 语句。"""
    if not sql.strip().upper().startswith("SELECT"):
        return "错误：出于安全考虑，仅允许 SELECT 查询"
    return (
        "| 姓名       | 部门   | 薪资     |\n"
        "|------------|--------|----------|\n"
        "| 张三       | 技术   | 25000    |\n"
        "| 李四       | 产品   | 22000    |\n"
        "| 王五       | 市场   | 18000    |"
    )


def main():
    llm = make_ollama()
    tools = [get_stock_price, send_email, query_database]

    app = create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个专业的金融助手。你可以查询股价、发送邮件和查询数据库。请用中文回答。",
    )

    print("=" * 60)
    print("阶段4 - OpenAI Tools Agent (金融助手)")
    print("=" * 60)
    print(f"\n可用工具: {[t.name for t in tools]}")

    queries = [
        "查询 AAPL 的股价",
        "查一下 TSLA 和 600519 的股价",
        "查询数据库中所有员工信息",
    ]

    for q in queries:
        print(f"\n--- {q} ---")
        result = app.invoke({"messages": [{"role": "user", "content": q}]})
        ai_msg = result["messages"][-1]
        print(f"✅ 回答: {ai_msg.content}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
