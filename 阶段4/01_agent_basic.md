# 01_agent_basic.py

## 功能说明

`AgentExecutor` + `create_react_agent` 是 LangChain 1.x 推荐使用的方式来创建能够使用工具的智能体。

## 核心特性

- **工具集成**: Agent 可以自主决定何时使用哪些工具
- **ReAct 范式**: Reasoning + Acting，让 Agent 先推理后行动
- **自动执行循环**: AgentExecutor 负责管理 Agent 的思考-行动-观察循环
- **错误处理**: 自动处理解析错误和工具执行异常

## 关键组件

### 1. Tool 定义

使用 `langchain_core.tools.Tool` 类定义代理可以使用的工具。

```python
from langchain_core.tools import Tool

def search_tool(query: str) -> str:
    # 搜索逻辑
    return search_result

search_tool = Tool(
    name="搜索",
    func=search_tool,
    description="用于搜索信息的工具。输入应该是要搜索的问题或关键词。"
)
```

### 2. ReAct 提示模板

从 langchain hub 获取预定义的 ReAct 提示模板，它指导 Agent 如何进行推理和行动。

```python
from langchain import hub
prompt = hub.pull("hwchase17/react")
```

### 3. Agent 创建

使用 `create_react_agent` 创建基于 ReAct 范式的代理。

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools, prompt)
```

### 4. AgentExecutor

将 Agent 与工具结合使用 AgentExecutor 来执行任务。

```python
from langchain_classic.agents import AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示思考过程
    handle_parsing_errors=True  # 自动处理解析错误
)
```

## 关键参数

| 参数 | 说明 |
|------|------|
| verbose | 是否显示 Agent 的详细思考过程 |
| handle_parsing_errors | 是否自动处理解析错误 |
| max_iterations | 最大迭代次数（防止无限循环） |
| early_stopping_method | 早停方法（"generate" 或 "force"） |

## 运行示例

```bash
python 阶段4/01_agent_basic.py
```

## 输出示例

```
=== 基础 Agent 示例 ===
这个 Agent 可以使用搜索和计算器两个工具
尝试提出需要使用工具的问题

--- 示例1: 搜索问题 ---
> Entering new AgentExecutor chain...
思考: 我需要搜索关于LangChain的信息
行动: 搜索
行动输入: LangChain 是什么？
观察: LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
思考: 我现在知道了答案
最终回答: LangChain 是一个用于开发由语言模型驱动的应用程序的框架。

> Finished chain.
最终回答: LangChain 是一个用于开发由语言模型驱动的应用程序的框架。

--- 示例2: 计算问题 ---
> Entering new AgentExecutor chain...
思考: 我需要计算 (25 * 4) + 10
行动: 计算器
行动输入: (25 * 4) + 10
观察: 结果: 110
思考: 我现在知道了答案
最终回答: (25 * 4) + 10 的结果是 110。

> Finished chain.
最终回答: (25 * 4) + 10 的结果是 110。

--- 示例3: 综合使用 ---
> Entering new AgentExecutor chain...
思考: 我需要首先搜索Agent的定义，然后进行计算
行动: 搜索
行动输入: Agent的定义
观察: Agent 是能够感知环境、做出决策并执行行动的实体。
思考: 现在我需要计算 100 除以 5
行动: 计算器
行动输入: 100 / 5
观察: 结果: 20.0
思考: 我现在知道了完整的答案
最终回答: Agent 是能够感知环境、做出决策并执行行动的实体。100 除以 5 的结果是 20.0。

> Finished chain.
最终回答: Agent 是能够感知环境、做出决策并执行行动的实体。100 除以 5 的结果是 20.0。
```

## 使用说明

1. Agent 会根据问题自动决定是否需要使用工具
2. 如果需要使用工具，Agent 会选择最合适的工具并执行
3. Agent 可能会多次使用工具来逐步解决问题
4. `verbose=True` 设置会显示 Agent 的完整思考过程
5. 在实际应用中，应替换模拟工具为真实的API调用