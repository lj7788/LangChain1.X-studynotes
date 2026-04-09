# 03_agent_openai_tools.md

## 功能说明

演示 **OpenAI Tool Calling** 范式的 Agent 创建与执行。

## 核心概念

### ReAct vs OpenAI Tool Calling 对比

| 特性 | ReAct (`create_react_agent`) | Tool Calling (`create_tool_calling_agent`) |
|------|---------------------------|------------------------------------------|
| 原理 | 思考→行动→观察 循环 | LLM 原生返回 `tool_calls` |
| Token 效率 | 较低（每步需输出推理文本） | 高（直接输出结构化调用） |
| 模型要求 | 通用（只需遵循指令） | 需要 function calling 支持 |
| 可解释性 | 高（完整思考链可见） | 中等（需额外日志） |
| 适用场景 | 复杂多步推理 | 精确工具调用 / API 操作 |

## 关键代码结构

```
create_tool_calling_agent(llm, tools, prompt)
       ↓
AgentExecutor(agent=agent, tools=tools)
       ↓
executor.invoke({"input": "用户问题"})
```

## 运行方式

```bash
python 阶段4/03_agent_openai_tools.py
```

## 本文件包含的工具

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `get_stock_price` | 查询股票价格 | `symbol`: 股票代码 |
| `send_email` | 发送邮件 | `to`, `subject`, `body` |
| `query_database` | SQL 数据库查询 | `sql`: SELECT 语句 |

## 注意事项

1. **模型兼容性**：确保使用的模型支持 function calling（如 GPT-4、GLM-4 等）
2. **安全限制**：本示例中的 `query_database` 仅允许 SELECT 语句
3. **max_iterations**：设置最大迭代次数防止无限循环
4. **handle_parsing_errors**：生产环境建议开启，避免解析异常中断流程
