# 06_langgraph_chatbot.py

## 功能说明

一个**生产级聊天机器人**的完整实现，整合了 LLM、Tools、记忆和多轮对话能力。

## 核心架构

```
START → call_llm ──(有tool_calls?)──▶ tools → call_llm（循环）
       │                                    │
       └────(无tool_calls)──────────────────┘→ END
```

## 关键组件

| 组件 | 作用 |
|------|------|
| `MessagesState` | 内置的消息状态类型（简化定义） |
| `add_messages` reducer | 自动将新消息追加到历史中（去重合并） |
| `bind_tools()` | 将工具集绑定为 function calling 格式 |
| `ToolMessage` | 工具执行结果的标准化消息类型 |
| `thread_id` | 会话隔离标识符（支持多用户并发） |

## 与 AgentExecutor 对比

| 维度 | AgentExecutor | LangGraph 自建 |
|------|--------------|----------------|
| 灵活性 | 固定循环模式 | 完全自定义流程 |
| 可观测性 | verbose 日志 | 可插入任意中间节点 |
| 记忆管理 | 外部 Memory 类 | 内置 State 管理 |
| 工具控制 | 全自动 | 手动控制何时调用 |

## 运行方式

```bash
python 阶段4/06_langgraph_chatbot.py
```

## 输出示例

```
============================================================
阶段4 - LangGraph 聊天机器人（含工具 + 记忆）
============================================================

开始对话！

👤 用户: 你好！你是谁？
🤖 助手: 你好！我是 AI 助手，可以帮助你...

👤 用户: 现在几点了？
  🔧 执行工具: get_current_time({})
🤖 助手: 现在是 2025年04月09日 16:30:00

👤 用户: 帮我计算 (256 + 128) * 4
  🔧 执行工具: calculate({'expression': '(256 + 128) * 4'})
🤖 助手: 计算结果是 1536

【对话统计】共 10 条消息
```

## 扩展方向

1. **持久化记忆**：使用 `checkpointer` 将 State 存储到 SQLite/Redis
2. **流式输出**：使用 `app.stream()` 替代 `app.invoke()`
3. **人机协作**：在循环中加入 `interrupt_before` 实现人工审批节点
