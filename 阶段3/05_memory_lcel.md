# 05_memory_lcel.py

## 功能说明

展示如何在 LCEL (LangChain Expression Language) 链中使用 Memory，以及使用 RunnableWithMessageHistory 实现更方便的记忆管理。

## 核心概念

### 1. 在 LCEL 中直接使用 Memory

通过 `config` 参数传递 memory：

```python
chain = prompt | llm
output = chain.invoke(
    {"question": "你好"},
    config={"memory": memory}
)
```

### 2. RunnableWithMessageHistory

LangChain 1.x 提供了 `RunnableWithMessageHistory` 来简化记忆管理：

```python
from langchain.runnables.history import RunnableWithMessageHistory

chat_with_history = RunnableWithMessageHistory(
    chat_chain,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)
```

关键参数：
- `chat_chain`: 基础对话链
- `get_session_history`: 返回每个会话的 memory
- `input_messages_key`: 用户输入的 key
- `history_messages_key`: 历史消息的 key

## 两种方式对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| config 传递 | 灵活控制 | 需要手动管理 |
| RunnableWithMessageHistory | 自动管理多会话 | 功能相对固定 |

## 运行示例

```bash
python 阶段3/05_memory_lcel.py
```

## 输出示例

```
=== LCEL + Memory 对话示例 ===

--- 对话 1 ---
用户: 你好，我叫李明
助手: 你好李明！很高兴认识你。有什么我可以帮助你的吗？

--- 对话 2 ---
用户: 我刚才告诉你我的名字是什么？
助手: 你刚才告诉我你叫李明。

--- 对话 3 ---
用户: 你知道我喜欢什么吗？
助手: 对不起，你还没有告诉我你喜欢什么。

=== 使用 RunnableWithMessageHistory ===

--- 会话 1 ---
用户: 我叫王芳，是一名教师
助手: 你好王芳！很高兴认识你。作为一名教师...

--- 会话 2（同一会话）---
用户: 我是谁？
助手: 你叫王芳，是一名教师，这是你之前告诉我的。

--- 会话 3（新会话，无历史）---
用户: 我是谁？
助手: 对不起，我不知道你是谁，因为我们还没有进行过对话。
```
