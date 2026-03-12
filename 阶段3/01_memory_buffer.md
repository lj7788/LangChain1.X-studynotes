# 01_memory_buffer.py

## 功能说明

`ChatMessageHistory` + `RunnableWithMessageHistory` 是 LangChain 1.x 推荐使用的对话记忆方式。

## 核心特性

- **会话隔离**: 通过 session_id 区分不同用户的对话历史
- **自动历史管理**: RunnableWithMessageHistory 自动将历史消息注入 prompt
- **灵活存储**: 可自定义存储方式（内存、数据库、文件等）

## 关键组件

### 1. ChatMessageHistory

用于存储对话消息，包含 `messages` 列表。

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("你好")
history.add_ai_message("你好，有什么可以帮你的？")
```

### 2. RunnableWithMessageHistory

将对话链与消息历史绑定的运行时组件。

```python
conversation = RunnableWithMessageHistory(
    prompt | llm,              # 对话链
    get_session_history,       # 获取会话历史的函数
    input_messages_key="question",      # 输入消息的 key
    history_messages_key="history"      # 历史消息的 key
)
```

### 3. MessagesPlaceholder

在 prompt 中占位，用于插入历史消息。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息插入位置
    ("human", "{question}")
])
```

## 关键参数

| 参数 | 说明 |
|------|------|
| session_id | 会话标识符，用于区分不同用户的对话 |
| input_messages_key | 用户输入在字典中的 key |
| history_messages_key | 历史消息在 prompt 中的变量名 |

## 运行示例

```bash
python 阶段3/01_memory_buffer.py
```

## 输出示例

```
=== 对话 1 ===
用户: 你好，我叫张三，请记住我的名字
助手: 你好张三！我已经记住你的名字了。有什么我可以帮助你的吗？

=== 对话 2 ===
用户: 我刚才告诉你我的名字是什么？
助手: 你刚才告诉我你叫张三。

=== 查看记忆内容 ===
记忆中的消息数量: 4
消息 1: HumanMessage - 你好，我叫张三，请记住我的名字...
消息 2: AIMessage - 你好张三！我已经记住你的名字了。...
消息 3: HumanMessage - 我刚才告诉你我的名字是什么？...
消息 4: AIMessage - 你刚才告诉我你叫张三。...

=== 对话 3 ===
用户: 我是谁？
助手: 你叫张三，这是你之前告诉我的。

=== 清除记忆后（新会话）===
用户: 我是谁？
助手: 对不起，我不知道你是谁，因为我们还没有进行过对话。
```
