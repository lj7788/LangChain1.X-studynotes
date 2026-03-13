# 04_memory_persist.py

## 功能说明

展示如何保存和加载对话历史的持久化存储。

## 核心特性

- **JSON 序列化**: 将消息保存为 JSON 格式
- **跨会话持久化**: 应用重启后可以加载历史记忆
- **消息类型保留**: 正确处理 HumanMessage、AIMessage 等不同类型

## 关键组件

### 保存函数

```python
def save_memory(chat_history: ChatMessageHistory):
    messages = []
    for msg in chat_history.messages:
        messages.append({
            "type": type(msg).__name__,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        })
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
```

### 加载函数

```python
def load_memory() -> ChatMessageHistory:
    memory = ChatMessageHistory()
    # 加载并重建消息...
    return memory
```

## 运行示例

```bash
python 阶段3/04_memory_persist.py
```

## 输出示例

```
=== 场景1：创建新对话并保存 ===

用户: 你好！我叫张三
助手: 你好张三！很高兴认识你...

用户: 我喜欢机器学习
助手: 机器学习是一个非常有趣且有前景的领域...

=== 保存记忆 ===
记忆已保存到: 阶段3/data/memory.json

=== 场景2：清除记忆（模拟重启应用）===
记忆已清除

=== 场景3：加载保存的记忆 ===
记忆已加载，共 4 条消息

=== 使用加载的记忆继续对话 ===

用户: 我是谁？
助手: 你叫张三，这是你之前告诉我的。

用户: 我刚才说我喜欢什么？
助手: 你刚才告诉我你喜欢机器学习。
```
