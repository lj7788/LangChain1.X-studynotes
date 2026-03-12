# 04_memory_persist.py

## 功能说明

展示如何保存和加载 ConversationBufferMemory 的对话历史，实现跨会话持久化记忆。

## 核心特性

- **JSON 格式存储**: 使用 JSON 文件保存对话消息
- **消息类型支持**: 支持 HumanMessage、AIMessage、SystemMessage
- **完整恢复**: 恢复后可以继续之前的对话

## 实现方法

### 保存记忆

```python
def save_memory(memory: ConversationBufferMemory):
    messages = []
    for msg in memory.chat_memory.messages:
        messages.append({
            "type": type(msg).__name__,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        })
    
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
```

### 加载记忆

```python
def load_memory(memory: ConversationBufferMemory):
    from langchain.schema import HumanMessage, AIMessage
    
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    for msg in messages:
        msg_type = msg["type"]
        content = msg["content"]
        
        if msg_type == "HumanMessage":
            memory.chat_memory.messages.append(HumanMessage(content=content))
        elif msg_type == "AIMessage":
            memory.chat_memory.messages.append(AIMessage(content=content))
```

## 扩展：使用 LangChain 内置方法

LangChain 1.x 还提供了其他持久化方式：

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageMemory(
        file_path="./data/chat_history.json",
        encoder=JsonMessageEncoder()
    )
)
```

## 运行示例

```bash
python 阶段3/04_memory_persist.py
```

## 输出示例

```
=== 场景1：创建新对话并保存 ===

用户: 你好！我叫张三
助手: 你好张三！很高兴认识你。有什么我可以帮助你的吗？

用户: 我喜欢机器学习
助手: 机器学习是一个非常有趣且有前景的领域！你是对哪个方向感兴趣呢...

=== 保存记忆 ===
记忆已保存到: /path/to/memory.json

=== 场景2：清除记忆（模拟重启应用）===
记忆已清除

=== 场景3：加载保存的记忆 ===
记忆已加载，共 4 条消息

=== 使用加载的记忆继续对话 ===
用户: 我是谁？
助手: 你叫张三，这是你之前告诉我的。

用户: 我刚才说我喜欢什么？
助手: 你刚才说你喜欢机器学习。
```
