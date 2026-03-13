# 17_memory_file_history.py

## 功能说明

使用 `FileChatMessageHistory` 实现对话历史的持久化存储。

## 核心组件

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

history = FileChatMessageHistory("data/chat_history.json")
```

## 特点

- ✅ 自动保存对话历史到 JSON 文件
- ✅ 跨会话持久化
- ✅ 代码简洁
- ✅ LangChain 1.x 推荐方式

## 运行示例

```bash
python 阶段3/17_memory_file_history.py
```
