# 02_memory_summary.py

## 功能说明

`ConversationSummaryMemory` 使用 LLM 自动总结对话历史，节省 token 消耗。适合长对话场景。

## 核心特性

- **自动摘要**: LLM 自动将对话历史压缩为简洁摘要
- **节省 Token**: 相比完整对话历史，摘要大幅减少 token 使用
- **保留关键信息**: 自动提取用户的关键信息（姓名、职业、喜好等）

## 关键组件

### ConversationSummaryMemory

```python
from langchain_classic.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)
```

### 关键方法

| 方法 | 说明 |
|------|------|
| load_memory_variables() | 加载记忆变量 |
| save_context() | 保存对话上下文 |

## 运行示例

```bash
python 阶段3/02_memory_summary.py
```

## 输出示例

```
=== 对话 1 ===
用户: 我叫张三，是一名软件工程师。我喜欢编程和读书。
助手: 你好张三！很高兴认识你。作为一名软件工程师，编程和读书都是很好的爱好...

=== 查看摘要记忆 ===
记忆内容:
用户和助手进行了初次对话。用户自我介绍叫张三，是一名软件工程师，喜欢编程和读书。助手表示很高兴认识用户。

=== 对话 2 ===
用户: 你喜欢什么运动？我喜欢打篮球。
助手: 这真是太棒了！篮球是一项非常受欢迎的运动...

=== 对话 3 ===
用户: 总结一下我告诉你的关于我自己的信息
助手: 根据我们的对话，你告诉我以下关于你自己的信息：
1. 名字：张三
2. 职业：软件工程师
3. 爱好：编程、读书、打篮球
```
