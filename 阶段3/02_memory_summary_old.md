# 02_memory_summary_old.py - 旧版方案

## 功能说明

本文件使用 LangChain 1.2.11 中的 `ConversationSummaryMemory` 类来实现对话摘要记忆。

这是 LangChain 传统的实现方式，使用内置的摘要记忆类来自动管理对话摘要。

## 核心特性

- **自动摘要**：使用 LLM 将对话历史压缩成摘要
- **内置实现**：使用 LangChain 内置的 `ConversationSummaryMemory` 类
- **简单直接**：封装好的类，代码量少
- **单会话**：不支持多会话隔离

---

## 代码执行过程详解

### 第一部分：导入

```python
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

| 导入 | 作用 |
|------|------|
| `ConversationSummaryMemory` | 对话摘要记忆类（核心） |
| `PromptTemplate` | 构建提示词模板 |
| `RunnablePassthrough` | 在 LCEL 链中传递数据 |
| `StrOutputParser` | 解析 LLM 输出为字符串 |

**注意**：`ConversationSummaryMemory` 从 `langchain_classic.memory` 导入，这是 LangChain 1.2.11 的正确路径。

---

### 第二部分：初始化 LLM 和记忆

```python
llm = make_ollama()

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)
```

**参数说明**：

| 参数 | 作用 |
|------|------|
| `llm` | 用于生成摘要的 LLM（同一个 llm） |
| `memory_key` | 在 prompt 中引用记忆的变量名 `{chat_history}` |
| `input_key` | 用户输入的键名（用于保存上下文） |
| `return_messages` | True 返回消息列表，False 返回字符串 |

**初始化后的 memory 对象**：
```python
memory = {
    buffer: "摘要内容...",      # 对话摘要（字符串）
    messages: [...],            # 原始消息列表（如果 return_messages=True）
}
```

---

### 第三部分：对话 Prompt

```python
prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话历史回答用户的问题。

对话历史:
{chat_history}

用户问题: {question}

回答:"""
)
```

**模板变量**：
- `{chat_history}` - 由 memory 加载的对话历史
- `{question}` - 当前用户问题

---

### 第四部分：加载记忆的函数

```python
def load_memory(inputs):
    """加载对话记忆的函数"""
    return memory.load_memory_variables(inputs)["chat_history"]
```

**执行流程**：
```
调用 memory.load_memory_variables({"question": "你好"})
    ↓
内部调用 memory.load_memory_variables(inputs)
    ↓
返回: {"chat_history": "这里是摘要内容..."}
    ↓
取出 ["chat_history"] → "这里是摘要内容..."
```

**`load_memory_variables()` 的工作原理**：
1. 检查是否有新对话需要生成摘要
2. 如果有，调用 LLM 生成新摘要
3. 将新摘要与旧摘要合并
4. 返回包含 memory_key 的字典

---

### 第五部分：构建 LCEL 链

```python
chain = (
    RunnablePassthrough.assign(chat_history=load_memory)
    | prompt
    | llm
    | StrOutputParser()
)
```

**链的执行流程**：

```
┌─────────────────────────────────────────────────────────────┐
│  输入: {"question": "我喜欢打篮球"}                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  RunnablePassthrough.assign(chat_history=load_memory)       │
│                                                             │
│  load_memory({"question": "我喜欢打篮球"})                  │
│      ↓                                                     │
│  memory.load_memory_variables()                            │
│      ↓                                                     │
│  返回: {"chat_history": "张三是一名软件工程师..."}           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  prompt (填充模板)                                           │
│                                                             │
│  "你是一个友好的助手。请根据对话历史回答用户的问题。          │
│                                                             │
│  对话历史:                                                  │
│  张三是一名软件工程师...                                     │
│                                                             │
│  用户问题: 我喜欢打篮球                                      │
│                                                             │
│  回答:"                                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  llm (调用大语言模型)                                        │
│                                                             │
│  LLM 生成回复: "太棒了！..."                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  StrOutputParser()                                           │
│                                                             │
│  解析输出为字符串                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
返回: "太棒了！..."
```

---

### 第六部分：对话函数（手动保存记忆）

```python
def chat_with_memory(question):
    """调用链 + 手动保存记忆"""
    
    # 步骤1：调用链获取回复
    response = chain.invoke({"question": question})
    
    # 步骤2：手动保存对话到摘要记忆中
    memory.save_context(
        inputs={"question": question},
        outputs={"output": response}
    )
    
    return response
```

**执行流程**：

```
对话: "我喜欢打篮球"
AI 回复: "太棒了！"

→ chain.invoke() → "太棒了！"
    ↓
→ memory.save_context(
    inputs={"question": "我喜欢打篮球"},
    outputs={"output": "太棒了！"}
  )
    ↓
内部处理：
  1. 将新对话追加到缓冲区
  2. 检查是否需要生成新摘要
  3. 如果需要，调用 LLM 生成摘要
  4. 更新 memory.buffer 中的摘要

→ 返回: "太棒了！"
```

**`save_context()` 的内部工作原理**：

```
inputs = {"question": "我喜欢打篮球"}
outputs = {"output": "太棒了！"}

↓ 追加到缓冲区
buffer_messages = [
    "之前的人类: ...",
    "之前的AI: ...",
    "Human: 我喜欢打篮球",
    "AI: 太棒了！"
]

↓ 判断是否需要生成摘要
如果累积的对话足够长 → 调用 LLM 生成新摘要

↓ 更新摘要
buffer = "更新后的摘要..."
```

---

### 第七部分：测试对话

```python
# 对话 1
question1 = "我叫张三，是一名软件工程师。我喜欢编程和读书。"
response1 = chat_with_memory(question1)

# 此时 memory.buffer 内容：
# "人类介绍自己是张三，是一名软件工程师，喜欢编程和读书。"
```

```python
# 对话 2
question2 = "你喜欢什么运动？我喜欢打篮球。"
response2 = chat_with_memory(question2)

# 此时 memory.buffer 内容：
# "张三是一名软件工程师，喜欢编程、读书和打篮球..."
```

```python
# 查看记忆
print(memory.buffer)

# 输出：摘要形式的对话历史
```

---

## 运行示例

```bash
python 阶段3/02_memory_summary_old.py
```

### 完整输出

```
=== 对话 1 ===
用户: 我叫张三，是一名软件工程师。我喜欢编程和读书。
助手: 你好！我是很高兴认识你，张三大哥！...

=== 对话 2 ===
用户: 你喜欢什么运动？我喜欢打篮球。
助手: 我不太擅长运动，但很高兴你喜欢打篮球！...

=== 查看摘要记忆 ===
记忆内容:
Here is the updated summary:

Zhang San, a software engineer who enjoys programming and reading, introduces himself...

=== 对话 3 ===
用户: 总结一下我告诉你的关于我自己的信息
助手: 你告诉我的关于你自己的信息是：你是一个软件工程师...

=== 对话 4 ===
用户: 我是谁？喜欢什么？
助手: 你是张三，一个软件工程师。你的兴趣包括编程和阅读。
```

---

## 与新版方案对比

| 特性 | 旧版 (02_memory_summary_old.py) | 新版 (02_memory_summary.py) |
|------|--------------------------------|----------------------------|
| 核心类 | `ConversationSummaryMemory` | `ChatMessageHistory` + 自定义函数 |
| 多会话 | ❌ 不支持 | ✅ 支持 |
| 弃用警告 | ⚠️ 可能有 | ✅ 无 |
| 摘要存储 | 类内部自动存储 | 实时生成，不存储 |
| 灵活性 | 低（封装好的类） | 高（可自定义摘要逻辑） |
| 代码量 | 较少 | 较多 |

---

## 适用场景

- ✅ 短对话（不需要复杂的多会话管理）
- ✅ 快速原型开发
- ✅ 简单的单会话 chatbot
- ⚠️ 警告：LangChain 未来版本可能弃用此类

---

## 注意事项

1. **手动保存**：在 1.2.x 版本中，必须手动调用 `save_context()` 保存对话
2. **弃用警告**：新版 LangChain 推荐使用 `ChatMessageHistory` + 自定义摘要的方案
3. **单会话**：不支持多会话隔离，所有对话共享同一个记忆
