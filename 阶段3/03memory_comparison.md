# LangChain Memory 三种方案对比

本文档对比 LangChain 中实现对话记忆的三种主要方式，帮助你根据实际需求选择合适的方案。

---

## 方案概览

| 方案 | 文件 | 核心类 | 记忆形式 |
|------|------|--------|----------|
| **方案一：Buffer** | `01_memory_buffer.py` | `ChatMessageHistory` + `RunnableWithMessageHistory` | 原始对话消息 |
| **方案二：Summary (旧版)** | `02_memory_summary_old.py` | `ConversationSummaryMemory` | AI 生成的摘要 |
| **方案三：Summary (新版)** | `02_memory_summary.py` | `ChatMessageHistory` + 自定义摘要函数 | 动态生成的摘要 |

---

## 方案一：ChatMessageHistory + RunnableWithMessageHistory

### 代码示例

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 定义 Prompt（使用 MessagesPlaceholder）
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{question}")
])

# 2. 创建会话历史存储
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 3. 创建 RunnableWithMessageHistory
conversation = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 4. 调用（自动注入历史）
response = conversation.invoke(
    {"question": "你好，我叫张三"},
    config={"configurable": {"session_id": "session_001"}}
)
```

### 工作原理

```
用户问题 → RunnableWithMessageHistory
                ↓
         get_session_history(session_id) 获取历史
                ↓
         prompt 注入 history 变量
                ↓
         LLM 生成回复
                ↓
         自动保存到 ChatMessageHistory
```

### 优点

- ✅ **多会话支持**：通过 `session_id` 隔离不同会话
- ✅ **自动管理**：`RunnableWithMessageHistory` 自动加载和保存历史
- ✅ **完整保留**：保存所有对话细节，信息不丢失
- ✅ **无弃用风险**：使用官方推荐的 1.x API

### 缺点

- ❌ **Token 消耗高**：完整保存所有对话，context 越来越长
- ❌ **长对话瓶颈**：超出 LLM 上下文限制后效果下降

### 适用场景

- 短对话（少于 10 轮）
- 需要多会话隔离
- 需要保留完整对话细节

---

## 方案二：ConversationSummaryMemory（旧版）

### 代码示例

```python
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.runnables import RunnablePassthrough

# 1. 初始化摘要记忆
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="question",
    return_messages=True
)

# 2. 手动加载记忆
def load_memory(inputs):
    return memory.load_memory_variables(inputs)["chat_history"]

# 3. 构建链
chain = (
    RunnablePassthrough.assign(chat_history=load_memory)
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 对话（需手动保存）
def chat_with_memory(question):
    response = chain.invoke({"question": question})
    memory.save_context(
        inputs={"question": question},
        outputs={"output": response}
    )
    return response
```

### 工作原理

```
新对话 → 加载旧摘要 → 合并新对话 → LLM 生成新摘要 → 保存
```

每次调用 `save_context()` 时：
1. 将新对话追加到缓冲区
2. 调用 LLM 生成更新的摘要
3. 存储新的摘要（覆盖旧的）

### 优点

- ✅ **节省 Token**：摘要比完整对话短很多
- ✅ **内置实现**：代码量少，使用简单
- ✅ **自动摘要**：LLM 自动生成摘要

### 缺点

- ❌ **单会话**：不支持多会话隔离
- ❌ **手动保存**：需要手动调用 `save_context()`
- ⚠️ **弃用风险**：LangChain 未来版本可能弃用此类
- ❌ **可能丢细节**：摘要可能丢失一些对话细节

### 适用场景

- 单会话的简单 chatbot
- 长对话（需要节省 token）
- 快速原型开发

---

## 方案三：ChatMessageHistory + 自定义摘要（新版）

### 代码示例

```python
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 会话历史存储
store = {}
def get_session_history(session_id: str = "default"):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. 自定义摘要生成
def generate_summary(history):
    # 拼接原始对话
    msg_text = "\n".join([
        f"用户: {m.content}" if m.type == "human" else f"助手: {m.content}"
        for m in history.messages
    ])
    # LLM 生成摘要
    summary_prompt = PromptTemplate.from_template("...")
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"messages": msg_text})

# 3. 对话函数
def chat_with_summary(question, session_id="default"):
    response = chain.invoke({"question": question, "session_id": session_id})
    history = get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(response)
    return response, generate_summary(history)
```

### 工作原理

```
用户问题
    ↓
ChatMessageHistory（保存原始对话）
    ↓
每次调用 generate_summary() → LLM 实时生成摘要
    ↓
prompt 使用摘要
    ↓
LLM 生成回复
```

### 优点

- ✅ **多会话支持**：通过 `session_id` 隔离
- ✅ **消除警告**：不使用旧版类，无弃用风险
- ✅ **灵活可控**：摘要生成逻辑可自定义
- ✅ **保留原始对话**：`ChatMessageHistory` 完整保存
- ✅ **实时生成**：每次对话时动态生成摘要

### 缺点

- ❌ **代码量多**：需要自己实现摘要逻辑
- ❌ **额外 LLM 调用**：每次对话都要调用 LLM 生成摘要
- ❌ **实时性**：不能像方案二那样增量更新摘要

### 适用场景

- 长对话（需要摘要节省 token）
- 多会话场景
- 需要消除 LangChain 弃用警告
- 需要自定义摘要生成逻辑

---

## 核心对比表

| 特性 | 方案一 (Buffer) | 方案二 (Summary旧版) | 方案三 (Summary新版) |
|------|---------------|---------------------|---------------------|
| **核心类** | `ChatMessageHistory` + `RunnableWithMessageHistory` | `ConversationSummaryMemory` | `ChatMessageHistory` + 自定义函数 |
| **记忆形式** | 原始消息 | AI 摘要 | 原始消息 + 动态摘要 |
| **多会话** | ✅ | ❌ | ✅ |
| **Token 消耗** | 高 | 低 | 中 |
| **信息完整度** | 100% | 取决于摘要质量 | 100%（原始消息保留） |
| **自动保存** | ✅ | ❌ | ❌ |
| **弃用风险** | 无 | 有 | 无 |
| **代码复杂度** | 低 | 低 | 高 |
| **灵活性** | 中 | 低 | 高 |

---

## 性能对比（以 5 轮对话为例）

假设每轮对话平均 100 tokens：

| 方案 | 历史大小（Token） | 备注 |
|------|-----------------|------|
| 方案一 | 5 × 100 × 2 = **1000** | 所有消息 × 2（用户+AI） |
| 方案二 | **~150** | 摘要通常 100-200 tokens |
| 方案三 | 原始: **1000** + 摘要: **~150** | 两者都存储，但只用摘要 |

---

## 选择建议

```
┌─────────────────────────────────────────────────────────────┐
│                      如何选择？                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  对话轮次少（<10轮）？                                       │
│      │                                                       │
│      ├── ✅ 是 → 方案一（Buffer）最简单                      │
│      │                                                       │
│      └── ❌ 否 → 需要多会话？                                │
│              │                                               │
│              ├── ✅ 是 → 方案三（新版摘要）                  │
│              │                                               │
│              └── ❌ 否 → 对话很长？                          │
│                      │                                       │
│                      ├── ✅ 是 → 方案二或三                  │
│                      │                                       │
│                      └── ❌ 否 → 方案一                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 快速选择

| 场景 | 推荐方案 |
|------|---------|
| 简单 chatbot，原型开发 | 方案一 |
| 长对话，单会话 | 方案二 |
| 长对话，多会话 | 方案三 |
| 生产环境，无弃用风险 | 方案三 |
| 需要自定义摘要逻辑 | 方案三 |

---

## 代码文件对应

| 文档 | 文件 |
|------|------|
| [01_memory_buffer.py](01_memory_buffer.py) | 方案一 |
| [02_memory_summary_old.py](02_memory_summary_old.py) | 方案二 |
| [02_memory_summary.py](02_memory_summary.py) | 方案三 |

---

## 总结

- **方案一**：最简单，适合短对话和多会话
- **方案二**：最直接，但有弃用风险
- **方案三**：最灵活，推荐用于生产环境

根据你的具体需求（对话长度、会话数量、灵活性要求）选择合适的方案。
