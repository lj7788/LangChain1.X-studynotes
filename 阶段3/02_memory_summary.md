# 02_memory_summary.py - 新版混合方案

## 功能说明

本文件采用**混合方案**：结合 `ChatMessageHistory` 存储原始对话 + 自定义 LLM 动态生成摘要。

这是在 LangChain 1.2.11 中推荐的现代写法，完全消除弃用警告，同时支持多会话。

## 核心特性

- **多会话支持**：通过 `session_id` 隔离不同会话
- **动态摘要**：每次对话时实时生成摘要，不额外存储
- **消除警告**：不使用旧版 `ConversationSummaryMemory` 类
- **灵活可控**：摘要生成逻辑可自定义

---

## 代码执行过程详解

### 第一部分：导入和初始化

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
```

| 导入 | 作用 |
|------|------|
| `PromptTemplate` | 构建提示词模板 |
| `RunnablePassthrough` | 在 LCEL 链中传递数据 |
| `StrOutputParser` | 解析 LLM 输出为字符串 |
| `BaseChatMessageHistory` | 聊天历史基类 |
| `ChatMessageHistory` | 存储原始对话消息 |

```python
llm = make_ollama()
```

初始化 LLM（大语言模型）实例。

---

### 第二部分：会话历史存储

```python
store = {}

def get_session_history(session_id: str = "default") -> BaseChatMessageHistory:
    """获取/创建会话历史"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

**执行流程**：
```
调用 get_session_history("test")
    ↓
检查 store 中是否存在 "test" 键
    ↓
不存在 → 创建新的 ChatMessageHistory() → 存入 store
    ↓
返回 ChatMessageHistory 实例
```

**核心数据结构**：
```python
store = {
    "session_001": ChatMessageHistory(messages=[...]),
    "session_002": ChatMessageHistory(messages=[...]),
    "default": ChatMessageHistory(messages=[...])
}
```

---

### 第三部分：自定义摘要生成函数

```python
def generate_summary(history: BaseChatMessageHistory) -> str:
    """生成对话历史的简洁摘要"""
    if not history.messages:
        return "暂无对话历史"
    
    # 步骤1：拼接原始对话
    msg_text = "\n".join([
        f"用户: {msg.content}" if msg.type == "human" else f"助手: {msg.content}"
        for msg in history.messages
    ])
    
    # 步骤2：构建摘要 Prompt
    summary_prompt = PromptTemplate.from_template("""
    请简洁总结以下对话，保留用户的关键信息（姓名、职业、喜好等）：
    {messages}
    总结：
    """)
    
    # 步骤3：生成摘要
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"messages": msg_text}).strip()
```

**执行流程**：

```
用户说了: "我叫张三，是一名软件工程师"
AI 回复: "你好，很高兴认识你"

→ history.messages = [
    HumanMessage(content="我叫张三..."),
    AIMessage(content="你好...")
  ]

→ msg_text = """
    用户: 我叫张三，是一名软件工程师
    助手: 你好，很高兴认识你
  """

→ summary_prompt 填充后:
    "请简洁总结以下对话，保留用户的关键信息...
     用户: 我叫张三，是一名软件工程师
     助手: 你好，很高兴认识你
     总结："

→ LLM 生成摘要: "用户张三是一名软件工程师"

→ 返回: "用户张三是一名软件工程师"
```

---

### 第四部分：对话 Prompt

```python
prompt = PromptTemplate.from_template(
    """你是一个友好的助手。请根据对话摘要回答用户问题。

对话摘要:
{chat_summary}

用户问题: {question}

回答:"""
)
```

**模板变量**：
- `{chat_summary}` - 由 `generate_summary()` 生成的摘要
- `{question}` - 当前用户问题

---

### 第五部分：加载摘要的函数

```python
def load_summary(inputs):
    # 获取会话历史
    history = get_session_history(inputs.get("session_id", "default"))
    
    # 生成摘要
    summary = generate_summary(history)
    
    # 返回合并后的输入
    return {
        "chat_summary": summary,
        "question": inputs["question"]
    }
```

**执行流程**：
```
输入: {"question": "我喜欢打篮球", "session_id": "test"}

→ 调用 get_session_history("test")
→ 获取 history (假设已有2条消息)
→ 调用 generate_summary(history) → "张三是一名软件工程师..."

→ 返回:
{
    "chat_summary": "张三是一名软件工程师...",
    "question": "我喜欢打篮球"
}
```

---

### 第六部分：构建 LCEL 链

```python
chain = (
    RunnablePassthrough.assign(chat_summary=load_summary)
    | prompt
    | llm
    | StrOutputParser()
)
```

**链的执行流程**：

```
┌─────────────────────────────────────────────────────────────┐
│  输入: {"question": "你喜欢什么运动？", "session_id": "test"} │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  RunnablePassthrough.assign(chat_summary=load_summary)     │
│                                                             │
│  load_summary() 被调用，返回:                                │
│  {                                                          │
│      "chat_summary": "张三是一名软件工程师...",             │
│      "question": "你喜欢什么运动？",                         │
│      "session_id": "test"                                   │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  prompt (填充模板)                                           │
│                                                             │
│  生成完整提示词:                                             │
│  "你是一个友好的助手。请根据对话摘要回答...                  │
│                                                             │
│  对话摘要:                                                  │
│  张三是一名软件工程师...                                     │
│                                                             │
│  用户问题: 你喜欢什么运动？                                   │
│                                                             │
│  回答:"                                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  llm (调用大语言模型)                                        │
│                                                             │
│  LLM 生成回复: "我不太擅长运动..."                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  StrOutputParser()                                           │
│                                                             │
│  解析输出为字符串: "我不太擅长运动..."                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
返回: "我不太擅长运动..."
```

---

### 第七部分：对话函数

```python
def chat_with_summary(question, session_id="default"):
    # 1. 获取回复
    response = chain.invoke({"question": question, "session_id": session_id})
    
    # 2. 保存对话到历史
    history = get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(response)
    
    # 3. 返回回复和当前摘要
    return response, generate_summary(history)
```

**执行流程**：

```
对话 1: "我叫张三，是一名软件工程师"

→ chain.invoke() → 返回 "你好，张三！..."
→ history.add_user_message("我叫张三...")
→ history.add_ai_message("你好，张三！...")
→ generate_summary(history) → "用户张三是一名软件工程师"

→ 返回: ("你好，张三！...", "用户张三是一名软件工程师")
```

---

## 运行示例

```bash
python 阶段3/02_memory_summary.py
```

### 完整输出

```
=== 对话 1 ===
用户: 我叫张三，是一名软件工程师。我喜欢编程和读书。
助手: 你好！我是很高兴认识你，张三大哥！作为一个软件工程师...

=== 对话 2 ===
用户: 你喜欢什么运动？我喜欢打篮球。
助手: 我不太擅长运动，但很高兴你喜欢打篮球！...

=== 查看摘要记忆 ===
记忆内容:
用户张三是一名软件工程师，喜欢编程、阅读和打篮球。...

=== 对话 3 ===
用户: 总结一下我告诉你的关于我自己的信息
助手: 你告诉我的关于你自己的信息是：你是一个软件工程师...

=== 对话 4 ===
用户: 我是谁？喜欢什么？
助手: 你是张三，一个软件工程师。你的兴趣包括编程和阅读。
```

---

## 与旧版方案对比

| 特性 | 旧版 (02_memory_summary_old.py) | 新版 (02_memory_summary.py) |
|------|--------------------------------|----------------------------|
| 核心类 | `ConversationSummaryMemory` | `ChatMessageHistory` + 自定义函数 |
| 多会话 | ❌ 不支持 | ✅ 支持 |
| 弃用警告 | ⚠️ 可能有 | ✅ 无 |
| 摘要存储 | 类内部存储 | 实时生成，不存储 |
| 灵活性 | 低（封装好的类） | 高（可自定义摘要逻辑） |

---

## 适用场景

- ✅ 长对话（需要摘要节省 token）
- ✅ 多会话场景
- ✅ 需要消除 LangChain 弃用警告
- ✅ 需要自定义摘要生成逻辑
