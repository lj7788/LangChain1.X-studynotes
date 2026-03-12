# 提示词模板高级用法

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/07_prompt_advanced.py` 中的代码。

---

## 完整代码

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_prompt = SystemMessagePromptTemplate.from_template(
    "你是一个 {role}，用 {tone} 的语气回答问题"
)
human_prompt = HumanMessagePromptTemplate.from_template("{question}")

prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

result = prompt.invoke(
    {
        "role": "技术专家",
        "tone": "专业",
        "question": "什么是 LangChain?"
    }
)
print("格式化后的提示词:")
for msg in result.messages:
    print(f"- {msg.type}: {msg.content}")
```

---

## 代码逐行解析

### 第 1 行：导入 ChatPromptTemplate
```python
from langchain_core.prompts import ChatPromptTemplate
```
- **ChatPromptTemplate**: 聊天格式的提示词模板容器

---

### 第 2 行：导入消息模板
```python
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
```
- **SystemMessagePromptTemplate**: 系统消息模板
- **HumanMessagePromptTemplate**: 用户消息模板

---

### 第 4-6 行：创建系统消息模板
```python
system_prompt = SystemMessagePromptTemplate.from_template(
    "你是一个 {role}，用 {tone} 的语气回答问题"
)
```
- 定义 AI 的角色和行为
- 包含 `{role}` 和 `{tone}` 两个变量

---

### 第 8 行：创建用户消息模板
```python
human_prompt = HumanMessagePromptTemplate.from_template("{question}")
```
- 用户的提问模板
- 包含 `{question}` 变量

---

### 第 10 行：组合消息模板
```python
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
```
- 使用 `from_messages` 组合多个消息模板
- 顺序：系统消息 → 用户消息

---

### 第 12-17 行：调用 invoke 并输出
```python
result = prompt.invoke(
    {
        "role": "技术专家",
        "tone": "专业",
        "question": "什么是 LangChain?"
    }
)
print("格式化后的提示词:")
for msg in result.messages:
    print(f"- {msg.type}: {msg.content}")
```
- 使用 `invoke()` 方法返回 `ChatPromptValue` 对象
- 该对象有 `.messages` 属性，可以遍历每条消息
- 传入所有变量的值到字典中

---

## 执行流程

```
输入: prompt.invoke({"role": "技术专家", "tone": "专业", "question": "什么是 LangChain?"})
    ↓
┌─────────────────────────────────────┐
│ SystemMessagePromptTemplate        │
│ "你是一个 {role}，用 {tone}..."     │
│ ↓ invoke                            │
│ "你是一个 技术专家，用 专业的..."    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ HumanMessagePromptTemplate          │
│ "{question}"                        │
│ ↓ invoke                            │
│ "什么是 LangChain?"                 │
└─────────────────────────────────────┘
    ↓
组合: [SystemMessage, HumanMessage]
    ↓
输出: ChatPromptValue(messages=[...])
```

---

## 输出结果

```
格式化后的提示词:
- system: 你是一个 技术专家，用 专业的语气回答问题
- human: 什么是 LangChain?
```

---

## 核心概念

### 消息模板类型

| 类型 | 说明 |
|------|------|
| SystemMessagePromptTemplate | 系统消息，设置 AI 角色/行为 |
| HumanMessagePromptTemplate | 用户消息，用户的提问 |
| AIMessagePromptTemplate | AI 消息，预设 AI 回复 |

### format() vs invoke() 区别

| 方法 | 返回类型 | 适用场景 |
|------|---------|---------|
| `prompt.format(role="...", question="...")` | `str` 字符串 | 直接获取格式化后的文本 |
| `prompt.invoke({"role": "...", "question": "..."})` | `ChatPromptValue` 对象 | LCEL 链式调用，有 `.messages` 属性 |

### 组合多个消息
- 使用 `ChatPromptTemplate.from_messages()` 组合
- 支持任意数量的消息
- 顺序很重要（影响对话上下文）

### 优势
- 更灵活地控制对话格式
- 支持复杂的提示词结构
- 便于模板复用
