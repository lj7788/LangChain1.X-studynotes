# LCEL 链式调用

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/10_chain_llm.py` 中的代码。

---

## 完整代码

```python
from tools import make_model
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{question}")

model = make_model()

chain = prompt | model

result = chain.invoke({"question": "LangChain 是什么?"})
print("回答:", result.content)
```

---

## 代码逐行解析

### 第 1 行：导入工具函数
```python
from tools import make_model
```
- 使用 `tools.py` 中的 `make_model` 函数创建模型

---

### 第 2 行：导入提示词模板
```python
from langchain_core.prompts import ChatPromptTemplate
```
- **ChatPromptTemplate**: 聊天格式的提示词模板

---

### 第 4 行：创建提示词模板
```python
prompt = ChatPromptTemplate.from_template("{question}")
```
- 简单模板，只有一个变量 `question`

---

### 第 6 行：创建模型实例
```python
model = make_model()
```
- 使用 `make_model()` 创建 ChatOpenAI 模型

---

### 第 8 行：创建 LCEL 链
```python
chain = prompt | model
```
- 使用 `|` 管道操作符组合提示词模板和模型
- 这就是 LCEL（LangChain Expression Language）

---

### 第 10-11 行：执行链
```python
result = chain.invoke({"question": "LangChain 是什么?"})
print("回答:", result.content)
```
- `invoke` 返回 `AIMessage` 对象
- 使用 `.content` 获取文本内容

---

## 执行流程

```
输入: {"question": "LangChain 是什么?"}
    ↓
┌─────────────────────────────────────┐
│ ChatPromptTemplate                  │
│ "{question}"                        │
│ ↓ format                            │
│ "LangChain 是什么?"                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ ChatOpenAI 模型                     │
│ 返回 AIMessage 对象                 │
└─────────────────────────────────────┘
    ↓
输出: AIMessage(content="...")
```

---

## 输出结果

```
回答: LangChain 是一个用于构建 LLM 应用的框架...
```

---

## 核心概念

### LCEL (LangChain Expression Language)
- 使用管道操作符 `|` 组合组件
- 推荐的新方式，LangChain 1.x 官方推荐
- 语法简洁，灵活性高

### 返回值
- 返回 `AIMessage` 对象
- 使用 `.content` 获取文本内容
