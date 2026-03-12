# 顺序链 - LCEL 顺序链

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/11_chain_sequential.py` 中的代码。

---

## 完整代码

```python
from tools import make_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

model = make_model()

chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model

def extract_text(input):
    return {"text": input.content}

chain2_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容: {text}"
)
chain2 = chain2_prompt | model

overall_chain = chain1 | RunnableLambda(extract_text) | chain2

result = overall_chain.invoke({"text": "LangChain 是一个用于构建 LLM 应用的框架"})
print("最终结果:", result.content)
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

---

### 第 3 行：导入 RunnableLambda
```python
from langchain_core.runnables import RunnableLambda
```
- 用于创建自定义转换函数

---

### 第 5 行：创建模型实例
```python
model = make_model()
```

---

### 第 7-9 行：创建第一条链（翻译链）
```python
chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model
```
- 输入 `{text}` → 翻译成英文

---

### 第 11-13 行：创建转换函数
```python
def extract_text(input):
    return {"text": input.content}
```
- 将 `AIMessage` 对象转换为字典格式
- 因为 chain1 返回的是 `AIMessage`，需要提取 `.content` 并包装成字典

---

### 第 15-17 行：创建第二条链（总结链）
```python
chain2_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容: {text}"
)
chain2 = chain2_prompt | model
```
- 输入 `{text}` → 总结内容

---

### 第 19 行：组合成顺序链
```python
overall_chain = chain1 | RunnableLambda(extract_text) | chain2
```
- 使用 `|` 管道操作符依次连接
- 中间需要 `RunnableLambda` 转换格式

---

### 第 21-22 行：执行链
```python
result = overall_chain.invoke({"text": "LangChain 是一个用于构建 LLM 应用的框架"})
print("最终结果:", result.content)
```
- 返回 `AIMessage` 对象，使用 `.content` 获取文本

---

## 执行流程

```
输入: {"text": "LangChain 是一个用于构建 LLM 应用的框架"}
    ↓
┌─────────────────────────────────────┐
│ chain1: 翻译链                       │
│ "LangChain 是一个用于构建..."        │
│ ↓ 翻译成英文                         │
│ "LangChain is a framework..."       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ RunnableLambda: 格式转换             │
│ AIMessage → {"text": "..."}         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ chain2: 总结链                       │
│ 总结英文内容                         │
│ "A framework for building..."       │
└─────────────────────────────────────┘
    ↓
输出: AIMessage(content="A framework for building...")
```

---

## 输出结果

```
最终结果: A framework for building LLM applications.
```

---

## 核心概念

### LCEL 顺序链
- 使用管道操作符 `|` 连接多个链
- 前一个链的输出自动传递给下一个链
- 需要注意输出格式的转换

### RunnableLambda
- 用于创建自定义转换函数
- 可以转换输入/输出格式
- 在格式不匹配时非常有用

### vs SimpleSequentialChain

| 特性 | SimpleSequentialChain | LCEL 顺序链 |
|------|---------------------|------------|
| 语法 | 类封装 | 管道操作符 |
| 格式转换 | 自动 | 需手动处理 |
| 灵活性 | 较低 | 高 |
| 推荐 | - | ✅ |

---

## 注意事项

- LCEL 顺序链需要手动处理格式转换
- `chain1` 返回 `AIMessage`，需要转换为字典才能传给 `chain2`
- 使用 `RunnableLambda` 可以灵活处理各种格式转换
