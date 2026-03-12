# 顺序链 - SimpleSequentialChain

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/11_chain_sequential.py` 中的代码。

---

## 完整代码

```python
from langchain.chains import SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)

chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model

chain2_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容: {text}"
)
chain2 = chain2_prompt | model

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

result = overall_chain.invoke("LangChain 是一个用于构建 LLM 应用的框架")
print("最终结果:", result)
```

---

## 代码逐行解析

### 第 1 行：导入 SimpleSequentialChain
```python
from langchain.chains import SimpleSequentialChain
```
- **SimpleSequentialChain**: 简单的顺序链，只有一个输入和一个输出

---

### 第 2-3 行：导入模型和提示词
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

---

### 第 5-10 行：创建模型实例
```python
model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)
```

---

### 第 12-14 行：创建第一条链（翻译链）
```python
chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model
```
- 输入 `{text}` → 翻译成英文

---

### 第 16-18 行：创建第二条链（总结链）
```python
chain2_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容: {text}"
)
chain2 = chain2_prompt | model
```
- 输入 `{text}` → 总结内容

---

### 第 20-21 行：创建顺序链
```python
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
```
- `chains`: 子链列表，按顺序执行
- `verbose=True`: 显示执行过程

---

### 第 23-24 行：执行链
```python
result = overall_chain.invoke("LangChain 是一个用于构建 LLM 应用的框架")
print("最终结果:", result)
```

---

## 执行流程

```
输入: "LangChain 是一个用于构建 LLM 应用的框架"
    ↓
┌─────────────────────────────────────┐
│ Chain 1: 翻译链                     │
│ "将以下内容翻译成英文: {text}"      │
│                                     │
│ "LangChain 是一个用于构建..."       │
│ ↓                                   │
│ "LangChain is a framework for..."  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Chain 2: 总结链                     │
│ "用一句话总结以下内容: {text}"      │
│                                     │
│ "LangChain is a framework for..."  │
│ ↓                                   │
│ "A framework for building LLM apps"│
└─────────────────────────────────────┘
    ↓
输出: "A framework for building LLM apps"
```

---

## 输出结果

```
> Entering new SimpleSequentialChain chain...
"LangChain 是一个用于构建 LLM 应用的框架"
"A framework for building LLM applications."

> Ending SimpleSequentialChain chain.
最终结果: A framework for building LLM applications.
```

---

## 核心概念

### SimpleSequentialChain
- 简单的顺序链
- 每个子链只有一个输入和一个输出
- 上一个链的输出自动作为下一个链的输入

### 参数说明

| 参数 | 说明 |
|------|------|
| chains | 子链列表 |
| verbose | 是否打印执行过程 |

### 限制
- 每个链只能有一个输入和一个输出
- 输入输出都是字符串

### 适用场景
- 简单的流水线任务
- 翻译 → 总结
- 提取 → 转换
