# LLMChain 链式调用

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/10_chain_llm.py` 中的代码。

---

## 完整代码

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{question}")

model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)

chain = LLMChain(llm=model, prompt=prompt)

result = chain.invoke({"question": "LangChain 是什么?"})
print("回答:", result["text"])
```

---

## 代码逐行解析

### 第 1 行：导入 LLMChain
```python
from langchain.chains import LLMChain
```
- **LLMChain**: LangChain 提供的经典链式调用类
- 将提示词模板和模型组合成链

---

### 第 2 行：导入 ChatOpenAI
```python
from langchain_openai import ChatOpenAI
```
- 对话模型

---

### 第 3 行：导入提示词模板
```python
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
```

---

### 第 5 行：创建提示词模板
```python
prompt = ChatPromptTemplate.from_template("{question}")
```
- 简单模板，只有一个变量 `question`

---

### 第 7-12 行：创建模型实例
```python
model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)
```

---

### 第 14 行：创建 LLMChain
```python
chain = LLMChain(llm=model, prompt=prompt)
```
- `llm`: 语言模型
- `prompt`: 提示词模板
- 组合成一条链

---

### 第 16-17 行：执行链
```python
result = chain.invoke({"question": "LangChain 是什么?"})
print("回答:", result["text"])
```
- `invoke` 返回字典，包含 `text` 键

---

## 执行流程

```
输入: {"question": "LangChain 是什么?"}
    ↓
LLMChain
    ↓
┌─────────────────────────────────────┐
│ 1. prompt.format() → 格式化提示词   │
│ 2. model.invoke() → 调用模型        │
│ 3. 返回 {"question": "...", "text": "..."} │
└─────────────────────────────────────┘
    ↓
输出: {"question": "LangChain 是什么?", "text": "LangChain 是..."}
```

---

## 输出结果

```
回答: LangChain 是一个用于构建 LLM 应用的框架...
```

---

## 核心概念

### LLMChain
- LangChain 提供的经典链式调用类
- 简化了提示词 + 模型的组合
- 自动传递输入到提示词，输出到模型

### 返回值
- 返回包含所有变量的字典
- 模型输出在 `text` 键中

### vs LCEL

| 特性 | LLMChain | LCEL |
|------|---------|------|
| 语法 | 类封装 | 管道操作符 `|` |
| 灵活性 | 较低 | 高 |
| 推荐 | 简单场景 | 复杂场景 |

### LCEL 等价写法

```python
chain = prompt | model

result = chain.invoke({"question": "LangChain 是什么?"})
# result 是 AIMessage 对象
```
