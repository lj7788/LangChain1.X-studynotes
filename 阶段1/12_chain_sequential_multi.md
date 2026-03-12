# 并行链 - RunnableParallel

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/12_chain_sequential_multi.py` 中的代码。

---

## 完整代码

```python
from tools import make_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
model = make_model()

chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model | output_parser

chain2_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成法语: {text}"
)
chain2 = chain2_prompt | model | output_parser

overall_chain = RunnableParallel(
    english_text=chain1,
    french_text=chain2
)

result = overall_chain.invoke({"text": "LangChain 是一个 LLM 应用框架"})
print("英文:", result["english_text"])
print("法语:", result["french_text"])
```

---

## 代码逐行解析

### 第 1 行：导入工具函数
```python
from tools import make_model
```
- 使用 `tools.py` 中的 `make_model` 函数创建模型

---

### 第 2-4 行：导入 LCEL 组件
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
```
- `ChatPromptTemplate`: 提示词模板
- `RunnableParallel`: 并行执行多个链
- `StrOutputParser`: 将输出解析为字符串

---

### 第 6 行：创建输出解析器
```python
output_parser = StrOutputParser()
```
- 将 `AIMessage` 对象转换为字符串

---

### 第 8 行：创建模型实例
```python
model = make_model()
```

---

### 第 10-12 行：创建第一条链（英文翻译）
```python
chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model | output_parser
```
- 提示词 → 模型 → 输出解析器
- 最终输出字符串

---

### 第 14-16 行：创建第二条链（法语翻译）
```python
chain2_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成法语: {text}"
)
chain2 = chain2_prompt | model | output_parser
```

---

### 第 18-21 行：创建并行链
```python
overall_chain = RunnableParallel(
    english_text=chain1,
    french_text=chain2
)
```
- 同时执行两条链
- 返回字典，包含两个结果

---

### 第 23-25 行：执行链
```python
result = overall_chain.invoke({"text": "LangChain 是一个 LLM 应用框架"})
print("英文:", result["english_text"])
print("法语:", result["french_text"])
```

---

## 执行流程

```
输入: {"text": "LangChain 是一个 LLM 应用框架"}
    ↓
┌─────────────────────────────────────┐
│ RunnableParallel                    │
│                                     │
│ ┌─────────────┐   ┌─────────────┐  │
│ │ chain1      │   │ chain2      │  │
│ │ 英文翻译    │   │ 法语翻译    │  │
│ └─────────────┘   └─────────────┘  │
│      ↓                 ↓            │
│ English text     French text        │
└─────────────────────────────────────┘
    ↓
输出: {
    "english_text": "...",
    "french_text": "..."
}
```

---

## 输出结果

```
英文: LangChain is an LLM application framework.
法语: LangChain est un framework d'application LLM.
```

---

## 核心概念

### RunnableParallel
- 并行执行多个链
- 所有链同时运行，提高效率
- 返回字典，包含各个链的结果

### StrOutputParser
- 将 `AIMessage` 对象转换为纯字符串
- 简化输出处理

### LCEL 管道组合
- `prompt | model | output_parser` 组合成完整链
- 语法简洁直观

### vs SequentialChain

| 特性 | SequentialChain | RunnableParallel |
|------|-----------------|------------------|
| 执行方式 | 顺序执行 | 并行执行 |
| 效率 | 较低 | 高 |
| 适用场景 | 依赖关系 | 独立任务 |

---

## 注意事项

- `RunnableParallel` 适用于相互独立的链
- 如果链之间有依赖关系，需要使用顺序链
- 使用 `StrOutputParser` 可以简化字符串处理
