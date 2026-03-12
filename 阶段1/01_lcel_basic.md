# LCEL 基础示例

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/01_lcel_basic.py` 中的代码。

---

## 完整代码

```python
from tools import make_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("用一句话介绍 {topic}")

model = make_model()

output_parser = StrOutputParser()

chain = prompt | model #| output_parser

result = chain.invoke({"topic": "智能体"})


print(result)
```

---

## 代码逐行解析

### 第 1 行：导入 make_model
```python
from tools import make_model
```
- 从 `tools` 模块导入 `make_model` 函数，用于创建模型实例

---

### 第 3 行：导入输出解析器
```python
from langchain_core.output_parsers import StrOutputParser
```
- **StrOutputParser**: 将模型输出解析为字符串的解析器

---

### 第 4 行：导入提示词模板
```python
from langchain_core.prompts import ChatPromptTemplate
```
- **ChatPromptTemplate**: 用于创建聊天格式的提示词模板

---

### 第 6 行：创建提示词模板
```python
prompt = ChatPromptTemplate.from_template("用一句话介绍 {topic}")
```
- 使用 `from_template` 方法创建提示词模板
- `{topic}` 是一个变量占位符，运行时会被替换

---

### 第 8 行：创建模型实例
```python
model = make_model()
```
- 调用 `make_model()` 函数创建一个 ChatOpenAI 模型实例

---

### 第 10 行：创建输出解析器
```python
output_parser = StrOutputParser()
```
- 创建字符串输出解析器实例

---

### 第 12 行：构建 LCEL 链
```python
chain = prompt | model #| output_parser
```
- 使用 `|` 管道操作符将组件串联成链
- `prompt | model` 表示：先将提示词格式化，然后传递给模型
- 注意：`#| output_parser` 被注释掉了，所以没有解析输出

---

### 第 14-15 行：执行链
```python
result = chain.invoke({"topic": "智能体"})
print(result)
```
- 调用 `invoke` 方法传入输入数据
- `{topic}` 会被替换为 "智能体"
- 输出是模型的原始响应（AIMessage 对象）

---

## 执行流程

```
输入: {"topic": "智能体"}
    ↓
prompt (格式化提示词)
    ↓
"用一句话介绍 智能体"
    ↓
model (调用大模型)
    ↓
AIMessage(content="智能体是...")
    ↓
输出: AIMessage 对象
```

---

## 核心概念

### LCEL (LangChain Expression Language)
LCEL 是 LangChain 的表达式语言，使用 `|` 管道操作符将多个组件串联成链。

### 管道操作符 `|`
- 将前一个组件的输出作为下一个组件的输入
- 代码简洁，可读性强
- 支持链式调用 `.invoke()`、`.stream()` 等方法
