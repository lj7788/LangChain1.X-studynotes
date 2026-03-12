# 输出解析器 - 字符串

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/08_output_parser_str.py` 中的代码。

---

## 完整代码

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = parser.invoke("你好，这是一段文本回复")
print("解析结果:", result)
print("类型:", type(result))
```

---

## 代码逐行解析

### 第 1 行：导入 StrOutputParser
```python
from langchain_core.output_parsers import StrOutputParser
```
- **StrOutputParser**: 将模型输出解析为纯字符串的解析器

---

### 第 3 行：创建解析器实例
```python
parser = StrOutputParser()
```

---

### 第 5 行：调用解析器
```parser = StrOutputParser()```
```python
result = parser.invoke("你好，这是一段文本回复")
```
- 接收字符串输入
- 直接返回字符串（无变化）

---

### 第 6-7 行：输出结果
```python
print("解析结果:", result)
print("类型:", type(result))
```

---

## 执行流程

```
输入: "你好，这是一段文本回复"
    ↓
StrOutputParser.invoke()
    ↓
输出: "你好，这是一段文本回复"
```

---

## 输出结果

```
解析结果: 你好，这是一段文本回复
类型: <class 'str'>
```

---

## 核心概念

### StrOutputParser
- 最简单的输出解析器
- 将输入直接转为字符串
- 通常用于链式调用中提取文本内容

### 典型用法

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("{question}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# 链式调用
chain = prompt | model | parser
result = chain.invoke({"question": "什么是 LangChain?"})
# result 现在是纯字符串
```

### 与 AIMessage 的区别

| 对象 | 类型 | 内容 |
|------|------|------|
| AIMessage | 对象 | 包含 content, additional_kwargs 等 |
| 字符串 | str | 纯文本内容 |
