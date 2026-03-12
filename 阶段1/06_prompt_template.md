# 提示词模板基础

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/06_prompt_template.py` 中的代码。

---

## 完整代码

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("请用 {language} 介绍 {topic}")

formatted_prompt = prompt.format(language="中文", topic="LangChain")
print("格式化后的提示词:")
print(formatted_prompt)
```

---

## 代码逐行解析

### 第 1 行：导入 ChatPromptTemplate
```python
from langchain_core.prompts import ChatPromptTemplate
```
- **ChatPromptTemplate**: 用于创建聊天格式的提示词模板

---

### 第 3 行：创建提示词模板
```python
prompt = ChatPromptTemplate.from_template("请用 {language} 介绍 {topic}")
```
- 使用字符串模板创建提示词
- `{language}` 和 `{topic}` 是变量占位符

---

### 第 5 行：格式化提示词
```python
formatted_prompt = prompt.format(language="中文", topic="LangChain")
```
- 调用 `format` 方法，传入实际参数
- 变量被替换为实际值

---

### 第 6-7 行：输出结果
```python
print("格式化后的提示词:")
print(formatted_prompt)
```

---

## 执行流程

```
输入: prompt.format(language="中文", topic="LangChain")
    ↓
请用 {language} 介绍 {topic}
    ↓
请用 中文 介绍 LangChain
    ↓
输出: ChatPromptTemplate(messages=[...])
```

---

## 输出结果

```
格式化后的提示词:
ChatPromptTemplate(messages=[HumanMessage(content='请用 中文 介绍 LangChain')])
```

---

## 核心概念

### 提示词模板 (Prompt Template)
- 用于动态构建提示词
- 支持变量占位符 `{variable}`
- 通过 `format()` 方法填充变量

### ChatPromptTemplate
- 聊天格式的提示词模板
- 会将格式化后的内容包装为 `HumanMessage`
- 支持多种消息类型组合
