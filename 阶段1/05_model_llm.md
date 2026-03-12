# OpenAI LLM 模型

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/05_model_llm.py` 中的代码。

---

## ⚠️ 常见问题说明

在使用 Gitee AI API 时，可能会遇到以下错误：

```
openai.BadRequestError: Error code: 400 - {'error': {'code': '400', 'message': "[Bad Request] Validation error for body application/json: input don't match type STRING"}}
```

**原因：**
- Gitee AI 的 API 可能只支持聊天接口（Chat API），不支持传统的 Completions 接口
- 使用 `OpenAI` 类时传入的是字符串格式，但 Gitee AI 期望的是消息列表格式

**解决方案：**
1. 使用 `ChatOpenAI` 替代 `OpenAI`
2. 传入消息列表格式而非纯字符串

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    model="GLM-4.7-Flash",
    temperature=0,
    base_url="https://ai.gitee.com/v1",
    api_key="your-api-key"
)

# 使用消息列表格式
response = model.invoke([HumanMessage(content="请用一句话介绍 LangChain")])
print("模型回复:", response.content)
```

---

## 完整代码

```python
from langchain_openai import OpenAI

model = OpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)

response = model.invoke("请用一句话介绍 LangChain")
print("模型回复:", response)
```

---

## 代码逐行解析

### 第 1 行：导入 OpenAI
```python
from langchain_openai import OpenAI
```
- **OpenAI**: 用于传统 LLM 的 OpenAI 兼容接口
- 与 ChatOpenAI 的区别：接收字符串，返回字符串

---

### 第 3-8 行：创建模型实例
```python
model = OpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)
```
- 使用 Gitee AI 的 OpenAI 兼容 API
- `temperature=0`: 生成结果更确定性

---

### 第 10-12 行：调用模型并输出
```python
response = model.invoke("请用一句话介绍 LangChain")
print("模型回复:", response)
```
- `invoke` 接收**字符串**（而非消息列表）
- 直接返回字符串文本

---

## ChatModel vs LLM 对比

| 特性 | ChatOpenAI | OpenAI |
|------|-----------|--------|
| 输入格式 | 消息列表 | 字符串 |
| 输出格式 | AIMessage 对象 | 字符串 |
| 适用场景 | 对话 | 纯文本生成 |
| 系统提示 | 支持 | 不支持 |

---

## 执行流程

```
输入: "请用一句话介绍 LangChain"
    ↓
model.invoke()
    ↓
输出: "LangChain 是一个用于构建 LLM 应用的框架"
```

---

## 核心概念

### OpenAI LLM
- 传统的大语言模型接口
- 输入输出都是纯文本
- 适用于简单的文本生成任务
- 不支持多轮对话上下文

### 推荐使用 ChatModel
- 现代应用推荐使用 `ChatOpenAI` (ChatModel)
- ChatModel 更灵活，支持系统消息
- 更符合 LLM 应用的发展趋势
