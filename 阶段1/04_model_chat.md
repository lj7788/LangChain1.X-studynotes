# ChatModel 对话模型

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/04_model_chat.py` 中的代码。

---

## 完整代码

```python
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)

messages = [
    SystemMessage(content="你是一个有帮助的助手"),
    HumanMessage(content="你好，请介绍一下自己")
]

response = model.invoke(messages)
print("模型回复:", response.content)
print("完整响应:", response)
```

---

## 代码逐行解析

### 第 1-2 行：导入并忽略警告
```python
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")
```

---

### 第 4 行：导入 ChatOpenAI
```python
from langchain_openai import ChatOpenAI
```
- **ChatOpenAI**: 用于聊天模型的 OpenAI 兼容接口

---

### 第 5 行：导入消息类型
```python
from langchain_core.messages import HumanMessage, SystemMessage
```
- **HumanMessage**: 用户消息
- **SystemMessage**: 系统消息（设置 AI 角色/行为）

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
- 使用 Gitee AI 的 OpenAI 兼容 API
- `temperature=0`: 生成结果更确定性

---

### 第 14-17 行：构建消息列表
```python
messages = [
    SystemMessage(content="你是一个有帮助的助手"),
    HumanMessage(content="你好，请介绍一下自己")
]
```
- **SystemMessage**: 设置 AI 角色为"有帮助的助手"
- **HumanMessage**: 用户的实际提问

---

### 第 19-21 行：调用模型并输出
```python
response = model.invoke(messages)
print("模型回复:", response.content)
print("完整响应:", response)
```
- `invoke` 接收消息列表，返回 `AIMessage` 对象
- `response.content`: 模型回复的文本内容
- `response`: 完整的响应对象（包含元数据）

---

## 消息类型

| 类型 | 说明 | 用途 |
|------|------|------|
| SystemMessage | 系统消息 | 设置 AI 角色、行为、约束 |
| HumanMessage | 用户消息 | 用户的输入 |
| AIMessage | AI 消息 | 模型的回复 |

---

## 执行流程

```
输入: messages (消息列表)
    ↓
[
    SystemMessage("你是一个有帮助的助手"),
    HumanMessage("你好，请介绍一下自己")
]
    ↓
model.invoke()
    ↓
输出: AIMessage(content="你好！我是...")
```

---

## 核心概念

### ChatModel vs LLM
- **ChatModel**: 对话模型，接收消息列表，返回对话式回复
- **LLM**: 纯文本模型，接收字符串，返回文本

### 消息格式的优势
- 支持多轮对话上下文
- 可以设置系统角色
- 更符合人类对话习惯
