# LCEL 并行执行

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/03_lcel_parallel.py` 中的代码。

---

## 完整代码

```python
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

prompt1 = ChatPromptTemplate.from_template("用中文介绍 {topic}")
prompt2 = ChatPromptTemplate.from_template("用英文介绍 {topic}")

model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)

parser = StrOutputParser()

chain = RunnableParallel(
    chinese=prompt1 | model | parser,
    english=prompt2 | model | parser
)

result = chain.invoke({"topic": "LangChain"})
print("中文:", result["chinese"])
print("英文:", result["english"])
```

---

## 代码逐行解析

### 第 1-2 行：导入并忽略警告
```python
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")
```
- 忽略 Pydantic V1 兼容性警告

---

### 第 4-5 行：导入必要的组件
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
```
- **RunnableParallel**: 用于并行执行多个链的组件

---

### 第 7-8 行：创建两个提示词模板
```python
prompt1 = ChatPromptTemplate.from_template("用中文介绍 {topic}")
prompt2 = ChatPromptTemplate.from_template("用英文介绍 {topic}")
```
- `prompt1`: 中文介绍
- `prompt2`: 英文介绍

---

### 第 10-15 行：创建模型实例
```python
model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.io/v1",
    api_key="your-gitee-ai-api-key"
)
```
- 使用 Gitee AI 的 OpenAI 兼容 API
- 模型：Qwen2.5-7B-Instruct

---

### 第 17 行：创建输出解析器
```python
parser = StrOutputParser()
```

---

### 第 19-22 行：构建并行链
```python
chain = RunnableParallel(
    chinese=prompt1 | model | parser,
    english=prompt2 | model | parser
)
```
- **RunnableParallel**: 并行运行多个链
- `chinese`: 中文介绍链
- `english`: 英文介绍链
- 两者同时执行，互不影响

---

### 第 24-26 行：执行并输出
```python
result = chain.invoke({"topic": "LangChain"})
print("中文:", result["chinese"])
print("英文:", result["english"])
```
- 一次调用同时获取中英文介绍
- `result` 是一个字典，包含 `chinese` 和 `english` 两个键

---

## 执行流程

```
输入: {"topic": "LangChain"}
    ↓
┌─────────────────────────────────────┐
│  RunnableParallel (并行执行)        │
│  ┌─────────────┐  ┌─────────────┐  │
│  │ prompt1     │  │ prompt2     │  │
│  │    ↓        │  │    ↓        │  │
│  │ model       │  │ model       │  │
│  │    ↓        │  │    ↓        │  │
│  │ parser      │  │ parser      │  │
│  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────┘
    ↓
输出: {"chinese": "...", "english": "..."}
```

---

## 核心概念

### RunnableParallel
- 用于**并行执行**多个 Runnable 组件
- 接收多个命名参数，每个参数是一个独立的链
- 所有链同时运行，提高效率
- 返回一个字典，键是参数名，值是对应链的输出

### 适用场景
- 同一输入需要多种不同处理
- 并行调用多个 API
- 同时生成多种格式的输出
