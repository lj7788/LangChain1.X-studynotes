# 顺序链 - SequentialChain (多输入多输出)

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/12_chain_sequential_multi.py` 中的代码。

---

## 完整代码

```python
from langchain.chains import SequentialChain
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
    "将以下内容翻译成法语: {text}"
)
chain2 = chain2_prompt | model

overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["text"],
    output_variables=["english_text", "french_text"],
    verbose=True
)

result = overall_chain.invoke({"text": "LangChain 是一个 LLM 应用框架"})
print("英文:", result["english_text"])
print("法语:", result["french_text"])
```

---

## 代码逐行解析

### 第 1 行：导入 SequentialChain
```python
from langchain.chains import SequentialChain
```
- **SequentialChain**: 顺序链，支持多个输入和输出

---

### 第 2-3 行：导入模型和提示词
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

---

### 第 5-10 行：创建模型实例
```python
model = ChatOpenAI(...)
```

---

### 第 12-14 行：创建第一条链（英文翻译）
```python
chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model
```

---

### 第 16-18 行：创建第二条链（法语翻译）
```python
chain2_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成法语: {text}"
)
chain2 = chain2_prompt | model
```

---

### 第 20-25 行：创建顺序链
```python
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["text"],
    output_variables=["english_text", "french_text"],
    verbose=True
)
```
- `chains`: 子链列表
- `input_variables`: 输入变量列表
- `output_variables`: 输出变量列表

---

### 第 27-29 行：执行并输出
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
│ Chain 1: 英文翻译                    │
│ 输入: {text}                        │
│ "LangChain 是一个 LLM 应用框架"     │
│ ↓                                   │
│ "LangChain is an LLM application framework"│
│ 输出键: english_text                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Chain 2: 法语翻译                   │
│ 输入: {text}                        │
│ "LangChain 是一个 LLM 应用框架"     │
│ ↓                                   │
│ "LangChain est un framework d'application LLM"│
│ 输出键: french_text                 │
└─────────────────────────────────────┘
    ↓
输出: {
    "text": "...",
    "english_text": "...",
    "french_text": "..."
}
```

---

## 输出结果

```
> Entering new SequentialChain chain...
> Entering new LLMChain chain...
Prompt after formatting:
将以下内容翻译成英文: LangChain 是一个 LLM 应用框架
> Ending LLMChain chain...
> Entering new LLMChain chain...
Prompt after formatting:
将以下内容翻译成法语: LangChain 是一个 LLM 应用框架
> Ending LLMChain chain...
> Ending SequentialChain chain...
英文: LangChain is an LLM application framework.
法语: LangChain est un framework d'application LLM.
```

---

## 核心概念

### SequentialChain vs SimpleSequentialChain

| 特性 | SimpleSequentialChain | SequentialChain |
|------|----------------------|-----------------|
| 输入输出 | 单个 | 多个 |
| 灵活性 | 低 | 高 |
| 配置 | 简单 | 需要指定变量 |

### 关键参数

| 参数 | 说明 |
|------|------|
| chains | 子链列表 |
| input_variables | 输入变量名列表 |
| output_variables | 输出变量名列表 |

### 输入输出映射
- `input_variables`: 初始输入的变量
- `output_variables`: 最终输出的变量
- 中间链的输出会自动传递（如果变量名匹配）

### 适用场景
- 并行生成多个结果
- 需要保留中间结果
- 复杂的多步骤流水线
