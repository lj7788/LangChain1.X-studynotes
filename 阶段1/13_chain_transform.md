# TransformChain 转换链

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/13_chain_transform.py` 中的代码。

---

## 完整代码

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableLambda

def transform_func(inputs):
    text = inputs["text"]
    words = text.split()
    return {"word_count": len(words), "original_text": text}

chain = TransformChain(
    input_variables=["text"],
    output_variables=["word_count", "original_text"],
    transform=transform_func
)

result = chain.invoke({"text": "LangChain 是一个强大的 LLM 应用框架"})
print("单词数:", result["word_count"])
print("原文:", result["original_text"])
```

---

## 代码逐行解析

### 第 1 行：导入 TransformChain
```python
from langchain_core.runnables import RunnableLambda
```
- **TransformChain**: 用于数据转换的链

---

### 第 2 行：导入 RunnableLambda
```python
from langchain_core.runnables import RunnableLambda
```
- 用于创建可运行的函数

---

### 第 4-7 行：定义转换函数
```python
def transform_func(inputs):
    text = inputs["text"]
    words = text.split()
    return {"word_count": len(words), "original_text": text}
```
- 接收输入字典
- 计算单词数量
- 返回转换后的字典

---

### 第 9-13 行：创建 TransformChain
```python
chain = TransformChain(
    input_variables=["text"],
    output_variables=["word_count", "original_text"],
    transform=transform_func
)
```
- `input_variables`: 输入变量
- `output_variables`: 输出变量
- `transform`: 转换函数

---

### 第 15-17 行：执行并输出
```python
result = chain.invoke({"text": "LangChain 是一个强大的 LLM 应用框架"})
print("单词数:", result["word_count"])
print("原文:", result["original_text"])
```

---

## 执行流程

```
输入: {"text": "LangChain 是一个强大的 LLM 应用框架"}
    ↓
┌─────────────────────────────────────┐
│ TransformChain                      │
│                                     │
│ 1. 提取 text: "LangChain 是一个..." │
│ 2. split(): ['LangChain', '是', ...]│
│ 3. len(): 11                        │
│ 4. 返回: {                           │
│      "word_count": 11,              │
│      "original_text": "..."         │
│    }                                 │
└─────────────────────────────────────┘
    ↓
输出: {
    "word_count": 11,
    "original_text": "LangChain 是一个强大的 LLM 应用框架"
}
```

---

## 输出结果

```
单词数: 11
原文: LangChain 是一个强大的 LLM 应用框架
```

---

## 核心概念

### TransformChain
- 用于数据转换的链
- 类似于 ETL 中的 Transform 步骤
- 不调用 LLM，纯数据处理

### 适用场景

| 场景 | 示例 |
|------|------|
| 文本处理 | 提取关键词、计算长度 |
| 数据清洗 | 格式化、过滤 |
| 特征工程 | 向量化、编码 |

### 与 LCEL 对比

```python
# TransformChain 方式
chain = TransformChain(
    input_variables=["text"],
    output_variables=["word_count"],
    transform=transform_func
)

# LCEL 方式 (更简洁)
chain = RunnableLambda(lambda x: {"word_count": len(x["text"].split())})
```

### 组合使用

```python
# TransformChain + LLMChain
transform_chain = TransformChain(...)

prompt = ChatPromptTemplate.from_template("{summary}")
llm_chain = LLMChain(llm=model, prompt=prompt)

# 使用 LCEL 组合
full_chain = transform_chain | llm_chain
```
