# RunnableLambda 详解

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/02_lcel_runnable.py` 中的代码。

---

## 完整代码

```python
from tools import make_model
from langchain_core.runnables import RunnableLambda

def transform_func(data):
    return {"topic": data["topic"].upper()}

chain = RunnableLambda(transform_func)

result = chain.invoke({"topic": "hello"})
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

### 第 2 行：导入 RunnableLambda
```python
from langchain_core.runnables import RunnableLambda
```
- **RunnableLambda**: LangChain 提供的工具，用于将**普通 Python 函数**包装成 **Runnable 对象**
- 为什么要包装？因为 LangChain 的链式操作（如 `|` 管道操作）要求对象必须是 Runnable 类型

---

### 第 4-5 行：定义转换函数
```python
def transform_func(data):
    return {"topic": data["topic"].upper()}
```
- 接收一个字典 `data`
- 将 `data["topic"]` 转换为**大写**
- 返回新的字典

---

### 第 7 行：创建 Runnable 对象
```python
chain = RunnableLambda(transform_func)
```
- 用 `RunnableLambda` 包装函数，得到一个 **Runnable 对象**
- 现在这个函数可以调用 `.invoke()`、`.stream()` 等方法

---

### 第 9-10 行：调用并输出
```python
result = chain.invoke({"topic": "hello"})
print(result)
```

**执行流程**：
```
输入: {"topic": "hello"}
    ↓
transform_func({"topic": "hello"})
    ↓
{"topic": "HELLO"}  # hello 被转为大写
    ↓
输出: {"topic": "HELLO"}
```

**运行结果**：
```python
{'topic': 'HELLO'}
```

---

## 为什么需要 RunnableLambda？

| 方式 | 说明 |
|------|------|
| 普通函数 | `def transform_func(data): ...` |
| RunnableLambda | `RunnableLambda(transform_func)` |

**核心区别**：只有 Runnable 对象才能：
1. 使用管道操作符 `|` 串联成链
2. 调用 `.invoke()`、`.stream()`、`.batch()` 等方法
3. 与 LangChain 其他组件（如模型、提示词）无缝集成

---

## 在链式调用中的使用

这段代码虽然简单，但可以这样扩展：

```python
from langchain_core.runnables import RunnableLambda

def transform_func(data):
    return {"topic": data["topic"].upper()}

chain = RunnableLambda(transform_func)

# 单个调用
result = chain.invoke({"topic": "hello"})

# 批量调用
results = chain.batch([{"topic": "hello"}, {"topic": "world"}])
# 输出: [{'topic': 'HELLO'}, {'topic': 'WORLD'}]
```

这就是 LCEL 的基础用法！
