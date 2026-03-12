# 输出解析器 - JSON

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/09_output_parser_json.py` 中的代码。

---

## 完整代码

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

parser = JsonOutputParser(pydantic_object=Person)

json_str = '{"name": "张三", "age": 25, "city": "北京"}'
result = parser.invoke(json_str)

print("解析结果:", result)
print("类型:", type(result))
print("姓名:", result["name"])
```

---

## 代码逐行解析

### 第 1 行：导入 JsonOutputParser
```python
from langchain_core.output_parsers import JsonOutputParser
```
- **JsonOutputParser**: 将 JSON 字符串解析为 Python 字典的解析器

---

### 第 2 行：导入 BaseModel
```python
from langchain_core.pydantic_v1 import BaseModel
```
- **BaseModel**: Pydantic 模型基类，用于定义数据结构

---

### 第 4-7 行：定义数据模型
```python
class Person(BaseModel):
    name: str
    age: int
    city: str
```
- 使用 Pydantic 定义数据结构
- `name`: 字符串类型
- `age`: 整数类型
- `city`: 字符串类型

---

### 第 9 行：创建 JSON 解析器
```python
parser = JsonOutputParser(pydantic_object=Person)
```
- 传入 Pydantic 模型，解析时会验证数据

---

### 第 11-12 行：解析 JSON
```python
json_str = '{"name": "张三", "age": 25, "city": "北京"}'
result = parser.invoke(json_str)
```

---

### 第 14-16 行：输出结果
```python
print("解析结果:", result)
print("类型:", type(result))
print("姓名:", result["name"])
```

---

## 执行流程

```
输入: '{"name": "张三", "age": 25, "city": "北京"}'
    ↓
JsonOutputParser.invoke(pydantic_object=Person)
    ↓
┌─────────────────────────────────────┐
│ 1. 解析 JSON 字符串                  │
│ 2. 验证数据类型                      │
│ 3. 转换为 Python 字典                │
└─────────────────────────────────────┘
    ↓
输出: {"name": "张三", "age": 25, "city": "北京"}
```

---

## 输出结果

```
解析结果: {'name': '张三', 'age': 25, 'city': '北京'}
类型: <class 'dict'>
姓名: 张三
```

---

## 核心概念

### JsonOutputParser
- 将 JSON 字符串解析为 Python 字典
- 结合 Pydantic 可以验证数据结构
- 适用于需要结构化输出的场景

### Pydantic 模型
- 用于定义期望的数据结构
- 自动类型验证
- 类型错误时抛出异常

### 典型用法

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Answer(BaseModel):
    explanation: str
    confidence: float

parser = JsonOutputParser(pydantic_object=Answer)

prompt = ChatPromptTemplate.from_template(
    "解释什么是 {topic}，返回 JSON 格式"
)
# 注意：需要在提示词中说明输出格式
# 可配合 JsonOutputParser.get_format_instructions() 使用
```

### JsonOutputParser.get_format_instructions()
```python
print(parser.get_format_instructions())
# 输出类似：
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
# {"properties": {"name": {"title": "name", "type": "string"}...}
```
