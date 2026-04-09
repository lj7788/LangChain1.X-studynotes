# 02_output_parsers.py

## 功能说明

全面演示 LangChain **输出解析器**，将 LLM 的自由文本输出转换为结构化数据。

## 四种解析器对比

| 解析器 | 输出类型 | 适用场景 |
|--------|---------|---------|
| `PydanticOutputParser` | Pydantic Model | 需要强类型约束、字段校验 |
| `JsonOutputParser` | `dict` / `list` | 灵活的 JSON 数据 |
| `XMLOutputParser` | `dict`(标签→值) | 需要标签化提取 |
| `CommaSeparatedListOutputParser` | `list[str]` | 列举类任务 |

## 关键模式

### Pydantic 方式（推荐用于生产）

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class MySchema(BaseModel):
    name: str = Field(description="名称")
    score: float = Field(description="分数")

parser = PydanticOutputParser(pydantic_object=MySchema)
prompt = "... {format_instructions} ..."
chain = prompt | llm | parser
result: MySchema = chain.invoke({...})
# result.name  ← 有自动补全！
```

### JSON 方式（快速原型）

```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
chain = prompt | llm | parser
result: dict = chain.invoke({...})
```

## 运行方式

```bash
python 阶段5/02_output_parsers.py
```
