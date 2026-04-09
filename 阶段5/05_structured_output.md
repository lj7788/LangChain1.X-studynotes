# 05_structured_output.py

## 功能说明

演示 **`with_structured_output()`** — LangChain 1.x 最推荐的结构化输出方式。

## 核心对比

| 方式 | 可靠性 | 类型安全 | 适用场景 |
|------|--------|---------|---------|
| `OutputParser` | 中（后处理） | 高（Pydantic） | 兼容所有模型 |
| `with_structured_output()` | **高（模型层保证）** | **高** | 支持 function calling 的模型 |

## 两种模式

### Pydantic 模式（推荐）
```python
class MySchema(BaseModel):
    name: str
    score: float

structured_llm = llm.with_structured_output(MySchema)
result: MySchema = structured_llm.invoke("...")
result.name   # ✅ 有类型提示和补全
```

### JSON Schema 模式
```python
schema = {"type": "object", "properties": {...}}
structured_llm = llm.with_structured_output(schema)
result: dict = structured_llm.invoke("...")
```

## 关键优势

1. **模型原生保证**：LLM 直接按 Schema 生成，无需二次解析
2. **零解析错误**：不会出现 JSON 格式不合法的问题
3. **类型完整**：Literal 枚举、List 嵌套等全部支持
4. **链式集成**：可直接嵌入 `prompt | structured_llm` 管道中

## 运行方式

```bash
python 阶段5/05_structured_output.py
```
