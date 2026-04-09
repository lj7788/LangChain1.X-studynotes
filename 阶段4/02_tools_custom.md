# 02_tools_custom.py

## 功能说明

演示 LangChain 1.x 中 **自定义工具** 的多种定义方式与 **工具绑定** 机制。

## 核心特性

| 特性 | 说明 |
|------|------|
| `@tool` 装饰器 | 一行代码将函数转为 Tool，自动推断参数 Schema |
| 多参数工具 | 支持默认值、类型注解、复杂参数 |
| `bind_tools()` | 将工具集合绑定到 LLM，让 LLM 自主决定何时调用 |
| `invoke()` | 可直接调用工具，也可通过 LLM 间接调用 |

## 代码结构

### 4种工具定义方式

```
01_search          → 最简单的单参数工具（文档字符串作为描述）
02_calculator       → 带输入校验的工具
03_get_weather      → 多参数工具（含默认值）
04_list_files       → 返回复杂格式结果的工具
```

### 关键 API

```python
from langchain_core.tools import tool

@tool
def my_func(param: type) -> str:
    """工具描述（会被 LLM 看到）。"""
    ...

# 查看 Schema
print(my_func.args_schema.model_json_schema())

# 直接调用
result = my_func.invoke({"param": "value"})

# 绑定到 LLM
bound_llm = llm.bind_tools([my_func])
response = bound_llm.invoke("帮我查一下...")
if response.tool_calls:
    for tc in response.tool_calls:
        print(tc["name"], tc["args"])
```

## 运行方式

```bash
python 阶段4/02_tools_custom.py
```

## 输出示例

```
============================================================
阶段4 - 自定义工具（@tool 装饰器）
============================================================

【Part 1】已注册的工具

  工具名: search
  描述:   搜索信息。当需要查找事实、定义或知识时使用此工具...
  参数 Schema:
    {
      "type": "object",
      "properties": {
        "query": {"title": "Query", "type": "string"}
      },
      "required": ["query"]
    }

...

【Part 2】直接调用工具

> search('python'):
  Python 是一种高级编程语言，以其简洁的语法和强大的生态系统著称。

> calculator('123 * 456'):
  55888

> get_weather('北京', 'celsius'):
  北京 当前天气:
  - 温度: 22°C
  - 天气: 晴朗
  - 湿度: 35%

【Part 3】LLM 工具绑定 (bind_tools)

用户: 今天北京天气怎么样？
  → 工具调用: get_weather
    参数: {'city': '北京', 'unit': 'celsius'}
```

## 与 01_agent_basic.py 的区别

| 维度 | 01_agent_basic | 02_tools_custom |
|------|---------------|-----------------|
| Tool 定义方式 | `Tool(name=..., func=..., description=...)` | `@tool` 装饰器 |
| 执行方式 | `AgentExecutor` 自动循环 | `llm.bind_tools()` 单次调用 |
| 控制粒度 | Agent 自动决策 | 手动处理 tool_calls 结果 |
| 适用场景 | 需要多步推理的 Agent | 函数调用 / API 集成 |

## 最佳实践

1. **写好 docstring**：LLM 通过 docstring 判断何时调用工具
2. **使用类型注解**：帮助生成准确的 JSON Schema
3. **设置合理默认值**：减少不必要的参数询问
4. **输入校验**：避免注入风险（尤其是 eval/exec 类操作）
