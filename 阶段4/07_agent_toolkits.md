# 07_agent_toolkits.py

## 功能说明

全面介绍 **Toolkit（工具集）** 的概念、内置 Toolkit、自定义方式和与 Agent/LangGraph 的集成方法。

## 内容结构

| Part | 内容 | 对应代码 |
|------|------|---------|
| Part 1 | JSON Toolkit（内置） | `demo_json_tools()` |
| Part 2 | 文件操作 Toolkit（@tool 组合） | `demo_file_tools()` |
| Part 3 | 自定义 Toolkit 类封装 | `demo_custom_toolkit()` |
| Part 4 | Toolkit + AgentExecutor 集成 | `demo_integration()` |

## 三种 Toolkit 组织方式

### 1. 使用内置 Toolkit
```python
from langchain_community.tools import JsonSpec, JsonListKeysTool, JsonGetValueTool
spec = JsonSpec(dict_=my_data)
tool = JsonListKeysTool(spec=spec)```

### 2. @tool 手动组装（最常用）
```python
@tool
def tool_a(x): ...
@tool
def tool_b(y): ...
my_toolkit_tools = [tool_a, tool_b]  # 就是普通 list
```

### 3. Toolkit 类封装（可复用）
```python
class MyToolkit:
    @property
    def tools(self):  # 必须有此属性
        @tool
        def my_tool(...): ...
        return [my_tool]

tk = MyToolkit()
all_tools = tk.tools  # 取出工具列表给 Agent 用
```

## 运行方式

```bash
python 阶段4/07_agent_toolkits.py
```
