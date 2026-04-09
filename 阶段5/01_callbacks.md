# 01_callbacks.py

## 功能说明

演示 LangChain **回调系统**，通过 `BaseCallbackHandler` 监控 LLM 调用的全过程。

## 核心组件

| 类名 | 功能 |
|------|------|
| `TokenCountHandler` | 统计 LLM 输入/输出 Token 数量 |
| `TimingHandler` | 追踪每个 Chain 步骤的执行耗时 |
| `EventLogHandler` | 记录完整的调用事件流 |

## 关键 API

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs): ...
    def on_llm_end(self, response, **kwargs): ...
    def on_chain_start(self, serialized, inputs, **kwargs): ...
    def on_chain_end(self, serialized, outputs, **kwargs): ...

# 使用方式
result = llm.invoke(prompt, config={"callbacks": [MyHandler()]})
```

## 常用回调钩子

| 钩子 | 触发时机 | 参数 |
|------|---------|------|
| `on_llm_start` | LLM 开始生成前 | `serialized`, `prompts` |
| `on_llm_end` | LLM 返回后 | `response: LLMResult` |
| `on_chain_start` | Chain 开始时 | `serialized`, `inputs` |
| `on_chain_end` | Chain 结束时 | `serialized`, `outputs` |
| `on_tool_start` | 工具被调用时 | `serialized`, `input_str` |
| `on_tool_end` | 工具返回后 | `serialized`, `output` |

## 运行方式

```bash
python 阶段5/01_callbacks.py
```
