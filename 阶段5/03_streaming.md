# 03_streaming.py

## 功能说明

演示 LangChain 的 **三种流式输出** 模式。

## 三种流模式对比

| API | 类型 | 返回内容 | 适用场景 |
|-----|------|---------|---------|
| `chain.stream()` | 同步 generator | 文本 chunk | CLI 工具、简单脚本 |
| `chain.astream()` | 异步 async gen | 文本 chunk | Web 服务、高并发场景 |
| `chain.stream_events()` | 同步 generator | 事件字典 | 调试、可观测性 |

## 关键代码

### 基础流式
```python
for chunk in chain.stream({"topic": "..."}):
    print(chunk, end="", flush=True)  # 逐步输出
```

### 异步流式
```python
async for chunk in chain.astream({"topic": "..."}):
    print(chunk, end="", flush=True)

await chain.astream(...)  # 需要包裹在 async 中运行
```

### 事件级流（最详细）
```python
for event in chain.stream_events(inputs, version="v2"):
    event_type = event["event"]  # 如 "on_chat_model_stream"
    data = event["data"]
```

## 运行方式

```bash
python 阶段5/03_streaming.py
```
