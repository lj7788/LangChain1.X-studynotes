# 05_langgraph_conditional.py

## 功能说明

演示 LangGraph 的 **条件边** 和 **循环** 机制，构建一个带分支路由的智能客服系统。

## 核心概念

### 条件边 vs 普通边

| 边类型 | API | 说明 |
|--------|-----|------|
| **普通边** | `add_edge(a, b)` | 固定路径 a → b |
| **条件边** | `add_conditional_edges(node, route_fn, path_map)` | 根据 State 动态决定去向 |

### 条件路由函数签名

```python
def my_router(state: MyState) -> str:
    """必须返回目标节点名称或 END 常量"""
    if some_condition(state):
        return "node_a"    # 跳转到 node_a
    return END             # 或返回 "node_b"
```

### path_map 参数

```python
graph.add_conditional_edges(
    "source_node",           # 源节点
    router_function,         # 路由函数
    {
        "node_a": "node_a",  # 返回值 "node_a" → 跳转到 "node_a"
        "node_b": "node_b",
        "__end__": END,      # 返回值 END → 终止
    }
)
```

## 本例图结构

```
START → classify(分类) ──┬→ bot_agent(机器人) → resolve_check(确认) ──┬→ END
                         ├→ human_agent(人工) → END                  │
                         └→ END(直接结束)                             └→ human_agent → END
```

## 运行方式

```bash
python 阶段4/05_langgraph_conditional.py
```

## 测试用例

| 输入 | 预期路径 |
|------|---------|
| `"我登录不了"` | 分类(技术) → 机器人 → 确认 → 人工 |
| `"查账单"` | 分类(账单) → 机器人 → 确认 → 人工 |
| `"主营什么"` | 分类(咨询) → 机器人 → 确认 → **END** |
| `"我要投诉"` | 分类后直接 → 人工 → END |

## 进阶技巧

1. **循环实现**：让条件边的某个目标指向前面的节点即可形成循环
2. **多路分发**：路由函数可返回 3+ 个不同的目标
3. **动态决策**：路由逻辑可以基于 LLM 输出、规则引擎或外部 API 结果
