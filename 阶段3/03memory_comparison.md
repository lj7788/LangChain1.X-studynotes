# LangChain 1.x Memory 组件对比

LangChain 1.x 对 Memory 组件进行了重大重构，本文对比各方案的区别。

---

## 1.x 主要变化

### 导入路径变化

| 0.x (旧版) | 1.x (新版) |
|------------|------------|
| `from langchain.memory import ...` | `from langchain_classic.memory import ...` |
| `from langchain.chat_message_histories import ...` | `from langchain_community.chat_message_histories import ...` |

### 核心类变化

| 旧版 (0.x) | 新版 (1.x) |
|------------|------------|
| `ConversationBufferMemory` | `ChatMessageHistory` + `RunnableWithMessageHistory` |
| `ConversationSummaryMemory` | `langchain_classic.memory.ConversationSummaryMemory` |
| `EntityMemory` | 手动实现或 `langchain_classic.memory.ConversationEntityMemory` (已废弃) |

---

## 方案对比

| 方案 | 核心类 | 优点 | 缺点 | 适用场景 |
|------|--------|------|------|----------|
| **Buffer** | `ChatMessageHistory` + `RunnableWithMessageHistory` | 无弃用风险、自动管理、多会话支持 | Token消耗高 | 短对话 |
| **Summary** | `ConversationSummaryMemory` | 节省Token、代码简单 | 需手动保存 | 长对话单会话 |
| **自定义摘要** | `ChatMessageHistory` + 自定义LLM调用 | 无弃用风险、灵活可控 | 代码量多 | 长对话多会话 |

---

## 推荐选择

```
对话轮次 < 10轮？
    │
    ├── ✅ 是 → Buffer 方案
    │
    └── ❌ 否 → 需要多会话？
            │
            ├── ✅ 是 → 自定义摘要 或 VectorStoreRetrieverMemory
            │
            └── ❌ 否 → Summary 方案
```

---

## 代码文件

| 文件 | 内容 |
|------|------|
| 01_memory_buffer.py | Buffer 方案 |
| 02_memory_summary.py | Summary 方案 |
| 03_memory_persist.py | 持久化方案 |
| 05_memory_buffer_window.py | 窗口记忆 |
| 06_memory_token_buffer.py | Token计数记忆 |
| 07_memory_vectorstore.py | 向量存储记忆 |
| 08_memory_entity_kg.py | 实体关系提取 |
