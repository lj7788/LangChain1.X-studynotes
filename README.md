# LangChain 1.2.x 学习计划

**计划概览：**


| 阶段   | 内容                                                     | 周期      |
| ------ | -------------------------------------------------------- | --------- |
| 阶段一 | 基础入门（核心概念、LCEL、Model I/O、Chains）            | 第1-2周   |
| 阶段二 | 数据处理（Document Loaders、Text Splitters、Embeddings） | 第3-4周   |
| 阶段三 | 记忆与检索（Memory、Retrieval）                          | 第5-6周   |
| 阶段四 | Agent 与工具（Agents、Tools、LangGraph 基础）            | 第7-8周   |
| 阶段五 | 进阶应用（回调、输出验证、生产级特性）                   | 第9-10周  |
| 阶段六 | 项目实战（RAG 问答系统、聊天机器人、Agent 自动化）       | 第11-12周 |

## 阶段一：基础入门（第1-2周）

### 1. 核心概念理解

- LangChain 架构演进（从 0.x 到 1.x 的变化）
- LangChain 1.x 新特性
- LangChain 与 LangGraph 的关系

### 2. 环境准备

- 开发工具：Trae IDE (macOS)
- 大语言模型：MiniMax-M2.5
- Python 3.10+ 环境
- LangChain 1.2.x 安装
- 基础依赖：langchain-core, langchain-community
- API 密钥配置：在项目根目录创建 `.env` 文件，内容如下：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.minimaxi.com/v1
OPENAI_API_MODEL=MiniMax-M2.5
```

> 注意：代码会自动读取项目根目录的 `.env` 文件，请确保已配置正确的 API 密钥。

### 3. 核心组件

- **LCEL (LangChain Expression Language)**
  - 管道操作符 `|`
  - Runnable 接口
  - 组件组合
- **Model I/O**
  - Chat Models
  - LLMs
  - Output Parsers
  - Prompt Templates
- **Chains**
  - LLMChain
  - Sequential Chains
  - Transformation Chains

### 4. 示例代码


| 文件                         | 内容                         |
| ---------------------------- | ---------------------------- |
| 01_lcel_basic.py             | LCEL 基础 - 管道操作符`      |
| 02_lcel_runnable.py          | LCEL 自定义转换函数          |
| 03_lcel_parallel.py          | LCEL 并行执行                |
| 04_model_chat.py             | Chat Models 对话             |
| 05_model_llm.py              | LLM 基础使用                 |
| 06_prompt_template.py        | Prompt 模板基础              |
| 07_prompt_advanced.py        | 高级 Prompt 模板             |
| 08_output_parser_str.py      | 字符串输出解析器             |
| 09_output_parser_json.py     | JSON 输出解析器              |
| 10_chain_llm.py              | LLMChain 基础                |
| 11_chain_sequential.py       | 顺序链 SimpleSequentialChain |
| 12_chain_sequential_multi.py | 多输入输出顺序链             |
| 13_chain_transform.py        | TransformChain 转换链        |

**代码位置：** `./阶段1/`

---

## 阶段二：数据处理（第3-4周）

### 1. Document Loaders

- 文本文件加载
- PDF 加载
- 网页加载
- 数据库加载

### 2. Text Splitters

- 字符分割
- 递归分割
- Markdown 分割
- 代码分割

### 3. Embeddings

- OpenAI Embeddings
- 本地 Embeddings
- 向量存储

### 示例代码


| 文件 | 内容 |
| ---- | ---- |
| 01_document_loader_text.py | 文本文件加载 |
| 02_document_loader_csv.py | CSV 文件加载 |
| 03_document_loader_json.py | JSON 文件加载 |
| 04_document_loader_directory.py | 目录批量加载 |
| 05_document_loader_pdf.py | PDF 文件加载 |
| 06_document_loader_web.py | 网页加载 |
| 07_text_splitter_character.py | 字符分割器 |
| 08_text_splitter_recursive.py | 递归字符分割器 |
| 09_text_splitter_markdown.py | Markdown 分割器 |
| 10_text_splitter_python.py | Python 代码分割器 |
| 11_text_splitter_with_metadata.py | 带元数据的文档分割 |
| 12_split_by_chapter.py | 按章节分割文档 |
| 12_embeddings_openai.py | OpenAI Embeddings |
| 13_embeddings_huggingface.py | HuggingFace 本地 Embeddings |
| 13_embeddings_ollama.py | Ollama 本地 Embeddings |
| 14_embeddings_similarity.py | Embeddings 相似度计算 |
| 15_vectorstore_chroma.py | Chroma 向量数据库 |
| 16_vectorstore_faiss.py | FAISS 向量数据库 |
| 17_vectorstore_save_load.py | 向量数据库保存与加载 |
| 18_vectorstore_mmr.py | MMR 搜索 |
| 19_rag_pipeline.py | 完整 RAG 流程示例 |
| 20_rag_sg.py | **三国演义 RAG 实战** |
| 20_rag_sg_load.py | **三国演义 RAG - 加载向量数据库** |

**代码位置：** `./阶段2/`

### 🌟 特色示例：三国演义 RAG 实战

**20_rag_sg.py** 和 **20_rag_sg_load.py` 展示了一个完整的中文 RAG 应用：

- **中文嵌入模型**: 使用 `dengcao/Dmeta-embedding-zh:F16` 本地模型，无需 API Key
- **智能文档分割**: 按章节分割《三国演义》，保留章节标题作为元数据
- **MMR 检索**: 平衡相关性和多样性，获取更全面的上下文
- **向量数据库持久化**: 避免重复创建，快速启动应用
- **中文问答**: 基于检索内容回答关于《三国演义》的问题

**运行方式：**

```bash
# 首次运行：创建向量数据库
python 阶段2/20_rag_sg.py

# 后续运行：直接加载向量数据库
python 阶段2/20_rag_sg_load.py
```

**示例查询：**
- "诸葛亮有哪些著名的计谋？"
- "刘备三顾茅庐的故事是怎样的？"
- "赤壁之战的经过如何？"

---

## 阶段三：记忆与检索（第5-6周）

### 1. Memory (LangChain 1.x 核心组件)

#### 1.1 对话记忆（基础核心）

| 组件名 | 核心作用 | 1.x 版本关键特性 |
|--------|----------|------------------|
| **ConversationBufferMemory** | 存储完整对话历史（无截断） | 支持异步 async_save_context、自定义消息序列化 |
| **ConversationBufferWindowMemory** | 保留最近 k 轮对话 | 支持 return_messages 直接返回消息对象 |
| **ConversationSummaryMemory** | LLM 总结对话历史（压缩 Token） | 支持自定义总结模板、多语言总结 |
| **ConversationTokenBufferMemory** | 按 Token 数截断对话 | 适配 OpenAI/Anthropic 等多模型 Token 计算 |
| **ConversationSummaryBufferMemory** | 混合"窗口+总结"（优先窗口，超长则总结） | 1.x 主推的长对话记忆方案 |

#### 1.2 实体/结构化记忆（进阶）

| 组件名 | 核心作用 | 1.x 版本升级点 |
|--------|----------|----------------|
| **EntityMemory** | 提取/记忆对话中的实体 | 优化实体提取准确率，支持自定义实体类型 |
| **KnowledgeGraphMemory** | 构建实体关系图谱 | 支持 Neo4j 等外部图谱数据库持久化 |
| **VectorStoreRetrieverMemory** | 向量库存储记忆 | 适配 1.x 新的 VectorStore 接口 |

#### 1.3 持久化记忆（生产级）

| 组件名 | 核心作用 | 适用场景 |
|--------|----------|----------|
| **FileChatMessageHistory** | 对话历史存储到本地文件 | 单机持久化 |
| **RedisChatMessageHistory** | 对话历史存储到 Redis | 分布式/多实例场景 |
| **PostgresChatMessageHistory** | 对话历史存储到 PostgreSQL | 企业级持久化 |
| **MongoDBChatMessageHistory** | 对话历史存储到 MongoDB | 文档型数据库场景 |

### 2. Retrieval (LangChain 1.x 核心组件)

#### 2.1 核心检索器（Retriever）

| 组件名 | 核心作用 | 1.x 版本关键特性 |
|--------|----------|------------------|
| **VectorStoreRetriever** | 向量库相似度检索（RAG 核心） | 支持异步检索、批量检索、过滤条件优化 |
| **BM25Retriever** | 关键词/TF-IDF 检索 | 适配 1.x 新的 Document 格式 |
| **ParentDocumentRetriever** | 小片段检索+完整文档关联 | 支持多粒度分割 |
| **EnsembleRetriever** | 组合多个检索器 | 支持权重配置、结果融合策略 |
| **MultiQueryRetriever** | 生成多检索词提升召回率 | 支持自定义提示模板 |

#### 2.2 高级检索器

| 组件名 | 核心作用 | 适用场景 |
|--------|----------|----------|
| **SelfQueryRetriever** | 自然语言转检索条件 | 结构化元数据检索 |
| **ContextualCompressionRetriever** | 检索结果压缩 | 减少 Token 消耗 |
| **TimeWeightedVectorStoreRetriever** | 按时间权重检索 | 时序数据检索 |
| **MultiModalRetriever** | 多模态检索 | 图文混合检索场景 |

#### 2.3 RAG 检索套件

- `create_retrieval_chain`：快速构建检索→生成链路
- `create_stuff_documents_chain`：将检索结果直接传入 LLM
- `create_map_reduce_documents_chain`：分块总结→合并结果

### 示例代码

| 文件 | 内容 |
| ---- | ---- |
| 01_memory_buffer.py | ChatMessageHistory + RunnableWithMessageHistory |
| 02_memory_summary.py | ConversationSummaryMemory 对话摘要 |
| 03memory_comparison.md | 各种 Memory 对比 |
| 04_memory_persist.py | 记忆持久化 |
| 05_memory_lcel.py | LCEL 中使用 Memory |
| 06_retrieval_compression.py | ContextualCompressionRetriever |
| 07_retrieval_multi_query.py | MultiQueryRetriever |
| 08_retrieval_ensemble.py | EnsembleRetriever |
| 09_retrieval_time_weighted.py | TimeWeightedRetriever |
| 11_memory_buffer_window.py | 窗口记忆 |
| 12_memory_token_buffer.py | Token计数记忆 |
| 13_retrieval_self_query.py | SelfQueryRetriever |
| 14_retrieval_parent_document.py | ParentDocumentRetriever |
| 15_memory_vectorstore.py | VectorStoreRetrieverMemory |
| 16_memory_entity_kg.py | 实体关系提取与知识图谱 |
| 17_memory_file_history.py | FileChatMessageHistory 文件历史 |
| 18_multi_chunk_index.py | 多片段索引 |
| 19_llm_summary_compression.py | LLM 摘要压缩 |

**代码位置：** `./阶段3/`

---

## 阶段四：Agent 与工具（第7-8周）

### 1. Agents

- Agent Types
- Agent Executor
- 代理决策流程

### 2. Tools

- Tool 定义
- ToolKit
- 自定义工具
- 工具绑定

### 3. LangGraph 基础

- State
- Node
- Edge
- 条件边

---

## 阶段五：进阶应用（第9-10周）

### 1. 回调与监控

- Callbacks
- LangSmith 集成

### 2. 输出验证

- Pydantic Output Parser
- JSON Parser
- XML Parser

### 3. 生产级特性

- 错误处理
- 重试机制
- 流式输出

---

## 阶段六：项目实战（第11-12周）

### 推荐项目

1. **RAG 问答系统** - 结合知识库的自然语言问答
2. **AI 聊天机器人** - 带记忆的对话系统
3. **Agent 自动化任务** - 使用工具自动完成任务

---

## 学习资源

### 官方文档

- https://python.langchain.com/
- https://langchain-ai.github.io/langgraph/

### 推荐学习顺序

1. 先通读官方文档的 "Get Started"
2. 跟着官方教程动手实践
3. 阅读 LangChain 源码理解原理
4. 做项目巩固知识

---

## 环境依赖（已安装）

```
langchain==1.2.12
langchain-core==1.2.18
langgraph==1.1.1
langsmith==0.7.16
```

### 使用 Gitee AI 模型

所有示例代码使用 Gitee AI（https://ai.gitee.com/v1）作为模型提供者：

- **API 地址**: `https://ai.gitee.com/v1`
- **默认模型**: `Qwen/Qwen2.5-7B-Instruct`
- **API Key**: 需要在 Gitee AI 平台申请

运行示例前请安装依赖：

```bash
pip install langchain-openai
```
