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

- 开发工具：VS Code / Trae IDE (macOS)
- 大语言模型：GLM-4.7-Flash（可通过 .env 配置）
- Python 3.13.3 环境
- LangChain 1.2.x 安装
- 基础依赖：langchain-core, langchain-community, langchain-openai
- API 密钥配置：在项目根目录创建 `.env` 文件，内容如下：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.example.com/v1
OPENAI_API_MODEL=GLM-4.7-Flash
```

> 注意：代码会自动读取项目根目录的 `.env` 文件，请确保已配置正确的 API 密钥。

**工具说明：**

项目提供了 `tools.py` 工具文件，包含以下便捷函数：

- `make_model(model_name)`: 创建 ChatOpenAI 模型实例
- `make_openai(model_name)`: 创建 OpenAI LLM 模型实例
- `make_ollama(model_name)`: 创建本地 Ollama 模型实例
- `make_embedding(base_url, model)`: 创建本地 Embeddings 实例（基于 llama.cpp）

**本地模型支持：**

项目支持使用本地模型（通过 Ollama）：

- LLM: `gemma4:e4b-it-q4_K_M_opt`
- Embeddings: `bge-m3:latest`

使用前请确保已启动 Ollama 服务并下载相应模型。

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
| 11_2_chain_sequential.py     | 顺序链 SimpleSequentialChain |
| 11_chain_sequential.py       | 顺序链基础示例               |
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
| 12_embeddings_openai.py | OpenAI Embeddings |
| 12_split_by_chapter.py | 按章节分割文档 |
| 13_embeddings_huggingface.md | HuggingFace 本地 Embeddings 文档 |
| 13_embeddings_ollama.py | Ollama 本地 Embeddings |
| 14_embeddings_similarity.py | Embeddings 相似度计算 |
| 15_vectorstore_chroma.py | Chroma 向量数据库 |
| 16_vectorstore_faiss.py | FAISS 向量数据库 |
| 17_vectorstore_save_load.py | 向量数据库保存与加载 |
| 18_vectorstore_mmr.py | MMR 搜索 |
| 19_rag_pipeline.py | 完整 RAG 流程示例 |
| 20_rag_sg.md | 三国演义 RAG 实战文档 |
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

**Memory 组件：**

| 文件 | 内容 |
| ---- | ---- |
| 01_memory_buffer.py | ChatMessageHistory + RunnableWithMessageHistory |
| 02_memory_summary.py | ConversationSummaryMemory 对话摘要 |
| 03memory_comparison.md | Memory 组件对比文档 |
| 04_memory_persist.py | 记忆持久化 |
| 05_memory_lcel.py | LCEL 中使用 Memory |
| 11_memory_buffer_window.py | 窗口记忆（最近 k 轮对话） |
| 12_memory_token_buffer.py | Token 计数记忆 |
| 15_memory_vectorstore.py | VectorStoreRetrieverMemory 向量存储记忆 |
| 16_memory_entity_kg.py | 实体关系提取与知识图谱 |
| 17_memory_file_history.py | FileChatMessageHistory 文件持久化 |

**Retrieval 组件：**

| 文件 | 内容 |
| ---- | ---- |
| 06_retrieval_compression.py | ContextualCompressionRetriever 上下文压缩 |
| 07_retrieval_multi_query.py | MultiQueryRetriever 多查询检索 |
| 08_retrieval_ensemble.py | EnsembleRetriever 集成检索器 |
| 09_retrieval_time_weighted.py | TimeWeightedRetriever 时间加权检索 |
| 13_retrieval_self_query.py | SelfQueryRetriever 自查询检索 |
| 14_retrieval_parent_document.py | ParentDocumentRetriever 父文档检索 |

**高级检索方案：**

| 文件 | 内容 |
| ---- | ---- |
| 18_multi_chunk_index.py | 多粒度索引预保存方案 |
| 19_llm_summary_compression.py | LLM 摘要预压缩方案 |

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

### 示例代码

**Agents（Agent 类型与执行）：**

| 文件 | 内容 |
| ---- | ---- |
| 01_agent_basic.py | Agent 基础 - AgentExecutor + ReAct 范式 |
| 03_agent_openai_tools.py | OpenAI Tool Calling 范式 - 函数调用 Agent |

**Tools（工具定义与绑定）：**

| 文件 | 内容 |
| ---- | ---- |
| 02_tools_custom.py | @tool 装饰器自定义工具 + bind_tools 绑定 |
| 07_agent_toolkits.py | 工具集（Toolkit）封装与集成 |

**LangGraph（有状态图）：**

| 文件 | 内容 |
| ---- | ---- |
| 04_langgraph_basic.py | LangGraph 基础：State / Node / Edge |
| 05_langgraph_conditional.py | 条件边与循环路由 |
| 06_langgraph_chatbot.py | **带记忆的聊天机器人实战** |

**代码位置：** `./阶段4/`

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

### 示例代码

| 文件 | 内容 |
| ---- | ---- |
| 01_callbacks.py | 回调系统 - BaseCallbackHandler 自定义回调 |
| 02_output_parsers.py | 输出解析器 - Pydantic / JSON / XML / List |
| 03_streaming.py | 流式输出 - stream / astream / stream_events |
| 04_error_retry.py | 错误处理与重试 - with_retry / with_fallbacks |
| 05_structured_output.py | 结构化输出 - with_structured_output |

**代码位置：** `./阶段5/`

---

## 阶段六：项目实战（第11-12周）

### 推荐项目

1. **RAG 问答系统** - 结合知识库的自然语言问答
2. **AI 聊天机器人** - 带记忆的对话系统
3. **Agent 自动化任务** - 使用工具自动完成任务
4. **三国演义智能问答系统** - 基于全书的 RAG + LangGraph 专家系统（命令行）
5. **三国通 Web API** - FastAPI 后端 + Vue3 前端交互式聊天界面（**收官项目**）

### 示例代码

| 文件 | 内容 | 综合运用 |
| ---- | ---- | ---- |
| 01_rag_qa_system.py | **RAG 问答系统实战** - 基于知识库的自然语言问答 | LCEL + 文档处理 + 检索策略 + 对话记忆 |
| 02_chatbot_with_memory.py | **AI 聊天机器人实战** - 带意图识别和记忆的对话系统 | LangGraph + 结构化输出 + 工具调用 + RAG |
| 03_agent_automation.py | **Agent 自动化任务实战** - 数据分析自动化助手 | 任务规划 + 多步循环 + 工具协作 + 报告生成 |
| 04_sg_expert_demo.py | **三国演义智能问答系统** - 基于全书的 RAG + LangGraph 专家系统 | RAG检索 + 意图识别 + 多工具协作 + 人物事件分析 |
| 05_sg_api_server.py | **🏆 三国通 Web API** - FastAPI + Vue3 前端聊天界面 | FastAPI + RESTful API + Vue3 SPA + 会话管理 |

**代码位置：** `./阶段6/`

### 🏆 收官项目：三国通 Web 应用

基于《三国演义》111回全文的完整 RAG + LangGraph 智能问答 Web 应用，综合运用了**全部六个阶段**的知识点：

#### 技术栈总览

```
┌──────────────────────────────────────────────────────┐
│                    三国通 架构                         │
├──────────┬─────────────────────┬─────────────────────┤
│  前端     │      后端           │       数据层         │
│ Vue3+Axios│   FastAPI          │  FAISS + bge-m3     │
│ 暗色中国风 │  LangGraph 图      │  三国演义111回全文    │
│ 实时计时器 │  5路意图路由        │  MMR 检索策略        │
└──────────┴─────────────────────┴─────────────────────┘
```

#### 知识点覆盖

| 阶段 | 知识点 | 在项目中的应用 |
|------|--------|--------------|
| **阶段一** | LCEL、Prompt Template、Output Parser | 构建 RAG 链、结构化意图分类 |
| **阶段二** | FAISS 向量数据库、bge-m3 Embeddings、文档分割 | 全书向量化索引、MMR 检索 |
| **阶段三** | RunnableWithMessageHistory、MMR 检索 | 多轮对话记忆保持 |
| **阶段四** | LangGraph StateGraph、条件边、工具调用、add_messages | 5路意图路由、4个专用工具 |
| **阶段五** | with_structured_output(Pydantic)、错误重试 | 5分类意图识别、LLM 容错 |
| **阶段六** | 项目4全部能力 + FastAPI + Vue3 | 完整的 Web 服务部署 |

#### 运行方式

```bash
# 启动三国通 Web 服务
python 阶段6/05_sg_api_server.py

# 浏览器访问 http://localhost:8000
```

详细说明见 [阶段6 README](./阶段6/README.md)

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

## 环境依赖

### 核心依赖

```
langchain==1.2.12
langchain-core==1.2.18
langgraph==1.1.1
langsmith==0.7.16
langchain-openai
langchain-community
langchain-ollama
python-dotenv
```

### 安装依赖

```bash
# 安装基础依赖
pip install langchain langchain-core langchain-community langchain-openai

# 安装 Ollama 支持（可选，用于本地模型）
pip install langchain-ollama

# 安装环境变量管理
pip install python-dotenv
```

### API 配置

所有示例代码使用配置在 `.env` 文件中的 API：

- **API 地址**: 通过 `OPENAI_API_BASE` 环境变量配置
- **API Key**: 通过 `OPENAI_API_KEY` 环境变量配置
- **默认模型**: 通过 `OPENAI_API_MODEL` 环境变量配置（默认：GLM-4.7-Flash）

### 本地模型支持（可选）

项目支持使用本地模型（通过 Ollama）：

- LLM: `gemma4:e4b-it-q4_K_M_opt`
- Embeddings: `bge-m3:latest`

使用前请确保已启动 Ollama 服务并下载相应模型：

```bash
# 启动 Ollama 服务
ollama serve

# 下载模型
ollama pull gemma4:e4b-it-q4_K_M_opt
ollama pull bge-m3:latest

# 启动本地服务器（可选）
bash start-llama-servers.sh
```
