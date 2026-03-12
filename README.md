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

---

## 阶段三：记忆与检索（第5-6周）

### 1. Memory

- ConversationBufferMemory
- ConversationSummaryMemory
- Entity Memory
- 持久化记忆

### 2. Retrieval

- RetrievalQA
- VectorStoreRetriever
- ContextualCompression
- MultiQueryRetriever
- Ensemble Retriever

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
