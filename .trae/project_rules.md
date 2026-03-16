# 项目规则

## LangChain 版本要求

**重要：必须使用 LangChain 1.X 版本，禁止使用 0.X 版本**

### 导入规则

- ✅ **正确导入（LangChain 1.X）：**
  - `from langchain_core.prompts import PromptTemplate`
  - `from langchain_core.runnables import RunnablePassthrough`
  - `from langchain_core.output_parsers import StrOutputParser`
  - `from langchain_core.chat_history import BaseChatMessageHistory`
  - `from langchain_community.chat_message_histories import ChatMessageHistory`
  - `from langchain_openai import ChatOpenAI, OpenAIEmbeddings`
  - `from langchain_community.vectorstores import Chroma, FAISS`
  - `from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter`
  - `from langchain_core.documents import Document`

- ❌ **错误导入（LangChain 0.X）：**
  - `from langchain.prompts import PromptTemplate` ❌
  - `from langchain.llms import OpenAI` ❌
  - `from langchain.chat_models import ChatOpenAI` ❌
  - `from langchain.document_loaders import TextLoader` ❌

### 检索器导入

LangChain 1.X 中检索器的导入路径：
- `from langchain.retrievers import ContextualCompressionRetriever`
- `from langchain.retrievers.document_compressors import LLMChainExtractor`

### Embedding 配置

**使用 llama.cpp 兼容 API：**
```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(
    base_url="http://localhost:11435/v1",
    api_key="ollama"
)
```

**注意：**
- llama.cpp 的 embedding API 不接受 `model` 参数
- 使用端口 11435（embedding）和 11434（chat）

### Chat 模型配置

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama3.2",
    temperature=0,
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
```

### Embedding 模型配置

**使用 tools.py 中的函数：**

```python
from tools import make_embedding

embedding = make_embedding()
# 或自定义参数
embedding = make_embedding(base_url="http://localhost:11435/v1", model="embedding")
```

**LlamaCppEmbeddings 类（在 tools.py 中）：**
- 使用 OpenAI 客户端连接 llama.cpp 的 embedding API
- 默认端口：11435
- 默认模型：embedding
- 兼容 LangChain 的 embedding 接口（embed_documents, embed_query）
- 实现了 `__call__` 方法，支持 FAISS 等需要可调用对象的场景
- `embed_documents` 方法使用 batch_size=5 分批处理，平衡性能和稳定性

### 批量处理文档建议

当处理大量文档时（如 ParentDocumentRetriever），需要注意：

1. **llama.cpp embedding 限制**：
   - 单个输入最大 token 数：512（由 `--ubatch-size 512` 控制）
   - 建议子文档大小：200-300 字符
   - 使用 `batch_size=1` 逐个处理，避免批量超限

2. **Chroma 向量数据库限制**：
   - 最大批量大小：5461
   - 建议分批添加文档，每批 20-50 个文档
   - 示例代码：
     ```python
     batch_size = 20
     for i in range(0, len(documents), batch_size):
         batch = documents[i:i + batch_size]
         retriever.add_documents(batch)
     ```

3. **性能优化**：
   - 分批处理可以避免内存溢出
   - 显示进度信息，便于监控处理状态

### 模块映射

| LangChain 0.X | LangChain 1.X |
|--------------|--------------|
| `langchain.prompts` | `langchain_core.prompts` |
| `langchain.llms` | `langchain_openai` |
| `langchain.chat_models` | `langchain_openai` |
| `langchain.document_loaders` | `langchain_community.document_loaders` |
| `langchain.vectorstores` | `langchain_community.vectorstores` |
| `langchain.text_splitter` | `langchain_text_splitters` |
| `langchain.schema` | `langchain_core` |

### 环境信息

- 操作系统：macOS
- Python 版本：3.11
- LangChain 版本：1.2.11
- 后端服务：llama.cpp（兼容 Ollama API）
- Chat 端口：11434
- Embedding 端口：11435