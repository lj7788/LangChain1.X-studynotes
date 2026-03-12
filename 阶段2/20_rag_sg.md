# 三国演义 RAG 示例

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/20_rag_sg.py` 中的代码。

---

## 完整代码

```python
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from tools import make_ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

print("=== 三国演义 RAG 示例 ===")

txt_file = Path(__file__).parent / "data" / "sg.txt"

print("\n1. 加载三国演义文档...")
loader = TextLoader(str(txt_file), encoding="utf-8")
documents = loader.load()
text = documents[0].page_content
print(f"   文档加载成功，总长度: {len(text)} 字符")

print("\n2. 按章节分割文档...")
pattern = r"(第[一二三四五六七八九十百千]+回)"
parts = re.split(pattern, text)

chapters = []
current_chapter = ""

for i, part in enumerate(parts):
    if re.match(pattern, part):
        if current_chapter:
            chapters.append(current_chapter)
        current_chapter = part
    else:
        current_chapter += part

if current_chapter:
    chapters.append(current_chapter)

print(f"   分割为 {len(chapters)} 个章节")

print("\n3. 进一步分割章节为小块...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
)

all_splits = []
for i, chapter in enumerate(chapters):
    chapter_splits = text_splitter.split_text(chapter)
    for j, split in enumerate(chapter_splits):
        match = re.search(r"(第[一二三四五六七八九十百千]+回[^。\n]*)", split)
        title = match.group(1) if match else f"第{i+1}章"
        doc = Document(
            page_content=split,
            metadata={"chapter": i+1, "title": title}
        )
        all_splits.append(doc)

print(f"   进一步分割为 {len(all_splits)} 个文本块")

print("\n4. 创建向量数据库...")
embeddings = OllamaEmbeddings(model="dengcao/Dmeta-embedding-zh:F16")

vectorstore = FAISS.from_documents(
    documents=all_splits,
    embedding=embeddings
)
print(f"   向量数据库创建成功，包含 {vectorstore.index.ntotal} 个文档")

save_path = Path(__file__).parent / "faiss_index/sg_faiss_index"
save_path.parent.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(save_path))
print(f"   向量数据库已保存到 {save_path}")

print("\n5. 相似度搜索...")
query = "诸葛亮有哪些著名的计谋？"
results = vectorstore.similarity_search(query, k=8)
print(f"   查询: '{query}'")
print("   Top 8 相似内容:")
for i, doc in enumerate(results):
    title = doc.metadata.get("title", "无标题")
    content = doc.page_content[:300]
    print(f"   {i+1}. [{title}]")
    print(f"      {content}...")
    print()

print("\n6. 基于检索的问答 (RAG)...")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

prompt = ChatPromptTemplate.from_template("""你是一个熟悉《三国演义》的知识渊博的朋友。请基于以下《三国演义》的章节内容回答用户的问题。

注意：
1. 如果问题与三国演义无关，直接回答"这个问题与三国演义无关"
2. 如果无法从提供的章节内容中找到答案，回答"从提供的章节内容中找不到相关信息"
3. 回答要简洁明了

三国演义章节内容:
{context}

问题: {question}

回答:""")

model = make_ollama()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke(query)
print(f"\n问题: {query}")
print(f"回答: {answer}")
```

---

## 代码解析

### 文档加载
- 使用 `TextLoader` 加载《三国演义》文本文件
- 文件路径: `data/sg.txt`

### 按章节分割
- 使用正则表达式 `r"(第[一二三四五六七八九十百千]+回)"` 匹配章节标题
- 将文档按章节分割，保留章节标题

### 文本块分割
- 使用 `RecursiveCharacterTextSplitter` 将章节分割为小块
- `chunk_size=800`: 每块最大 800 字符
- `chunk_overlap=100`: 块之间重叠 100 字符
- 为每个文本块添加元数据：章节号和标题

### 向量数据库
- 使用 `OllamaEmbeddings` 和中文嵌入模型 `dengcao/Dmeta-embedding-zh:F16`
- 使用 `FAISS` 创建向量数据库
- 保存到本地: `faiss_index/sg_faiss_index`

### 相似度搜索
- 使用 `similarity_search` 查询最相关的 8 个文本块
- 显示每个结果的章节标题和内容预览

### RAG 问答
- 使用 MMR (最大边际相关性) 检索
- `k=10`: 返回 10 个结果
- `fetch_k=20`: 从 20 个候选中选择
- `lambda_mult=0.5`: 平衡相关性和多样性
- 使用 LCEL 构建 RAG 流程
- 使用中文 LLM 模型生成答案

### 关键特性
- 中文嵌入模型: 更适合中文文本
- 章节级别元数据: 便于追溯答案来源
- MMR 检索: 平衡相关性和多样性
- 向量数据库持久化: 避免重复创建
