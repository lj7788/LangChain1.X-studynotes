"""
阶段3 - 18_multi_chunk_index.py
Retrieval - 多粒度索引预保存方案（使用三国演义数据）

预先创建不同大小的 chunks：
- 细粒度（chunk_size=300）：用于精确检索
- 粗粒度（chunk_size=800）：用于返回完整上下文

检索时先从细粒度索引检索，然后返回对应的粗粒度文档。

特点：索引可以保存到本地，下次直接加载使用，无需重新创建。

核心概念：
- 多粒度索引：创建不同大小的文档块
- 细粒度检索：使用小块进行精确检索
- 粗粒度返回：返回大块提供完整上下文
- doc_id 关联：通过 doc_id 关联细粒度和粗粒度文档

工作流程：
1. 加载原始文档（三国演义章节）
2. 创建细粒度 chunks（300字符）
3. 创建粗粒度 chunks（800字符）
4. 为每个 chunk 添加 doc_id 元数据
5. 分别建立细粒度和粗粒度向量索引
6. 检索时：用细粒度检索，返回粗粒度

参数说明：
- fine_splitter.chunk_size: 细粒度块大小（300字符）
- coarse_splitter.chunk_size: 粗粒度块大小（800字符）
- chunk_overlap: 块之间的重叠字符数

优点：
- 检索精确（使用小块）
- 上下文完整（返回大块）
- 无需额外 LLM 调用
- 索引可持久化
- 查询速度快

缺点：
- 需要额外的存储空间
- 需要维护两层索引
- 首次建立索引较慢

使用场景：
- 需要平衡精确性和完整性的场景
- 长文档检索
- 需要快速响应的应用
"""

import os
import re
import shutil
import sys
from pathlib import Path

# ============ 导入 LangChain 相关模块 ============
from langchain_community.document_loaders import TextLoader  # 文本加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_core.documents import Document  # 文档对象
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain_ollama import OllamaEmbeddings  # Ollama embedding 模型

# 导入项目工具
sys.path.append("../")
from tools import make_ollama, make_embedding

# ============ 初始化 Embedding 模型 ============
# 使用 Ollama 的中文 embedding 模型
# 模型名称格式：dengcao/Dmeta-embedding-zh:F16（必须带版本号）
embedding = make_embedding()

# ============ 配置路径 ============
# data_index 目录用于保存索引数据
INDEX_DIR = Path(__file__).parent / "data_index" / "multi_chunk"
FINE_INDEX_DIR = INDEX_DIR / "fine"  # 细粒度索引保存目录
COARSE_INDEX_DIR = INDEX_DIR / "coarse"  # 粗粒度索引保存目录

# ============ 加载三国演义数据 ============
# 读取 sg.txt 文件
txt_file = Path(__file__).parent / "data" / "sg.txt"
loader = TextLoader(str(txt_file), encoding="utf-8")
documents = loader.load()
text = documents[0].page_content

# ============ 解析章节 ============
# 使用正则表达式匹配章回标题，例如：第1回、第1回
pattern = r"(第[一二三四五六七八九十百千]+回)"
# 按章节标题分割文本
parts = re.split(pattern, text)

# 重新组装每个章节的完整内容
chapters = []
current_chapter = ""
for i, part in enumerate(parts):
    if re.match(pattern, part):  # 如果是章节标题
        if current_chapter:  # 保存上一章
            chapters.append(current_chapter)
        current_chapter = part  # 开始新章节
    else:
        current_chapter += part  # 追加章节内容
if current_chapter:
    chapters.append(current_chapter)  # 保存最后一章

# ============ 创建不同粒度的文档 chunks ============
# 细粒度分割器：chunk_size=300，每个小块约 300 字符
# 用于精确检索，块越小检索越精准，但可能丢失上下文
fine_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,  # 块之间有 30 字符重叠，避免边界信息丢失
    separators=["\n\n", "\n", "。", " ", ""]
)

# 粗粒度分割器：chunk_size=800，每个块约 800 字符
# 用于返回完整上下文，保留更多语义信息
coarse_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " "]
)

# 分割文档
# 为每个章节创建两种粒度的文档，并添加元数据
# doc_id 用于关联细粒度和粗粒度文档
fine_docs = fine_splitter.split_documents([
    Document(page_content=ch, metadata={"chapter": i+1, "doc_id": f"doc_{i}"}) 
    for i, ch in enumerate(chapters)
])
coarse_docs = coarse_splitter.split_documents([
    Document(page_content=ch, metadata={"chapter": i+1, "doc_id": f"doc_{i}"}) 
    for i, ch in enumerate(chapters)
])

# ============ 打印基本信息 ============
print("=== 多粒度索引示例（使用三国演义） ===\n")
print(f"总章节数: {len(chapters)}")
print(f"细粒度 chunks 数: {len(fine_docs)}")
print(f"粗粒度 chunks 数: {len(coarse_docs)}")

# ============ 检查并加载/创建索引 ============
# 检查索引是否存在（通过检查目录中的 sqlite 数据库文件）
fine_db_exists = os.path.exists(FINE_INDEX_DIR / "chroma.sqlite3")
coarse_db_exists = os.path.exists(COARSE_INDEX_DIR / "chroma.sqlite3")

# 如果两个索引目录都存在，则加载已保存的索引
load_existing = fine_db_exists and coarse_db_exists

if load_existing:
    print("\n📂 发现已保存的索引，正在加载...")

# ============ 加载或创建索引 ============
if load_existing:
    # ===== 加载已保存的索引 =====
    # 直接从保存的目录加载向量数据库，无需重新计算 embedding
    fine_vectorstore = Chroma(
        embedding_function=embedding,
        collection_name="sg_fine_chunks",
        persist_directory=str(FINE_INDEX_DIR)
    )
    coarse_vectorstore = Chroma(
        embedding_function=embedding,
        collection_name="sg_coarse_chunks",
        persist_directory=str(COARSE_INDEX_DIR)
    )
    # 打印索引中的向量数量
    print(f"细粒度索引已加载: {fine_vectorstore._collection.count()} 条")
    print(f"粗粒度索引已加载: {coarse_vectorstore._collection.count()} 条")
else:
    # ===== 创建新索引 =====
    print("\n🆕 创建新的索引...")
    
    # 删除旧索引（如果存在但不完整）
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    
    # 创建索引保存目录
    FINE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    COARSE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 Chroma 向量数据库
    # collection_name 用于区分不同的集合
    fine_vectorstore = Chroma(
        embedding_function=embedding,
        collection_name="sg_fine_chunks",
        persist_directory=str(FINE_INDEX_DIR)
    )

    coarse_vectorstore = Chroma(
        embedding_function=embedding,
        collection_name="sg_coarse_chunks",
        persist_directory=str(COARSE_INDEX_DIR)
    )

    # 添加文档到向量数据库（会自动计算 embedding）
    fine_vectorstore.add_documents(fine_docs)
    coarse_vectorstore.add_documents(coarse_docs)

    # 显式保存索引到磁盘
    fine_vectorstore.persist()
    coarse_vectorstore.persist()

    print(f"\n细粒度索引已保存到: {FINE_INDEX_DIR}")
    print(f"粗粒度索引已保存到: {COARSE_INDEX_DIR}")


def search_with_granularity(query, k=3):
    """
    多粒度检索函数
    
    工作流程：
    1. 用细粒度索引检索，找到最相关的 k 个小块
    2. 提取这些小块对应的 doc_id
    3. 返回对应的粗粒度完整章节
    
    参数：
        query: 查询字符串
        k: 检索的细粒度文档数量
    
    返回：
        粗粒度文档列表（完整章节内容）
    """
    # 1. 用细粒度检索找到最相关的 k 个小块
    # 细粒度块小，检索更精确
    fine_results = fine_vectorstore.similarity_search(query, k=k)
    
    # 2. 收集这些小块对应的 doc_id
    # 用于在粗粒度文档中找到对应的完整章节
    doc_ids = set()
    for doc in fine_results:
        doc_ids.add(doc.metadata.get("doc_id"))
    
    # 3. 直接返回对应的原始章节（粗粒度文档）
    # 这样可以提供完整的上下文信息
    coarse_results = []
    for doc in coarse_docs:
        if doc.metadata.get("doc_id") in doc_ids:
            full_doc = Document(
                page_content=doc.page_content,
                metadata={"chapter": doc.metadata.get("chapter")}
            )
            coarse_results.append(full_doc)
    
    return coarse_results


# ============ 测试检索 ============
print("\n" + "="*50)
print("\n查询: 诸葛亮的计谋")
results = search_with_granularity("诸葛亮的计谋")
for i, doc in enumerate(results):
    title = f"第{doc.metadata.get('chapter')}回"
    print(f"\n--- 文档 {i+1} [{title}] ---")
    print(f"内容: {doc.page_content[:200]}...")

print("\n" + "="*50)
print("\n查询: 曹操和袁绍")
results = search_with_granularity("曹操和袁绍")
for i, doc in enumerate(results):
    title = f"第{doc.metadata.get('chapter')}回"
    print(f"\n--- 文档 {i+1} [{title}] ---")
    print(f"内容: {doc.page_content[:200]}...")

# ============ 方案总结 ============
print("\n" + "="*50)
print("\n✅ 多粒度索引方案总结：")
print("   - 预建立细粒度和粗粒度两层索引")
print("   - 检索时先找细粒度，确定相关章节")
print("   - 返回粗粒度完整章节内容")
print("   - 无需额外 LLM 调用，查询速度快")
print(f"   - 索引保存在: {INDEX_DIR}")
print("   - 下次运行会自动加载，无需重新创建索引")
