"""
阶段3 - 14_retrieval_parent_document.py
Retrieval - ParentDocumentRetriever 父子文档检索器（使用三国演义数据）

ParentDocumentRetriever 工作原理：
1. 将文档分割成较小的子文档（child），用于向量检索
2. 同时保存完整的父文档（parent），用于返回完整上下文
3. 检索时：用子文档做相似度匹配
4. 返回时：返回完整的父文档

这样既保证了检索的精确性，又保证了返回内容的完整性。

特点：索引可以保存到本地，下次直接加载使用，无需重新创建。
"""

import os
import re
import sys
from pathlib import Path

# ============ 导入 LangChain 相关模块 ============
from langchain_community.document_loaders import TextLoader  # 文本加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_classic.retrievers import ParentDocumentRetriever  # 父子文档检索器
from langchain_core.documents import Document  # 文档对象
from langchain_community.vectorstores import Chroma  # 向量数据库
from langchain_core.stores import InMemoryStore  # 内存存储
from langchain_ollama import OllamaEmbeddings  # Ollama embedding 模型

# 导入项目工具
sys.path.append("../")
from tools import make_ollama

# ============ 初始化 LLM 和 embedding 模型 ============
llm = make_ollama()
embedding = OllamaEmbeddings(model="dengcao/Dmeta-embedding-zh:F16")

# ============ 配置路径 ============
# parent_documents 目录用于保存原始章节内容
savePath = Path(__file__).parent / "data_index" / "chapters"
# indexSavePath 目录用于保存向量数据库索引
indexSavePath = Path(__file__).parent / "data_index" / "parent_child_documents"

# 检查索引是否已存在
# 通过检查 sqlite 数据库文件是否存在
hasIndexSave = os.path.exists(indexSavePath / "chroma.sqlite3")
hasSave = savePath.exists()

# ============ 加载或解析文档 ============
if hasSave:
    # 从已保存的章节文件加载
    parent_documents = []
    for file in savePath.iterdir():
        if file.suffix == ".txt":
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():  # 只保存非空内容
                    chapter_num = file.stem.split("_")[-1]
                    parent_documents.append(Document(
                        page_content=content, 
                        metadata={"chapter": chapter_num}
                    ))
else:
    # 加载三国演义文本
    txt_file = Path(__file__).parent / "data" / "sg.txt"
    loader = TextLoader(str(txt_file), encoding="utf-8")
    documents = loader.load()
    text = documents[0].page_content

    # 用正则表达式按章节分割（匹配 "第X回"）
    pattern = r"(第[一二三四五六七八九十百千]+回)"
    parts = re.split(pattern, text)

    # 重新组装每个章节
    chapters = []
    current_chapter = ""
    for i, part in enumerate(parts):
        if re.match(pattern, part):  # 如果匹配到章节名
            if current_chapter:
                chapters.append(current_chapter)
            current_chapter = part
        else:
            current_chapter += part
    if current_chapter:
        chapters.append(current_chapter)

    # 将章节转换为 Document 对象（父文档）
    parent_documents = [
        Document(page_content=ch, metadata={"chapter": i+1}) 
        for i, ch in enumerate(chapters)
    ]

    # 保存章节到文件
    savePath.mkdir(parents=True, exist_ok=True)
    for file in savePath.iterdir():
        if file.suffix == ".txt":
            file.unlink()

    for doc in parent_documents:
        if doc.page_content.strip():
            with open(savePath / f"chapter_{doc.metadata['chapter']}.txt", "w", encoding="utf-8") as f:
                f.write(doc.page_content)

print("=== ParentDocumentRetriever 示例（使用三国演义） ===\n")
print(f"总章节数（父文档数）: {len(parent_documents)}")

# ============ 关键配置 ============

# 1. 父文档分割器：将完整章节作为父文档（保留完整上下文）
# chunk_size=1500 表示每个父文档约 1500 字符
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", " "]
)

# 2. 子文档分割器：将父文档进一步分割成小块（用于精确检索）
# chunk_size=200 表示每个子文档约 200 字符
# 子文档越小，检索越精确，但可能丢失上下文
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separators=["\n\n", "\n", "。", " ", ""]
)

# ============ 加载或创建索引 ============
if hasIndexSave:
    print("\n📂 发现已保存的索引，正在加载...")
    
    # 从保存的目录加载向量数据库
    vectorstore = Chroma(
        embedding_function=embedding,
        collection_name="sg_parent_child_documents",
        persist_directory=str(indexSavePath),
    )
    
    # 创建 docstore
    docstore = InMemoryStore()
    
    # 重新添加文档以恢复 docstore
    # ParentDocumentRetriever 会自动处理 docstore
    print(f"索引已加载: {vectorstore._collection.count()} 条")
    
    # 重新添加文档到 retriever 以恢复 docstore
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(parent_documents)
else:
    print("\n🆕 创建新的索引...")
    
    # 创建索引保存目录
    indexSavePath.mkdir(parents=True, exist_ok=True)

    # 3. 创建向量数据库（存储子文档，用于检索）
    vectorstore = Chroma(
        embedding_function=embedding,
        collection_name="sg_parent_child_documents",
        persist_directory=str(indexSavePath),
    )

    # 4. 创建文档存储（存储父文档，用于返回完整内容）
    docstore = InMemoryStore()

    # 5. 创建父子文档检索器
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,    # 子文档向量库（用于检索）
        docstore=docstore,         # 父文档存储（用于返回完整内容）
        child_splitter=child_splitter,  # 子文档分割器
        parent_splitter=parent_splitter,  # 父文档分割器
    )

    # 添加文档，检索器会自动：
    # 1. 用 parent_splitter 分割成父文档
    # 2. 用 child_splitter 进一步分割成子文档
    # 3. 将子文档存入向量库，父文档存入 docstore
    retriever.add_documents(parent_documents)

    # 显式保存索引到磁盘
    vectorstore.persist()
    
    print(f"\n父子文档索引已保存到: {indexSavePath}")

# ============ 测试检索 ============

print("\n" + "="*50)
print("\n查询 1：诸葛亮的计谋")
# 检索流程：
# 1. 在子文档中找与"诸葛亮的计谋"最相似的文档
# 2. 找到这些子文档对应的父文档
# 3. 返回完整的父文档章节
results1 = retriever.invoke("诸葛亮的计谋")
for i, doc in enumerate(results1):
    title = f"第{doc.metadata.get('chapter')}回"
    print(f"\n--- 文档 {i+1} [{title}] ---")
    print(f"内容: {doc.page_content[:200]}...")

print("\n" + "="*50)
print("\n查询 2：曹操和袁绍")
results2 = retriever.invoke("曹操和袁绍")
for i, doc in enumerate(results2):
    title = f"第{doc.metadata.get('chapter')}回"
    print(f"\n--- 文档 {i+1} [{title}] ---")
    print(f"内容: {doc.page_content[:200]}...")

print("\n" + "="*50)
print("\n✅ ParentDocumentRetriever 方案总结：")
print("   - 预先建立父子文档关系")
print("   - 检索时：用子文档（200字）做相似度匹配")
print("   - 返回时：用父文档（1500字）返回完整上下文")
print("   - 优点：既精确又完整，无需额外 LLM 调用")
print("   - 缺点：需要额外的存储空间")
print("   - 适合：需要平衡精确性和完整性的场景")
print(f"   - 索引保存在: {indexSavePath}")
print("   - 下次运行会自动加载，无需重新创建索引")
