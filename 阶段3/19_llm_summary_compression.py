"""
阶段3 - 19_llm_summary_compression.py
Retrieval - LLM 摘要预压缩方案（使用三国演义数据）

对每个章节预先用 LLM 生成摘要：
1. 将三国演义按章节分割
2. 对每个章节用 LLM 生成简短的摘要
3. 预先建立摘要的向量索引

查询时：
1. 先检索摘要（快速定位相关章节）
2. 返回对应的完整章节内容
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import re
import os
import sys
sys.path.append("../")
from tools import make_ollama

llm = make_ollama()
embedding = OllamaEmbeddings(model="dengcao/Dmeta-embedding-zh:F16")

savePath = Path(__file__).parent / "data_index" / "chapters"
hasSave = savePath.exists()

if hasSave:
    print(f"已存在章节内容目录: {savePath}")
    chapter_docs = []
    for file in savePath.iterdir():
        if file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            chapter_num = int(file.stem.split("_")[-1]) - 1
            docs = loader.load()
            for doc in docs:
                doc.metadata = {"chapter": chapter_num}
            chapter_docs.extend(docs)
else:
    savePath.mkdir(parents=True)
    print(f"已创建章节内容目录: {savePath}")

    txt_file = Path(__file__).parent / "data" / "sg.txt"
    loader = TextLoader(str(txt_file), encoding="utf-8")
    documents = loader.load()
    text = documents[0].page_content

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

    chapter_docs = [Document(page_content=ch, metadata={"chapter": i + 1}) for i, ch in enumerate(chapters)]

print("=== LLM 摘要预压缩示例（使用三国演义） ===\n")
print(f"总章节数: {len(chapter_docs)}")

indexSavePath = Path(__file__).parent / "data_index" / "faiss_summaries"
hasIndexSave = indexSavePath.exists() and os.path.exists(str(indexSavePath / "index.faiss"))

if hasIndexSave:
    print(f"已存在章节摘要索引目录: {indexSavePath}")
    summary_vectorstore = FAISS.load_local(
        str(indexSavePath),
        embedding,
        allow_dangerous_deserialization=True
    )
    print(f"索引已加载")
else:
    if not indexSavePath.exists():
        indexSavePath.mkdir(parents=True)
    print(f"已创建章节摘要索引目录: {indexSavePath}")

    chapter_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", " "]
    )

    chunks = chapter_splitter.split_documents(chapter_docs)
    print(f"分割后的 chunks 数: {len(chunks)}")

    summary_prompt = ChatPromptTemplate.from_template("""请用一句话概括以下《三国演义》章节的核心内容（不超过50字）：

章节内容：
{content}

摘要：""")

    print("\n正在为每个章节生成摘要（预压缩）...")

    summaries = []
    for i, chunk in enumerate(chunks):
        msg = summary_prompt.format(content=chunk.page_content)
        response = llm.invoke(msg)
        summary_text = response.content if hasattr(response, 'content') else str(response)

        summary_doc = Document(
            page_content=summary_text.strip(),
            metadata={
                "chapter": chunk.metadata.get("chapter"),
                "chunk_index": i
            }
        )
        summaries.append(summary_doc)
        chapter_title = f"第{chunk.metadata.get('chapter')}回"
        print(f" {i+1}/{len(chunks)} {chapter_title}: {summary_text[:40]}...")

    print("\n正在构建 FAISS 索引...")
    summary_vectorstore = FAISS.from_documents(summaries, embedding)

    summary_vectorstore.save_local(str(indexSavePath))
    print("\n摘要索引已保存到: faiss_summaries")


def search_with_summary(query, k=2):
    """基于摘要检索：先找摘要，再返回完整内容"""
    summary_results = summary_vectorstore.similarity_search(query, k=k)

    full_contents = []
    for doc in summary_results:
        chapter = doc.metadata.get("chapter")
        for ch in chapter_docs:
            if ch.metadata.get("chapter") == chapter:
                full_contents.append(Document(
                    page_content=ch.page_content,
                    metadata={"chapter": chapter}
                ))
                break

    return full_contents


print("\n" + "=" * 50)
print("\n查询: 诸葛亮的计谋")
results = search_with_summary("诸葛亮的计谋")
for i, doc in enumerate(results):
    title = f"第{doc.metadata.get('chapter')}回"
    print(f"\n--- 章节 {i+1} [{title}] ---")
    print(f"内容: {doc.page_content[:200]}...")

print("\n" + "=" * 50)
print("\n查询: 曹操和袁绍的战争")
results = search_with_summary("曹操和袁绍的战争")
for i, doc in enumerate(results):
    title = f"第{doc.metadata.get('chapter')}回"
    print(f"\n--- 章节 {i+1} [{title}] ---")
    print(f"内容: {doc.page_content[:200]}...")

print("\n" + "=" * 50)
print("\n✅ LLM 摘要预压缩方案：")
print("   - 预先用 LLM 为每个章节生成摘要")
print("   - 检索时先搜索摘要（速度快）")
print("   - 返回时使用完整章节内容")
print("   - 需要一次性 LLM 调用生成摘要，但查询时无需 LLM")
print("   - 适合：长篇小说、知识库")
