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

embeddings = OllamaEmbeddings(model="dengcao/Dmeta-embedding-zh:F16")


save_path = Path(__file__).parent / "faiss_index/sg_faiss_index"
if not save_path.exists():
    print(f"   向量数据库文件不存在: {save_path}")
    exit(1)

vectorstore = FAISS.load_local(
    str(save_path),
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"   向量数据库创建成功，包含 {vectorstore.index.ntotal} 个文档")
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
