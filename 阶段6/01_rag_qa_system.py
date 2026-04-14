"""
阶段6 - 01_rag_qa_system.py
RAG 问答系统实战 - 基于知识库的自然语言问答

综合运用阶段1-5的知识：
- LCEL 管道操作 + Prompt 模板（阶段1）
- Document Loader + Text Splitter + Embeddings + VectorStore（阶段2）
- Retriever 检索策略 + Memory 对话记忆（阶段3）
- Agent 工具调用（阶段4）
- 流式输出 + 结构化输出 + 错误重试（阶段5）

项目架构：
1. 文档加载与索引构建（离线阶段）
2. 检索增强生成（在线阶段）
3. 对话式问答（带历史记忆）
4. 多轮交互式问答

图结构：

    ┌──────────────┐
    │  用户输入     │
    └──────┬───────┘
           ▼
    ┌──────────────┐     ┌───────────────────┐
    │  问题路由     │────▶│  直接回答（闲聊）  │
    │  (分类判断)   │     └───────────────────┘
    └──────┬───────┘
           │ 需要检索
           ▼
    ┌──────────────┐
    │  多查询检索   │  ← MultiQueryRetriever
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  上下文压缩   │  ← ContextualCompression
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  LLM 生成    │  ← 带对话记忆
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  返回答案     │
    └──────────────┘
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tools import make_ollama


# ========== Part 1: 文档索引构建 ==========

def build_vectorstore(persist_dir: str = None) -> FAISS:
    """构建向量数据库（支持持久化加载）"""
    if persist_dir is None:
        persist_dir = str(Path(__file__).parent / "faiss_index" / "python_faq")

    # 尝试加载已有索引
    if Path(persist_dir).exists():
        print("  [索引] 加载已有向量数据库...")
        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    print("  [索引] 首次构建向量数据库...")

    # 1. 加载文档
    data_path = Path(__file__).parent / "data" / "python_faq.txt"
    loader = TextLoader(str(data_path), encoding="utf-8")
    documents = loader.load()
    print(f"  [索引] 加载文档: {len(documents)} 个，共 {len(documents[0].page_content)} 字符")

    # 2. 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\nQ:", "\n\n", "\n", "。", "；", "，", " "],
    )
    splits = text_splitter.split_documents(documents)
    print(f"  [索引] 分割为 {len(splits)} 个文本块")

    # 3. 创建向量数据库
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print(f"  [索引] 向量数据库创建成功，共 {vectorstore.index.ntotal} 条记录")

    # 4. 持久化
    Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(persist_dir)
    print(f"  [索引] 已保存到 {persist_dir}")

    return vectorstore


# ========== Part 2: RAG 链构建 ==========

def create_rag_chain(vectorstore: FAISS):
    """创建带对话记忆的 RAG 链"""

    # 检索器 - 使用 MMR 平衡相关性和多样性
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
    )

    # 问答 Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "你是一个 Python 编程专家助手。请基于以下检索到的知识库内容回答用户的问题。\n\n"
            "规则：\n"
            "1. 优先使用知识库中的内容回答\n"
            "2. 如果知识库中没有相关信息，可以基于你的知识补充，但要标注「补充说明」\n"
            "3. 回答要简洁明了，包含代码示例（如果适用）\n"
            "4. 如果问题与编程无关，礼貌说明你的专长范围"
        )),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=(
            "知识库内容：\n{context}\n\n"
            "问题：{question}"
        )),
    ])

    # 闲聊 Prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个友好的 Python 编程助手。简单回答用户的闲聊问题。"),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{question}"),
    ])

    llm = make_ollama()

    # 构建问答链
    def format_docs(docs):
        return "\n---\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    chat_chain = chat_prompt | llm | StrOutputParser()

    return qa_chain, chat_chain, retriever


# ========== Part 3: 问题路由 ==========

def is_retrieval_needed(question: str) -> bool:
    """判断问题是否需要检索知识库"""
    # 简单规则：如果问题包含编程相关关键词则检索
    programming_keywords = [
        "python", "函数", "类", "方法", "列表", "字典", "装饰器",
        "异步", "并发", "测试", "调试", "flask", "django", "web",
        "拷贝", "元组", "集合", "迭代", "生成器", "gil", "协程",
        "wsgi", "asgi", "cors", "pytest", "类型提示", "args", "kwargs",
        "如何", "怎么", "什么", "区别", "原理", "实现", "用法",
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in programming_keywords)


# ========== Part 4: 交互式问答 ==========

def demo_rag_qa():
    """RAG 问答系统演示"""
    print("=" * 60)
    print("阶段6 - RAG 问答系统实战")
    print("=" * 60)

    # 构建索引
    print("\n【步骤1: 构建知识库索引】")
    vectorstore = build_vectorstore()

    # 创建 RAG 链
    print("\n【步骤2: 创建 RAG 链】")
    qa_chain, chat_chain, retriever = create_rag_chain(vectorstore)

    # 对话记忆存储
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 包装带记忆的链
    qa_with_history = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    chat_with_history = RunnableWithMessageHistory(
        chat_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 测试问题
    print("\n【步骤3: 测试问答】")

    questions = [
        "你好，你是谁？",
        "Python 中列表和元组有什么区别？",
        "深拷贝和浅拷贝的区别是什么？",
        "装饰器怎么用？",
        "谢谢你的解答！",
    ]

    session_id = "rag-session-001"
    config = {"configurable": {"session_id": session_id}}

    for i, question in enumerate(questions):
        print(f"\n{'─' * 50}")
        print(f"Q{i+1}: {question}")

        # 判断是否需要检索
        need_retrieval = is_retrieval_needed(question)
        print(f"  [路由] {'知识库检索' if need_retrieval else '直接回答'}")

        if need_retrieval:
            # 先展示检索结果
            docs = retriever.invoke(question)
            print(f"  [检索] 命中 {len(docs)} 条相关文档")
            for j, doc in enumerate(docs[:2]):
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"    [{j+1}] {preview}...")

            # 生成回答
            answer = qa_with_history.invoke(
                {"question": question}, config=config
            )
        else:
            answer = chat_with_history.invoke(
                {"question": question}, config=config
            )

        print(f"\n  A: {answer}")

    # 对话统计
    print(f"\n{'═' * 50}")
    history = get_session_history(session_id)
    print(f"【对话统计】共 {len(history.messages)} 条消息")
    for i, msg in enumerate(history.messages):
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        preview = msg.content[:60].replace("\n", " ")
        print(f"  [{i}] {role}: {preview}...")


# ========== Part 5: 检索质量评估 ==========

def demo_retrieval_quality():
    """评估检索质量"""
    print("\n\n" + "=" * 60)
    print("【附录: 检索质量评估】")
    print("=" * 60)

    vectorstore = build_vectorstore()

    test_queries = [
        ("Python 的装饰器是什么？", 1),
        ("如何处理跨域请求？", 1),
        ("asyncio.gather 和 wait 的区别？", 1),
    ]

    for query, expected_relevant in test_queries:
        print(f"\n查询: {query}")

        # 相似度搜索
        results = vectorstore.similarity_search_with_score(query, k=3)
        print(f"  Top 3 结果:")
        for doc, score in results:
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [距离: {score:.4f}] {preview}...")

        # MMR 搜索
        mmr_results = vectorstore.max_marginal_relevance_search(query, k=3)
        print(f"  MMR Top 3 结果:")
        for doc in mmr_results:
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"    {preview}...")


if __name__ == "__main__":
    demo_rag_qa()
    demo_retrieval_quality()
