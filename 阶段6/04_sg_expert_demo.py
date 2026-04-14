"""
阶段6 - 三国演义智能问答系统 Demo
========================================
基于《三国演义》120回全文的 RAG + LangGraph 智能问答系统

综合运用阶段1-5的知识：
- LCEL 管道操作 + Prompt 模板（阶段1）
- Document Loader + Text Splitter + Embeddings + VectorStore（阶段2）
- Retriever 检索策略 + Memory 对话记忆（阶段3）
- LangGraph 状态图 + 条件边 + 工具调用 Agent（阶段4）
- 结构化输出（意图识别/任务规划）+ 流式输出 + 错误重试（阶段5）

功能特性：
1. 📚 全书检索：基于120回原文的 RAG 检索增强问答
2. 🧭 意图识别：自动判断问题类型（闲聊 / 人物分析 / 事件查询 / 原文检索 / 综合分析）
3. 🔍 多工具协作：章节搜索、人物关系、事件时间线、势力对比等工具
4. 💬 对话记忆：多轮对话上下文保持
5. 📊 分析报告：对复杂问题自动规划并生成结构化分析报告

图结构：

              ┌──────────────┐
              │    START      │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  意图识别     │  ← with_structured_output
              └──────┬───────┘
                     │
     ┌───────────────┼───────────────┬───────────────┐
     ▼               ▼               ▼               ▼
 ┌────────┐    ┌──────────┐   ┌──────────┐    ┌──────────┐
 │ 闲聊   │    │ 原文检索  │   │ 人物/事件  │    │ 综合分析  │
 │ (chat) │    │  (rag)   │   │ (tools)  │    │ (agent)  │
 └───┬────┘    └────┬─────┘   └────┬─────┘    └────┬─────┘
     │               │              │                │
     └───────────────┴──────────────┴────────────────┘
                      ▼
               ┌──────────────┐
               │     END       │
               └──────────────┘
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated, Literal, List
from pydantic import BaseModel, Field

# LangChain 核心
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from tools import make_ollama


# ==================== 配置 ====================

DATA_DIR = Path(__file__).parent / "data" / "sg"
FAISS_DIR = Path(__file__).parent / "faiss_index" / "sg_novel"


# ==================== Part 1: 向量索引构建 ====================

def build_sg_vectorstore(rebuild: bool = False) -> FAISS:
    """构建三国演义向量数据库（支持持久化 + 索引完整性检查）

    Args:
        rebuild: 强制重建索引（忽略已有缓存）
    """

    # 加载所有章节文件，先获取完整列表
    chapter_files = sorted(DATA_DIR.glob("chapter_*.txt"), key=lambda f: int(re.search(r'\d+', f.name).group()))
    total_chapters = len(chapter_files)

    # ---- 索引检查：是否已有且完整 ----
    index_meta_file = FAISS_DIR / ".meta.json"

    if not rebuild and FAISS_DIR.exists() and index_meta_file.exists():
        print("  [索引] 发现已有索引，进行完整性校验...")

        try:
            with open(index_meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            cached_chapters = meta.get("total_chapters", 0)
            cached_records = meta.get("total_records", 0)
            cached_chunk_size = meta.get("chunk_size", 0)
            cached_chunk_overlap = meta.get("chunk_overlap", 0)

            print(f"  [索引-缓存] {cached_chapters} 章 / {cached_records} 条记录 "
                  f"(chunk={cached_chunk_size}, overlap={cached_chunk_overlap})")
            print(f"  [索引-当前] 数据目录共 {total_chapters} 章")

            if cached_chapters == total_chapters:
                print("  [索引] 章节数量匹配，直接加载缓存 ✓")
                embeddings = OllamaEmbeddings(model="bge-m3:latest")
                vs = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
                print(f"  [索引] 向量数据库就绪，共 {vs.index.ntotal} 条记录")
                return vs
            else:
                print(f"  [索引] ⚠ 章节数不匹配 (缓存{cached_chapters} ≠ 当前{total_chapters})，需要重建索引")

        except Exception as e:
            print(f"  [索引] ⚠ 缓存元数据损坏 ({e})，需要重建索引")

    # ---- 首次构建或重建 ----
    action = "强制重建" if rebuild else ("重建（数据有变化）" if FAISS_DIR.exists() else "首次构建")
    print(f"  [索引] {action}三国演义向量数据库...")

    all_docs = []
    print(f"  [索引] 发现 {total_chapters} 个章节文件，开始逐章加载...")

    for idx, cf in enumerate(chapter_files):
        loader = TextLoader(str(cf), encoding="utf-8")
        docs = loader.load()

        match = re.search(r'chapter_(\d+)\.txt', cf.name)
        if match:
            ch_num = int(match.group(1))
            for d in docs:
                d.metadata["chapter"] = ch_num
                first_line = d.page_content.split('\n')[0]
                d.metadata["title"] = first_line.strip()
        all_docs.extend(docs)

        # 进度提示
        if (idx + 1) % 20 == 0 or (idx + 1) == total_chapters:
            print(f"  [索引] 已加载 {idx + 1}/{total_chapters} 章...")

    total_chars = sum(len(d.page_content) for d in all_docs)
    print(f"  [索引] 共加载 {len(all_docs)} 个文档，{total_chars:,} 字符")

    # 分割文档 - 针对小说文本优化分隔符
    chunk_size = 500
    chunk_overlap = 80
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"  [索引] 分割为 {len(splits)} 个文本块 (chunk={chunk_size}, overlap={chunk_overlap})")

    # 创建向量数据库
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print(f"  [索引] 向量数据库创建成功，共 {vectorstore.index.ntotal} 条记录")

    # 持久化 + 写入元数据
    FAISS_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    meta = {
        "total_chapters": total_chapters,
        "total_records": vectorstore.index.ntotal,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_dir": str(DATA_DIR),
    }
    with open(index_meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  [索引] 已保存到 {FAISS_DIR} （含元数据校验文件）")

    return vectorstore


# ==================== Part 2: 工具定义 ====================

@tool
def search_chapters(keyword: str, top_k: int = 3) -> str:
    """搜索三国演义相关章节内容。当用户询问书中具体情节、原文引用时使用。

    Args:
        keyword: 搜索关键词或问题描述
        top_k: 返回结果数量，默认3条
    """
    try:
        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(keyword, k=top_k)
        if not docs:
            return "未找到相关内容"

        results = []
        for doc in docs:
            ch = doc.metadata.get("chapter", "?")
            title = doc.metadata.get("title", "")
            preview = doc.page_content[:300].replace("\n", " ")
            results.append(f"【第{ch}回】{title}\n{preview}...")

        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {e}"


@tool
def analyze_character(character_name: str) -> str:
    """分析三国人物。当用户问某个人物的性格、事迹、评价时使用。

    Args:
        character_name: 人物姓名，如"诸葛亮"、"曹操"、"关羽"
    """
    # 使用向量库搜索该人物相关段落
    try:
        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        # 构建多角度搜索词
        queries = [
            f"{character_name}的性格",
            f"{character_name}的事迹",
            f"{character_name}出场",
        ]
        all_docs = []
        seen_chapters = set()
        for q in queries:
            docs = vectorstore.similarity_search(q, k=2)
            for d in docs:
                ch = d.metadata.get("chapter", "?")
                if ch not in seen_chapters:
                    seen_chapters.add(ch)
                    all_docs.append(d)

        if not all_docs:
            return f"未找到关于「{character_name}」的详细记载"

        results = []
        for doc in all_docs[:4]:
            ch = doc.metadata.get("chapter", "?")
            preview = doc.page_content[:250].replace("\n", " ")
            results.append(f"[第{ch}回] {preview}...")

        return f"【{character_name}相关记载】\n" + "\n\n".join(results)
    except Exception as e:
        return f"人物分析失败: {e}"


@tool
def find_event(event_desc: str) -> str:
    """查找三国历史事件。当用户询问某个战役、政治事件、重要情节时使用。

    Args:
        event_desc: 事件描述，如"赤壁之战"、"桃园结义"、"三顾茅庐"
    """
    try:
        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(event_desc, k=5)

        if not docs:
            return f"未找到关于「{event_desc}」的事件记录"

        # 去重并整理
        results = []
        seen = set()
        for doc in docs:
            ch = doc.metadata.get("chapter", "?")
            if ch not in seen:
                seen.add(ch)
                title = doc.metadata.get("title", "")
                content = doc.page_content[:200].replace("\n", " ")
                results.append(f"【第{ch}回】{title}\n{content}...")

        return f"【事件：{event_desc}】\n" + "\n\n".join(results[:4])
    except Exception as e:
        return f"事件查找失败: {e}"


@tool
def compare_forces(force_a: str = "", force_b: str = "") -> str:
    """对比两个三国势力的优劣。用于比较魏蜀吴三方势力。

    Args:
        force_a: 第一个势力名称（如"蜀汉"、"曹魏"、"东吴"）
        force_b: 第二个势力名称
    """
    try:
        embeddings = OllamaEmbeddings(model="bge-m3:latest")
        vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)

        query = f"{force_a} {force_b} 对比 势力"
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            return f"暂无足够的关于{force_a}和{force_b}的对比信息"

        results = []
        for doc in docs[:3]:
            ch = doc.metadata.get("chapter", "?")
            content = doc.page_content[:250].replace("\n", " ")
            results.append(f"[第{ch}回] {content}...")

        return f"【{force_a} vs {force_b} 势力对比】\n" + "\n\n".join(results)
    except Exception as e:
        return f"势力对比失败: {e}"


TOOLS = [search_chapters, analyze_character, find_event, compare_forces]
tool_map = {t.name: t for t in TOOLS}


# ==================== Part 3: 意图识别（结构化输出）====================

class SGIntentResult(BaseModel):
    """三国问答意图识别结果"""
    intent: Literal["chat", "retrieval", "character", "event", "analysis"] = Field(
        description="""
        用户意图类型：
        - chat: 打招呼、闲聊、感谢、告别等非三国话题的对话
        - retrieval: 引用原文、查找具体情节、询问书中细节
        - character: 分析人物性格、生平事迹、人物评价
        - event: 查询战役、历史事件、关键情节
        - analysis: 复杂分析任务（势力对比、战略分析、综合报告）
        """
    )
    reasoning: str = Field(description="判断理由")
    tool_args: dict = Field(
        default_factory=dict,
        description="如果需要工具调用，填写参数；否则为空字典"
    )


# ==================== Part 4: 状态定义 ====================

class SGState(TypedDict):
    """三国智能问答系统状态"""
    messages: Annotated[list, add_messages]       # 对话历史
    intent: str                                    # 识别的意图
    reasoning: str                                 # 意图判断理由
    tool_args: dict                                # 工具参数
    vectorstore: object                            # 向量数据库实例（不序列化）


# ==================== Part 5: 节点函数 ====================

def node_classify(state: SGState) -> dict:
    """意图识别节点"""
    llm = make_ollama()
    structured_llm = llm.with_structured_output(SGIntentResult)

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    prompt = (
        "你是一个三国演义专家助手「三国通」的意图识别器。\n"
        "请判断以下用户问题的意图类型：\n\n"
        f"用户消息: {last_user_msg}\n\n"
        "意图分类规则：\n"
        "- chat: 打招呼、闲聊、感谢、告别、与三国无关的问题\n"
        "- retrieval: 要求引用原文、查找具体情节、询问书中某段描述\n"
        "- character: 问某个人物的性格、事迹、能力评价、人物关系\n"
        "- event: 问某个战役、政治事件、重要情节的经过\n"
        "- analysis: 复杂的分析需求，如势力对比、战略分析、多方对比等"
    )

    result: SGIntentResult = structured_llm.invoke(prompt)
    print(f"  [意图] {result.intent} | 理由: {result.reasoning[:60]}")

    return {
        "intent": result.intent,
        "reasoning": result.reasoning,
        "tool_args": result.tool_args or {},
    }


def node_chat(state: SGState) -> dict:
    """闲聊节点"""
    llm = make_ollama()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "你是「三国通」，一个热爱三国演义的AI助手。\n"
            "你博闻强识，对三国演义的每一回都烂熟于心。\n"
            "你说话风趣幽默，喜欢用三国典故来打比方。\n"
            "对于非三国的闲聊问题，友好回应，并巧妙地引向三国话题。\n"
            "回答时可以适当引用原文或诗词增加文采。"
        )),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": state["messages"]})
    print(f"  [闲聊] {response[:60]}...")
    return {"messages": [AIMessage(content=response)]}


def node_retrieval(state: SGState) -> dict:
    """原文检索节点 - RAG 检索增强"""
    llm = make_ollama()

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 搜索相关原文
    vectorstore = state.get("vectorstore")
    if vectorstore:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        docs = retriever.invoke(last_user_msg)
    else:
        # fallback: 直接用工具搜索
        search_result = search_chapters.invoke({"keyword": last_user_msg, "top_k": 5})
        docs = []

    context_parts = []
    if docs:
        print(f"  [检索] 命中 {len(docs)} 条相关文档")
        for i, doc in enumerate(docs[:4]):
            ch = doc.metadata.get("chapter", "?")
            title = doc.metadata.get("title", "")
            preview = doc.page_content[:400].replace("\n", " ")
            context_parts.append(f"【第{ch}回】{title}\n{preview}")
            print(f"    [{i+1}] 第{ch}回: {preview[:50]}...")
    else:
        search_result = search_chapters.invoke({"keyword": last_user_msg, "top_k": 5})
        context_parts.append(search_result)

    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "你是「三国通」，一个精通三国演义的学者。\n"
            "请严格基于以下检索到的原著原文回答用户问题。\n\n"
            "规则：\n"
            "1. 优先使用原文中的内容回答，标注出处（第几回）\n"
            "2. 可以适当解释和总结，但要忠实于原文\n"
            "3. 如果原文中没有相关信息，说明后补充你的知识\n"
            "4. 回答要有文采，适当引用原文中的精彩语句"
        )),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="原著原文：\n{context}\n\n问题：{question}"),
    ])

    history = [m for m in state["messages"][:-1]
               if isinstance(m, (HumanMessage, AIMessage))]

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "history": history,
        "context": context,
        "question": last_user_msg,
    })
    print(f"  [检索回答] {response[:60]}...")
    return {"messages": [AIMessage(content=response)]}


def node_character_event(state: SGState) -> dict:
    """人物/事件分析节点 - 调用专用工具"""
    llm = make_ollama()
    intent = state.get("intent", "")

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 根据意图选择工具
    bound_llm = llm.bind_tools(TOOLS)
    messages = [
        SystemMessage(content="你是「三国通」三国演义专家。请根据用户问题调用合适的工具获取信息，然后给出专业分析。"),
        *state["messages"],
    ]
    response = bound_llm.invoke(messages)

    new_messages = [response]

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            print(f"  [工具] {name}({args})")

            if name in tool_map:
                result = tool_map[name].invoke(args)
            else:
                result = f"未知工具: {name}"

            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        # LLM 基于工具结果生成最终回复
        all_msgs = list(state["messages"]) + new_messages
        final_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你是「三国通」，三国演义专家。请基于工具返回的信息，\n"
                "给出专业、详细、有见地的分析。回答要有深度，结合历史背景。\n"
                "适当引用原文或诗词增加文采。"
            )),
            MessagesPlaceholder(variable_name="all_msgs"),
        ])
        chain = final_prompt | llm | StrOutputParser()
        response = chain.invoke({"all_msgs": all_msgs})
        print(f"  [分析] {response[:60]}...")
        return {"messages": [AIMessage(content=response)]}
    else:
        # 无工具调用，直接用通用知识回答
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你是「三国通」，三国演义专家。请基于你的知识回答用户关于人物或事件的问题。"
            )),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm | StrOutputParser()
        resp = chain.invoke({"messages": state["messages"]})
        print(f"  [直接回答] {resp[:60]}...")
        return {"messages": [AIMessage(content=resp)]}


def node_analysis(state: SGState) -> dict:
    """综合分析节点 - 任务规划 + 多步执行"""
    llm = make_ollama()

    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # Step 1: 规划子任务
    plan_prompt = (
        "你是一个三国演义分析任务规划师。将用户的复杂分析需求分解为子任务。\n\n"
        "可用工具：\n"
        "- search_chapters: 搜索章节内容（参数: keyword, top_k）\n"
        "- analyze_character: 分析人物（参数: character_name）\n"
        "- find_event: 查找事件（参数: event_desc）\n"
        "- compare_forces: 势力对比（参数: force_a, force_b）\n\n"
        f"用户需求: {last_user_msg}\n\n"
        "请输出 JSON 格式的任务计划："
        '{"subtasks": [{"id":1, "desc":"子任务描述", "tool":"工具名", "args":{}}], "summary":"计划摘要"}'
    )

    # 用 LLM 生成计划（模拟结构化输出）
    plan_response = llm.invoke(plan_prompt)
    print(f"  [规划] {plan_response.content[:100]}...")

    # Step 2: 执行各子任务并收集结果
    step_results = []
    bound_llm = llm.bind_tools(TOOLS)

    exec_messages = [
        SystemMessage(content="你是三国演义分析执行助手。按步骤执行分析任务，使用合适的工具。"),
        HumanMessage(content=f"分析需求: {last_user_msg}\n请逐步执行分析并汇总。"),
    ]

    exec_response = bound_llm.invoke(exec_messages)
    new_messages = [exec_response]

    if hasattr(exec_response, 'tool_calls') and exec_response.tool_calls:
        for tc in exec_response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            print(f"  [执行] {name}({args})")

            if name in tool_map:
                result = tool_map[name].invoke(args)
            else:
                result = f"未知工具"

            step_results.append(f"[{name}] {str(result)[:150]}...")
            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    # Step 3: 生成综合分析报告
    if step_results:
        all_msgs = exec_messages + new_messages
        report_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "你是「三国通」首席分析师。请基于收集到的信息，撰写一份深入、专业的分析报告。\n\n"
                "报告要求：\n"
                "1. 开篇点题，明确核心观点\n"
                "2. 分层次展开分析（至少3个维度）\n"
                "3. 结合原文引用佐证\n"
                "4. 总结要点，给出独到见解\n"
                "5. 语言精炼有文采"
            )),
            MessagesPlaceholder(variable_name="msgs"),
        ])
        chain = report_prompt | llm | StrOutputParser()
        report = chain.invoke({"msgs": all_msgs})
    else:
        fallback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是「三国通」首席分析师，请对以下问题进行深入分析。"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = fallback_prompt | llm | StrOutputParser()
        report = chain.invoke({"messages": state["messages"]})

    print(f"  [报告] ({len(report)} 字符)")
    return {"messages": [AIMessage(content=report)]}


# ==================== Part 6: 条件路由 ====================

def route_by_intent(state: SGState) -> str:
    """根据意图路由到不同处理节点"""
    intent = state.get("intent", "chat")
    route_map = {
        "chat": "chat_node",
        "retrieval": "retrieval_node",
        "character": "char_event_node",
        "event": "char_event_node",
        "analysis": "analysis_node",
    }
    target = route_map.get(intent, "chat_node")
    print(f"  [路由] → {target}")
    return target


# ==================== Part 7: 构建 LangGraph ====================

def build_sg_graph(vectorstore=None):
    """构建三国智能问答系统 LangGraph 图"""
    graph = StateGraph(SGState)

    graph.add_node("classify", node_classify)
    graph.add_node("chat_node", node_chat)
    graph.add_node("retrieval_node", node_retrieval)
    graph.add_node("char_event_node", node_character_event)
    graph.add_node("analysis_node", node_analysis)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "chat_node": "chat_node",
            "retrieval_node": "retrieval_node",
            "char_event_node": "char_event_node",
            "analysis_node": "analysis_node",
        },
    )
    graph.add_edge("chat_node", END)
    graph.add_edge("retrieval_node", END)
    graph.add_edge("char_event_node", END)
    graph.add_edge("analysis_node", END)

    app = graph.compile()

    # 注入 vectorstore 到初始状态
    def invoke_with_vs(input_state):
        full_state = {**input_state, "vectorstore": vectorstore}
        return app.invoke(full_state)

    return app, invoke_with_vs


# ==================== Part 8: 主演示函数 ====================

def main(rebuild: bool = False):
    """运行三国演义智能问答系统

    Args:
        rebuild: 是否强制重建向量索引（忽略已有缓存）
    """
    print("=" * 64)
    print("   三国演义智能问答系统 - 三国通")
    print("   基于 RAG + LangGraph + 结构化输出 + 工具调用")
    print("=" * 64)

    # 1. 构建向量索引（带完整性检查）
    print("\n【第一步：构建知识库索引】")
    vectorstore = build_sg_vectorstore(rebuild=rebuild)

    # 2. 构建图
    print("\n【第二步：构建智能问答图】")
    app, invoke_app = build_sg_graph(vectorstore)

    # 打印图结构
    try:
        print("\n【图结构 Mermaid】\n")
        print(app.get_graph().draw_mermaid())
    except Exception:
        pass

    # 3. 测试对话
    test_questions = [
        ("你好！你是谁？", "chat"),
        ("桃园结义的三个人是谁？", "event"),
        ("诸葛亮是一个什么样的人？", "character"),
        ("赤壁之战中诸葛亮借东风是怎么回事？请引用原文", "retrieval"),
        ("曹操和刘备的用人之道有什么不同？", "analysis"),
        ("关羽温酒斩华雄是哪一回？", "retrieval"),
        ("谢谢你的讲解！再见！", "chat"),
    ]

    print("\n" + "=" * 64)
    print("开始对话测试")
    print("=" * 64)

    system_msg = SystemMessage(content=(
        "你是「三国通」，一个精通三国演义的AI助手。"
        "你对120回三国演义烂熟于心，能背诵其中的诗词和经典段落。"
        "你说话既有学术功底，又生动有趣。"
    ))

    state = {"messages": [system_msg], "intent": "", "reasoning": "", "tool_args": {}}

    for i, (question, expected_intent) in enumerate(test_questions):
        print(f"\n{'─' * 56}")
        print(f"Q{i+1} [{expected_intent.upper()}]: {question}")
        print(f"{'─' * 56}")

        state["messages"].append(HumanMessage(content=question))
        state["intent"] = ""
        state["reasoning"] = ""
        state["tool_args"] = {}

        result = invoke_app(state)

        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            last_ai = ai_messages[-1]
            print(f"\n🤖 三国通:\n")
            # 格式化长回复
            answer = last_ai.content
            if len(answer) > 300:
                print(answer[:300])
                print(f"\n  ... (共 {len(answer)} 字符)")
            else:
                print(answer)

        state = result

    # 对话统计
    print(f"\n{'═' * 64}")
    print("【对话统计】")
    print(f"  总消息数: {len(state['messages'])}")
    msg_types = {}
    for msg in state["messages"]:
        t = type(msg).__name__
        msg_types[t] = msg_types.get(t, 0) + 1
    for t, count in msg_types.items():
        print(f"  {t}: {count} 条")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="三国演义智能问答系统 - 三国通")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量索引（忽略已有缓存）")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
