"""
阶段6 - 三国演义智能问答 API 服务
====================================
基于 04_sg_expert_demo.py 的 FastAPI Web API + Vue 前端单页应用

启动方式:
    python 阶段6/05_sg_api_server.py [--rebuild] [--port 8000]

访问: http://localhost:8000
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
import re
import uuid
import threading
import time
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
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
from langchain_core.tools import tool

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tools import make_ollama

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sg_api")


# ==================== 配置 ====================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sg"
FAISS_DIR = BASE_DIR / "faiss_index" / "sg_novel"
STATIC_DIR = BASE_DIR / "static"

# 全局变量（应用启动时初始化）
_vectorstore = None
_app_graph = None
_invoke_app = None


# ==================== Part 1: 向量索引构建 ====================

def build_sg_vectorstore(rebuild: bool = False) -> FAISS:
    """构建三国演义向量数据库（支持持久化 + 索引完整性检查）"""
    chapter_files = sorted(DATA_DIR.glob("chapter_*.txt"), key=lambda f: int(re.search(r'\d+', f.name).group()))
    total_chapters = len(chapter_files)

    index_meta_file = FAISS_DIR / ".meta.json"
    if not rebuild and FAISS_DIR.exists() and index_meta_file.exists():
        print("  [索引] 发现已有索引，进行完整性校验...")
        try:
            with open(index_meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cached_chapters = meta.get("total_chapters", 0)
            print(f"  [索引-缓存] {cached_chapters} 章 | [索引-当前] {total_chapters} 章")
            if cached_chapters == total_chapters:
                print("  [索引] 章节数量匹配，直接加载缓存 ✓")
                embeddings = OllamaEmbeddings(model="bge-m3:latest")
                vs = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
                print(f"  [索引] 向量数据库就绪，共 {vs.index.ntotal} 条记录")
                return vs
        except Exception as e:
            print(f"  [索引] 缓存损坏 ({e})，重建")

    action = "强制重建" if rebuild else ("重建" if FAISS_DIR.exists() else "首次构建")
    print(f"  [索引] {action}三国演义向量数据库... ({total_chapters} 章)")

    all_docs = []
    for idx, cf in enumerate(chapter_files):
        loader = TextLoader(str(cf), encoding="utf-8")
        docs = loader.load()
        match = re.search(r'chapter_(\d+)\.txt', cf.name)
        if match:
            ch_num = int(match.group(1))
            for d in docs:
                d.metadata["chapter"] = ch_num
                d.metadata["title"] = d.page_content.split('\n')[0].strip()
        all_docs.extend(docs)
        if (idx + 1) % 20 == 0 or (idx + 1) == total_chapters:
            print(f"  [索引] 已加载 {idx + 1}/{total_chapters} 章...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        length_function=len, separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"  [索引] 分割为 {len(splits)} 个文本块")

    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    FAISS_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    with open(index_meta_file, "w", encoding="utf-8") as f:
        json.dump({"total_chapters": total_chapters, "total_records": vectorstore.index.ntotal}, f)
    print(f"  [索引] 完成，共 {vectorstore.index.ntotal} 条记录")
    return vectorstore


# ==================== Part 2: 工具定义 ====================

@tool
def search_chapters(keyword: str, top_k: int = 3) -> str:
    """搜索三国演义章节内容"""
    try:
        emb = OllamaEmbeddings(model="bge-m3:latest")
        vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
        docs = vs.similarity_search(keyword, k=top_k)
        results = []
        for doc in docs:
            ch = doc.metadata.get("chapter", "?")
            title = doc.metadata.get("title", "")
            preview = doc.page_content[:300].replace("\n", " ")
            results.append(f"【第{ch}回】{title}\n{preview}...")
        return "\n\n---\n\n".join(results) if results else "未找到相关内容"
    except Exception as e:
        return f"搜索失败: {e}"

@tool
def analyze_character(character_name: str) -> str:
    """分析三国人物"""
    try:
        emb = OllamaEmbeddings(model="bge-m3:latest")
        vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
        queries = [f"{character_name}的性格", f"{character_name}的事迹", f"{character_name}出场"]
        all_docs, seen = [], set()
        for q in queries:
            for d in vs.similarity_search(q, k=2):
                if d.metadata.get("chapter") not in seen:
                    seen.add(d.metadata.get("chapter"))
                    all_docs.append(d)
        results = [f"[第{d.metadata.get('channel','?')}] {d.page_content[:250]}..." for d in all_docs[:4]]
        return f"【{character_name}相关】\n" + "\n\n".join(results) or "未找到记载"
    except Exception as e:
        return f"分析失败: {e}"

@tool
def find_event(event_desc: str) -> str:
    """查找三国事件"""
    try:
        emb = OllamaEmbeddings(model="bge-m3:latest")
        vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
        docs = vs.similarity_search(event_desc, k=5)
        seen, results = set(), []
        for doc in docs:
            ch = doc.metadata.get("chapter", "?")
            if ch not in seen:
                seen.add(ch)
                results.append(f"【第{ch}回】{doc.metadata.get('title','')}\n{doc.page_content[:200].replace(chr(10),' ')}...")
        return f"【事件：{event_desc}】\n" + "\n\n".join(results[:4]) or "未找到"
    except Exception as e:
        return f"查找失败: {e}"

@tool
def compare_forces(force_a: str = "", force_b: str = "") -> str:
    """对比势力"""
    try:
        emb = OllamaEmbeddings(model="bge-m3:latest")
        vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
        docs = vs.similarity_search(f"{force_a} {force_b} 对比 势力", k=4)
        results = [f"[第{d.metadata.get('channel','?')}] {d.page_content[:250]}..." for d in docs[:3]]
        return f"【{force_a} vs {force_b}】\n" + "\n\n".join(results) or "暂无信息"
    except Exception as e:
        return f"对比失败: {e}"

TOOLS = [search_chapters, analyze_character, find_event, compare_forces]
tool_map = {t.name: t for t in TOOLS}


# ==================== Part 3-7: 意图识别 / 状态 / 节点 / 路由 / 图（同 04）====================

class SGIntentResult(BaseModel):
    intent: Literal["chat", "retrieval", "character", "event", "analysis"] = Field(
        description="-chat:闲聊 -retrieval:原文引用 -character:人物分析 -event:事件查询 -analysis:综合分析"
    )
    reasoning: str = Field(description="判断理由")
    tool_args: dict = Field(default_factory=dict, description="工具参数")

class SGState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str; reasoning: str; tool_args: dict; vectorstore: object

def node_classify(state: SGState) -> dict:
    logger.info("[classify] 开始意图分类...")
    t0 = time.time()
    logger.info(f"[classify] 1/4 创建 Ollama LLM 客户端...")
    llm = make_ollama()
    logger.info(f"[classify] 2/4 设置结构化输出 (with_structured_output)...")
    structured_llm = llm.with_structured_output(SGIntentResult)
    last_user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    logger.info(f"[classify] 3/4 调用 LLM 推理 (消息={last_user_msg[:60]}...)...")
    result: SGIntentResult = structured_llm.invoke(
        f"判断意图类型:\n用户消息: {last_user_msg}\n\n"
        "分类: chat=闲聊/招呼 retrieval=原文引用 character=人物 event=事件 analysis=综合对比"
    )
    logger.info(f"[classify] 4/4 结果: intent={result.intent}, 总耗时={time.time()-t0:.2f}s")
    return {"intent": result.intent, "reasoning": result.reasoning, "tool_args": result.tool_args or {}}

def node_chat(state: SGState) -> dict:
    logger.info("[chat_node] 闲聊节点处理中...")
    t0 = time.time()
    llm = make_ollama()
    chain = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是「三国通」，热爱三国演义的AI助手。说话风趣幽默，喜欢用典故打比方，适当引用原文诗词。"),
        MessagesPlaceholder(variable_name="messages"),
    ]) | llm | StrOutputParser()
    resp = chain.invoke({"messages": state["messages"]})
    logger.info(f"[chat_node] 完成, 耗时={time.time()-t0:.2f}s")
    return {"messages": [AIMessage(content=resp)]}

def node_retrieval(state: SGState) -> dict:
    logger.info("[retrieval_node] 原文检索节点处理中...")
    t0 = time.time()
    llm = make_ollama()
    last_user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    vs = state.get("vectorstore")
    docs = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}).invoke(last_user_msg) if vs else []
    logger.info(f"[retrieval_node] 检索到 {len(docs)} 个相关片段")
    ctx = "\n---\n".join([f"【第{d.metadata.get('channel','?')}】{d.page_content[:400]}" for d in docs[:4]]) if docs else search_chapters.invoke({"keyword": last_user_msg, "top_k": 5})
    chain = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是「三国通」学者。基于原文回答，标注出处(第几回)，有文采地引用精彩语句。"),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="原著原文:\n{context}\n\n问题:{question}")
    ]) | llm | StrOutputParser()
    history = [m for m in state["messages"][:-1] if isinstance(m, (HumanMessage,AIMessage))]
    resp = chain.invoke({"history":history, "context":ctx, "question":last_user_msg})
    logger.info(f"[retrieval_node] 完成, 耗时={time.time()-t0:.2f}s")
    return {"messages": [AIMessage(content=resp)]}

def node_char_event(state: SGState) -> dict:
    logger.info("[char_event_node] 人物/事件节点处理中...")
    t0 = time.time()
    llm = make_ollama()
    bound_llm = llm.bind_tools(TOOLS)
    msgs = [SystemMessage(content="你是「三国通」专家。调用工具获取信息后给出专业分析。"), *state["messages"]]
    response = bound_llm.invoke(msgs)
    new_msgs = [response]
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"[char_event_node] LLM 调用了 {len(response.tool_calls)} 个工具")
        for tc in response.tool_calls:
            name, args = tc["name"], tc["args"]
            new_msgs.append(ToolMessage(content=str(tool_map[name](args)) if name in tool_map else "未知", tool_call_id=tc["id"]))
        chain = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是「三国通」。基于工具信息给出专业深入分析，结合历史背景，适当引用原文诗词。"),
            MessagesPlaceholder(variable_name="all_msgs")
        ]) | llm | StrOutputParser()
        resp = chain.invoke({"all_msgs": list(state["messages"])+new_msgs})
        logger.info(f"[char_event_node] 完成(工具路径), 耗时={time.time()-t0:.2f}s")
        return {"messages": [AIMessage(content=resp)]}
    else:
        chain = ChatPromptTemplate.from_messages([
            SystemMessage(content="你是「三国通」，回答人物或事件问题。"),
            MessagesPlaceholder(variable_name="messages")
        ]) | llm | StrOutputParser()
        resp = chain.invoke({"messages": state["messages"]})
        logger.info(f"[char_event_node] 完成(直接回复), 耗时={time.time()-t0:.2f}s")
        return {"messages": [AIMessage(content=resp)]}

def node_analysis(state: SGState) -> dict:
    logger.info("[analysis_node] 综合分析节点处理中...")
    t0 = time.time()
    llm = make_ollama()
    last_user_msg = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    bound_llm = llm.bind_tools(TOOLS)
    exec_msgs = [SystemMessage(content="三国演义分析执行助手，使用工具逐步分析。"), HumanMessage(content=f"需求:{last_user_msg}")]
    exec_resp = bound_llm.invoke(exec_msgs)
    new_msgs = [exec_resp]
    step_results = []
    if hasattr(exec_resp, 'tool_calls') and exec_resp.tool_calls:
        for tc in exec_resp.tool_calls:
            name, args = tc["name"], tc["args"]
            r = str(tool_map[name](args)) if name in tool_map else "未知"
            step_results.append(r[:150])
            new_msgs.append(ToolMessage(content=r, tool_call_id=tc["id"]))
    if step_results:
        chain = ChatPromptTemplate.from_messages([
            SystemMessage(content="「三国通」首席分析师。撰写专业报告：1.核心观点 2.多维度分析 3.原文佐证 4.独到见解 5.语言精炼有文采"),
            MessagesPlaceholder(variable_name="msgs")
        ]) | llm | StrOutputParser()
        report = chain.invoke({"msgs": exec_msgs+new_msgs})
    else:
        report = (ChatPromptTemplate.from_messages([
            SystemMessage(content="「三国通」首席分析师，深入分析问题。"),
            MessagesPlaceholder(variable_name="messages")
        ]) | llm | StrOutputParser()).invoke({"messages":state["messages"]})
    logger.info(f"[analysis_node] 完成, 耗时={time.time()-t0:.2f}s")
    return {"messages": [AIMessage(content=report)]}

def route_by_intent(state: SGState) -> str:
    intent = state.get("intent", "unknown")
    target = {"chat":"chat_node","retrieval":"retrieval_node","character":"char_event_node","event":"char_event_node","analysis":"analysis_node"}.get(intent,"chat_node")
    logger.info(f"[route] intent={intent} → {target}")
    return target

def build_sg_graph(vectorstore=None):
    graph = StateGraph(SGState)
    graph.add_node("classify", node_classify)
    graph.add_node("chat_node", node_chat)
    graph.add_node("retrieval_node", node_retrieval)
    graph.add_node("char_event_node", node_char_event)
    graph.add_node("analysis_node", node_analysis)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_by_intent, {"chat_node":"chat_node","retrieval_node":"retrieval_node","char_event_node":"char_event_node","analysis_node":"analysis_node"})
    graph.add_edge("chat_node",END); graph.add_edge("retrieval_node",END)
    graph.add_edge("char_event_node",END); graph.add_edge("analysis_node",END)
    app = graph.compile()
    def invoke_with_vs(s):
        logger.info(f"[Graph] invoke_with_vs 被调用, messages数量={len(s.get('messages',[]))}")
        t0 = time.time()
        result = app.invoke({**s, "vectorstore":vectorstore})
        logger.info(f"[Graph] app.invoke 返回, 耗时={time.time()-t0:.2f}s")
        return result
    return app, invoke_with_vs


# ==================== 会话管理 ====================

class SessionStore:
    """内存会话管理器"""
    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._lock = threading.RLock()  # 使用 RLock（可重入锁），避免 append→get_or_create 嵌套死锁

    def get_or_create(self, session_id: str) -> list[dict]:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = [
                    {"role": "system", "content": (
                        "你是「三国通」，一个精通三国演义的AI助手。"
                        "你对120回三国演义烂熟于心，能背诵其中的诗词和经典段落。"
                        "你说话既有学术功底，又生动有趣，喜欢用三国典故打比方。"
                    )}
                ]
            return self._sessions[session_id]

    def append(self, session_id: str, role: str, content: str):
        with self._lock:
            self.get_or_create(session_id).append({"role": role, "content": content})

    def get_history(self, session_id: str) -> list[dict]:
        return self.get_or_create(session_id)

    def clear(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)


sessions = SessionStore()


# ==================== Pydantic 请求模型 ====================

class ChatRequest(BaseModel):
    message: str = Field(description="用户输入的消息")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会话ID（留空自动生成）")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    reasoning: str


class StatusResponse(BaseModel):
    status: str
    total_chapters: int
    total_records: int


# ==================== FastAPI 应用生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vectorstore, _app_graph, _invoke_app
    logger.info("=" * 56)
    logger.info("  三国演义智能问答系统 - API Server 启动中...")
    logger.info("=" * 56)
    t0 = time.time()
    _vectorstore = build_sg_vectorstore()
    logger.info(f"\n[就绪] 构建问答图...")
    _app_graph, _invoke_app = build_sg_graph(_vectorstore)
    logger.info(f"[就绪] API 服务已就绪！总启动耗时={time.time()-t0:.1f}s")
    yield
    logger.info("[关闭] 服务停止")


app = FastAPI(title="三国通 - 三国演义智能问答API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# ==================== API 路由 ====================

@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    """获取服务状态和索引信息（含Ollama连通性检测）"""
    logger.info("[API] GET /api/status")
    meta = {}
    meta_file = FAISS_DIR / ".meta.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 快速检测 Ollama 是否可达
    ollama_ok = False
    ollama_model = ""
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(req.read().decode())
        models = data.get("models", [])
        if models:
            ollama_model = models[0].get("name", "unknown")
            ollama_ok = True
        logger.info(f"[Health] Ollama OK, 模型: {[m['name'] for m in models[:3]]}")
    except Exception as e:
        logger.warning(f"[Health] Ollama 不可达: {e}")

    return StatusResponse(status="ready" if ollama_ok else "ollama_unreachable",
                         total_chapters=meta.get("total_chapters", 0),
                         total_records=meta.get("total_records", 0))


@app.get("/api/test-llm")
async def api_test_llm():
    """快速测试 Ollama LLM 是否能正常响应（绕过 LangGraph）"""
    logger.info("[Test] 开始直接测试 Ollama LLM ...")
    t0 = time.time()
    try:
        llm = make_ollama()
        logger.info(f"[Test] LLM 对象创建完成, 耗时={time.time()-t0:.2f}s")
        t1 = time.time()
        resp = llm.invoke("用一句话回答：诸葛亮是谁？")
        elapsed = time.time() - t0
        logger.info(f"[Test] LLM 响应完成! 总耗时={elapsed:.2f}s, 回复={str(resp.content)[:100]}")
        return {"ok": True, "elapsed": round(elapsed, 2), "reply_preview": str(resp.content)[:200]}
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[Test] LLM 测试失败: {e} (耗时{elapsed:.2f}s)", exc_info=True)
        return {"ok": False, "error": str(e), "elapsed": round(elapsed, 2)}


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    """
    发送消息并获取 AI 回复

    支持的意图类型：
    - chat: 闲聊打招呼
    - retrieval: 引用原文检索
    - character: 人物性格事迹分析
    - event: 战役事件查询
    - analysis: 综合分析报告

    注意：使用同步函数（非async），uvicorn 会自动将请求放入线程池执行，
    避免 Ollama/LangGraph 的同步阻塞调用卡住事件循环。
    """
    t_start = time.time()
    sid = req.session_id or str(uuid.uuid4())
    logger.info(f"[API] >>> 收到请求 session={sid[:8]} message={req.message[:80]}")

    # 将历史消息转为 LangChain messages 格式
    history = sessions.get_history(sid)
    lc_messages = []
    for msg in history:
        role = msg["role"]
        if role == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif role == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    logger.info(f"[API] 历史消息 {len(lc_messages)} 条, 准备调用 LangGraph...")

    # 追加当前用户消息
    lc_messages.append(HumanMessage(content=req.message))
    logger.info(f"[API] step1: 已追加用户消息到lc_messages")
    sessions.append(sid, "user", req.message)
    logger.info(f"[API] step2: 已记录到会话")

    # 调用 LangGraph
    state = {
        "messages": lc_messages,
        "intent": "", "reasoning": "", "tool_args": {},
    }
    logger.info(f"[API] step3: 已构建state dict, 进入try块")

    try:
        logger.info(f"[API] 调用 _invoke_app ...")
        result = _invoke_app(state)
        logger.info(f"[API] LangGraph 返回成功")

        # 提取回复和意图
        intent = result.get("intent", "unknown")
        reasoning = result.get("reasoning", "")

        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        reply = ai_messages[-1].content if ai_messages else "抱歉，未能生成回复。"

        # 记录到会话
        sessions.append(sid, "assistant", reply)

        elapsed = time.time() - t_start
        logger.info(f"[API] <<< 响应完成 intent={intent} 耗时={elapsed:.2f}s reply长度={len(reply)}")

        return ChatResponse(session_id=sid, reply=reply, intent=intent, reasoning=reasoning)

    except Exception as e:
        elapsed = time.time() - t_start
        logger.error(f"[API] !!! 异常: {type(e).__name__}: {e} (耗时{elapsed:.2f}s)", exc_info=True)
        error_reply = f"处理请求时出错: {str(e)}"
        sessions.append(sid, "assistant", error_reply)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def api_clear_session(session_id: str):
    """清除指定会话"""
    sessions.clear(session_id)
    return {"ok": True, "message": f"会话 {session_id} 已清除"}


@app.get("/")
async def serve_frontend():
    """返回 Vue 前端页面"""
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(404, "前端文件不存在，请先运行 python scripts/build_frontend.py 或手动创建 static/index.html")
    return FileResponse(str(index_html))


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="三国通 API Server")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()
    uvicorn.run("05_sg_api_server:app", host=args.host, port=args.port, reload=False)
