"""
阶段6 - 02_chatbot_with_memory.py
AI 聊天机器人实战 - 带记忆的对话系统

综合运用阶段1-5的知识：
- LangGraph 状态图 + 条件边 + 循环路由（阶段4）
- 对话记忆（阶段3）
- 工具调用 Agent（阶段4）
- 流式输出（阶段5）
- 结构化输出 - 意图识别（阶段5）

项目架构：
1. 意图识别（结构化输出）→ 路由到不同处理节点
2. 闲聊节点 → 直接 LLM 回复
3. 知识问答节点 → RAG 检索增强
4. 工具调用节点 → 执行计算/搜索等操作
5. 全程带对话记忆

图结构：

              ┌──────────────┐
              │    START      │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  意图识别     │  ← with_structured_output
              └──────┬───────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
  ┌───────────┐ ┌──────────┐ ┌───────────┐
  │   闲聊     │ │ 知识问答  │ │  工具调用  │
  │  (chat)   │ │  (rag)   │ │  (tools)  │
  └─────┬─────┘ └────┬─────┘ └─────┬─────┘
        │            │             │
        └────────────┼─────────────┘
                     ▼
              ┌──────────────┐
              │     END       │
              └──────────────┘
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import TypedDict, Annotated, Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from tools import make_ollama


# ========== Part 1: 工具定义 ==========

@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户询问时间、日期时使用。"""
    from datetime import datetime
    now = datetime.now()
    weekdays = "一二三四五六日"
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S 星期{weekdays[now.weekday()]}")


@tool
def calculate(expression: str) -> str:
    """执行数学计算。当用户需要算术运算时使用。

    Args:
        expression: 数学表达式，如 "(100 + 200) * 3"
    """
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "错误：包含非法字符"
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_knowledge(query: str) -> str:
    """搜索 Python 编程知识库。当用户询问 Python 编程相关问题时使用。

    Args:
        query: 搜索关键词或问题
    """
    # 尝试加载知识库
    try:
        persist_dir = str(Path(__file__).parent / "faiss_index" / "python_faq")
        if Path(persist_dir).exists():
            embeddings = OllamaEmbeddings(model="bge-m3:latest")
            vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            docs = vectorstore.similarity_search(query, k=3)
            if docs:
                return "\n---\n".join(doc.page_content for doc in docs)
        return "知识库暂未构建，请先运行 01_rag_qa_system.py 构建索引"
    except Exception as e:
        return f"搜索失败: {e}"


TOOLS = [get_current_time, calculate, search_knowledge]
tool_map = {t.name: t for t in TOOLS}


# ========== Part 2: 意图识别（结构化输出）==========

class IntentResult(BaseModel):
    """用户意图识别结果"""
    intent: Literal["chat", "knowledge", "tool"] = Field(
        description="用户意图类型：chat=闲聊/打招呼, knowledge=编程知识问答, tool=需要使用工具（计算/查时间等）"
    )
    reasoning: str = Field(description="判断理由")
    tool_name: str = Field(
        default="",
        description="如果 intent=tool，指定工具名称（get_current_time/calculate/search_knowledge），否则为空"
    )


# ========== Part 3: 状态定义 ==========

class ChatbotState(TypedDict):
    """聊天机器人状态"""
    messages: Annotated[list, add_messages]    # 对话历史
    intent: str                                  # 识别的意图
    reasoning: str                               # 意图判断理由
    tool_name: str                               # 需要调用的工具


# ========== Part 4: 节点函数 ==========

def node_classify_intent(state: ChatbotState) -> dict:
    """意图识别节点 - 使用结构化输出判断用户意图"""
    llm = make_ollama()
    structured_llm = llm.with_structured_output(IntentResult)

    messages = state["messages"]
    # 取最近一条用户消息
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    prompt = (
        f"请判断以下用户消息的意图类型：\n"
        f"用户消息: {last_user_msg}\n\n"
        f"规则：\n"
        f"- chat: 打招呼、闲聊、感谢、告别等非技术对话\n"
        f"- knowledge: 询问 Python 编程知识、概念、语法等\n"
        f"- tool: 需要计算、查询时间等需要工具操作"
    )

    result: IntentResult = structured_llm.invoke(prompt)
    print(f"  [意图] {result.intent} | 理由: {result.reasoning}")

    return {
        "intent": result.intent,
        "reasoning": result.reasoning,
        "tool_name": result.tool_name,
    }


def node_chat(state: ChatbotState) -> dict:
    """闲聊节点 - 直接 LLM 回复"""
    llm = make_ollama()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "你是一个友好、热情的 Python 编程助手，名叫「小派」。\n"
            "你的职责是帮助用户学习 Python 编程。\n"
            "对于闲聊，友好回应并引导到编程话题。"
        )),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": state["messages"]})
    print(f"  [闲聊] 回复: {response[:60]}...")
    return {"messages": [AIMessage(content=response)]}


def node_knowledge(state: ChatbotState) -> dict:
    """知识问答节点 - RAG 检索增强"""
    llm = make_ollama()

    # 获取最后一条用户消息
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 搜索知识库
    search_result = search_knowledge.invoke({"query": last_user_msg})

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "你是一个专业的 Python 编程助手。请基于以下知识库内容回答用户问题。\n"
            "如果知识库内容不足，可以适当补充，但标注「补充说明」。\n"
            "回答要简洁，包含代码示例（如果适用）。"
        )),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="知识库内容:\n{context}\n\n问题: {question}"),
    ])

    # 提取历史（不含最后一条）
    history = [m for m in state["messages"][:-1]
               if isinstance(m, (HumanMessage, AIMessage))]

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "history": history,
        "context": search_result,
        "question": last_user_msg,
    })
    print(f"  [知识] 回复: {response[:60]}...")
    return {"messages": [AIMessage(content=response)]}


def node_tool_call(state: ChatbotState) -> dict:
    """工具调用节点 - 执行工具并生成回复"""
    tool_name = state.get("tool_name", "")
    llm = make_ollama()

    # 获取最后一条用户消息
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 用 LLM + 绑定工具来决定调用
    bound_llm = llm.bind_tools(TOOLS)
    messages = state["messages"]
    response = bound_llm.invoke(messages)

    # 如果有工具调用，执行工具
    new_messages = [response]
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            print(f"  [工具] 调用: {name}({args})")

            if name in tool_map:
                result = tool_map[name].invoke(args)
            else:
                result = f"未知工具: {name}"

            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        # 再次调用 LLM 生成最终回复
        all_messages = list(state["messages"]) + new_messages
        final_response = llm.invoke(all_messages)
        new_messages.append(final_response)
        print(f"  [工具] 最终回复: {final_response.content[:60]}...")
    else:
        print(f"  [工具] 直接回复: {response.content[:60] if response.content else '(空)'}...")

    return {"messages": new_messages}


# ========== Part 5: 条件路由 ==========

def route_by_intent(state: ChatbotState) -> str:
    """根据意图路由到不同节点"""
    intent = state.get("intent", "chat")
    route_map = {
        "chat": "chat_node",
        "knowledge": "knowledge_node",
        "tool": "tool_node",
    }
    target = route_map.get(intent, "chat_node")
    print(f"  [路由] → {target}")
    return target


# ========== Part 6: 构建图 ==========

def build_chatbot_graph():
    """构建聊天机器人 LangGraph"""
    graph = StateGraph(ChatbotState)

    # 添加节点
    graph.add_node("classify", node_classify_intent)
    graph.add_node("chat_node", node_chat)
    graph.add_node("knowledge_node", node_knowledge)
    graph.add_node("tool_node", node_tool_call)

    # 添加边
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "chat_node": "chat_node",
            "knowledge_node": "knowledge_node",
            "tool_node": "tool_node",
        },
    )
    graph.add_edge("chat_node", END)
    graph.add_edge("knowledge_node", END)
    graph.add_edge("tool_node", END)

    return graph.compile()


# ========== Part 7: 主函数 ==========

def main():
    print("=" * 60)
    print("阶段6 - AI 聊天机器人实战（带记忆 + 意图识别）")
    print("=" * 60)

    app = build_chatbot_graph()

    # 打印图结构
    try:
        print("\n【图结构 Mermaid】\n")
        print(app.get_graph().draw_mermaid())
    except Exception:
        pass

    # 系统消息
    system_msg = SystemMessage(content=(
        "你是小派，一个 Python 编程助手。"
    ))

    # 测试对话
    conversations = [
        "你好！你是谁呀？",
        "Python 中装饰器怎么用？",
        "帮我算一下 (256 + 128) * 4",
        "现在几点了？",
        "深拷贝和浅拷贝有什么区别？",
        "谢谢你的帮助！",
    ]

    print("\n" + "=" * 60)
    print("开始对话\n")

    state = {"messages": [system_msg]}

    for user_input in conversations:
        print(f"\n👤 用户: {user_input}")
        state["messages"].append(HumanMessage(content=user_input))

        # 重置意图状态
        state["intent"] = ""
        state["reasoning"] = ""
        state["tool_name"] = ""

        result = app.invoke(state)

        # 获取最后的 AI 回复
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            last_ai = ai_messages[-1]
            print(f"🤖 小派: {last_ai.content}")

        # 更新状态
        state = result

    # 对话统计
    print(f"\n{'═' * 60}")
    print(f"【对话统计】共 {len(state['messages'])} 条消息")
    msg_types = {}
    for msg in state["messages"]:
        t = type(msg).__name__
        msg_types[t] = msg_types.get(t, 0) + 1
    for t, count in msg_types.items():
        print(f"  {t}: {count} 条")


if __name__ == "__main__":
    main()
