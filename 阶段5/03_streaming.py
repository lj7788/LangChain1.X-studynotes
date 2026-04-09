"""
阶段5 - 03_streaming.py
流式输出（Streaming）

核心概念：
- stream() / astream(): 逐 token 输出，提升用户体验
- stream_events(): 更细粒度的事件级流
- 流式 vs 非流式: 内存占用、响应延迟、用户体验

适用场景：
- 聊天机器人（逐字显示）
- 长文本生成（避免长时间等待空白）
- 实时翻译、摘要

LangChain 1.x 支持三种流模式：
1. chain.stream()     → 同步，yield chunk
2. chain.astream()    → 异步，async yield chunk
3. chain.stream_events() → 同步，yield event dict（含元信息）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from tools import make_ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def demo_sync_stream():
    """同步流式输出 - stream()"""
    print("=" * 50)
    print("Part 1: 同步流式输出 (stream)")
    print("=" * 50 + "\n")

    llm = make_ollama()
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的四行短诗")
    chain = prompt | llm | StrOutputParser()

    print("📝 正在生成诗歌...\n")

    full_text = ""
    for i, chunk in enumerate(chain.stream({"topic": "编程"})):
        full_text += chunk
        print(chunk, end="", flush=True)  # 逐字符打印

    print(f"\n\n完整文本 ({len(full_text)} 字符):")
    print(f"  {full_text}")


async def demo_async_stream():
    """异步流式输出 - astream()"""
    print("\n" + "=" * 50)
    print("Part 2: 异步流式输出 (astream)")
    print("=" * 50 + "\n")

    llm = make_ollama()
    prompt = ChatPromptTemplate.from_template("用3句话介绍{topic}")
    chain = prompt | llm | StrOutputParser()

    print("🤖 正在生成介绍...\n")

    full_text = ""
    async for chunk in chain.astream({"topic": "LangChain"}):
        full_text += chunk
        print(chunk, end="", flush=True)

    print(f"\n\n共收到 {len(full_text)} 字符")


async def demo_stream_events():
    """事件级流式输出 - astream_events()（异步）"""
    print("\n" + "=" * 50)
    print("Part 3: 事件级流 (astream_events)")
    print("=" * 50 + "\n")

    llm = make_ollama()
    prompt = ChatPromptTemplate.from_template("说: {text}")
    chain = prompt | llm

    event_types_seen = set()

    print("📡 捕获所有事件:\n")
    async for event in chain.astream_events(
        {"text": "你好世界"},
        version="v2",
    ):
        event_type = event["event"]
        event_types_seen.add(event_type)

        # 简洁打印关键事件
        if "on_chat_model_stream" == event_type:
            content = event["data"]["chunk"].content
            if content:
                print(f"  [chunk] {content}", end="")
        elif event_type in ("on_chain_start", "on_chain_end"):
            name = event.get("name", "")
            tag = "START" if "start" in event_type else "END"
            print(f"\n  [{tag}] {name}")

    print(f"\n\n捕获到 {len(event_types_seen)} 种事件类型:")
    for t in sorted(event_types_seen):
        print(f"  • {t}")


def main():
    # Part 1: 同步流
    demo_sync_stream()

    # Part 2: 异步流
    asyncio.run(demo_async_stream())

    # Part 3: 事件级流（异步）
    asyncio.run(demo_stream_events())


if __name__ == "__main__":
    main()
