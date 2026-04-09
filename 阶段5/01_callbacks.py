"""
阶段5 - 01_callbacks.py
回调与监控系统（Callbacks）

核心概念：
- BaseCallbackHandler：自定义回调处理器基类
- 回调钩子点：on_llm_start, on_llm_end, on_tool_start, on_chain_start 等
- 可观测性：监控 LLM 调用、Token 消耗、执行时间

LangChain 的回调系统让你可以"监听"整个链路执行的每个环节，
非常适合用于日志记录、性能监控和调试。

回调类型：
1. 同步回调 (callbacks=[]) - 用于 invoke()
2. 异步回调 (async_callbacks=[]) - 用于 ainvoke()

本示例实现：
- Token 计数回调
- 执行时间追踪回调
- 完整事件日志回调
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time, json
from tools import make_ollama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ========== 自定义回调处理器 ==========

class TokenCountHandler(BaseCallbackHandler):
    """Token 计数回调 - 统计 LLM 输入输出的 token 数量"""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 返回结果时触发"""
        self.call_count += 1
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, 'token_usage') and gen.token_usage:
                    usage = gen.token_usage
                    self.total_input_tokens += usage.get('prompt_tokens', 0)
                    self.total_output_tokens += usage.get('completion_tokens', 0)

    def summary(self):
        return (
            f"\n📊 Token 统计:\n"
            f"   调用次数: {self.call_count}\n"
            f"   输入 Token: {self.total_input_tokens}\n"
            f"   输出 Token: {self.total_output_tokens}"
        )


class TimingHandler(BaseCallbackHandler):
    """时间追踪回调 - 记录每个步骤的耗时"""

    def __init__(self):
        self.timings = {}
        self._start_times = {}
        self._counter = 0

    def on_chain_start(self, serialized, inputs, **kwargs):
        # serialized 可能是 None 或不含 name 的对象
        name = getattr(serialized, "name", None) or ({} if serialized is None else serialized).get("name", f"chain_{self._counter}")
        self._counter += 1
        self._current_name = name
        self._start_times[name] = time.perf_counter()

    def on_chain_end(self, outputs, **kwargs):
        # 新版 on_chain_end 第一个位置参数是 outputs 而非 serialized
        name = getattr(self, '_current_name', 'unknown')
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.timings[name] = round(elapsed * 1000, 2)

    def summary(self):
        lines = ["\n⏱️  执行耗时:"]
        for name, ms in self.timings.items():
            lines.append(f"   {name}: {ms:.1f}ms")
        return "\n".join(lines)


class EventLogHandler(BaseCallbackHandler):
    """完整事件日志回调 - 打印所有事件（用于调试）"""

    # 始终启用所有回调
    def __init__(self):
        self.events = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._log("LLM_START", serialized, {"prompt_length": len(prompts[0]) if prompts else 0})

    def on_llm_end(self, response, **kwargs):
        text = response.generations[0][0].text if response.generations else ""
        self._log("LLM_END", {}, {"output_preview": text[:80]})

    def on_chain_start(self, serialized, inputs, **kwargs):
        self._log("CHAIN_START", serialized, {})

    def on_chain_end(self, serialized, outputs, **kwargs):
        self._log("CHAIN_END", serialized, {})

    def _log(self, event_type, serialized, extra):
        entry = {
            "type": event_type,
            "name": serialized.get("name", ""),
            **extra,
        }
        self.events.append(entry)
        print(f"  🔔 [{event_type}] {serialized.get('name', '')} | {extra}")


def main():
    llm = make_ollama()

    print("=" * 60)
    print("阶段5 - 回调与监控（Callbacks）")
    print("=" * 60)

    # ---- Part 1: Token 计数 ----
    print("\n【Part 1】Token 计数回调\n")
    token_handler = TokenCountHandler()

    result1 = llm.invoke(
        "用一句话解释什么是机器学习",
        config={"callbacks": [token_handler]}
    )
    print(f"回复: {result1.content}")

    result2 = llm.invoke(
        "列出 Python 的三个主要优点",
        config={"callbacks": [token_handler]}
    )
    print(f"回复: {result2.content}")
    print(token_handler.summary())

    # ---- Part 2: 时间追踪 ----
    print("\n" + "-" * 40)
    print("\n【Part 2】执行时间追踪\n")
    timing_handler = TimingHandler()

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template("请回答: {question}")
    chain = prompt | llm | StrOutputParser()

    chain.invoke(
        {"question": "什么是 LangChain？"},
        config={"callbacks": [timing_handler]}
    )
    print(timing_handler.summary())

    # ---- Part 3: 事件日志 ----
    print("\n" + "-" * 40)
    print("\n【Part 3】完整事件日志\n")
    event_handler = EventLogHandler()

    llm.invoke(
        "说 hello",
        config={"callbacks": [event_handler]}
    )

    print(f"\n共捕获 {len(event_handler.events)} 个事件")


if __name__ == "__main__":
    main()
