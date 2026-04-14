"""
阶段5 - 04_error_retry.py
错误处理与重试机制

核心概念：
- @retry 装饰器：自动重试失败的 LLM 调用
- with_fallbacks：多模型故障转移（主模型挂了切备用）
- try/except 包裹：精确控制异常处理策略
- RunnableRetry：链级别的重试配置

生产环境中 LLM API 可能出现的问题：
1. 速率限制 (429 Rate Limit)
2. 服务暂时不可用 (503)
3. 网络超时 (Timeout)
4. 输出格式错误（需要重新解析）

本示例演示：
- 单次调用重试
- 链级别重试与退避
- 多模型 fallback
- 自定义异常处理
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import random
from typing import Optional
from tools import make_ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ========== Part 1: 手动重试 ==========

def demo_manual_retry():
    """手动 try/except 重试"""
    print("=" * 50)
    print("Part 1: 手动重试 (try/except)")
    print("=" * 50 + "\n")

    llm = make_ollama()
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  第 {attempt}/{max_retries} 次尝试...", end=" ")
            result = llm.invoke("说 OK")
            print(f"成功! 回复: {result.content[:30]}")
            return result.content
        except Exception as e:
            print(f"失败: {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(1 * attempt)  # 指数退避
    print("  已达到最大重试次数")


# ========== Part 2: RunnableWithRetry 链级重试 ==========

def demo_chain_retry():
    """使用 RunnableParallel / RunnableWithRetry 进行链级重试"""
    print("\n" + "=" * 50)
    print("Part 2: 链级重试 (.with_retry())")
    print("=" * 50 + "\n")

    from langchain_core.runnables import RunnableLambda

    llm = make_ollama()
    prompt = ChatPromptTemplate.from_template("回答: {question}")
    chain = prompt | llm | StrOutputParser()

    # 为整个链添加重试
    retryable_chain = chain.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
        exponential_jitter_params={"max": 10},
    )

    print("  配置: 最多3次重试, 指数退避最大等待10秒\n")
    result = retryable_chain.invoke({"question": "1+1等于几?"})
    print(f"  结果: {result}")


# ========== Part 3: Fallback 多模型容灾 ==========

def demo_fallback():
    """多模型 fallback：主模型失败时自动切换备用模型"""
    print("\n" + "=" * 50)
    print("Part 3: 多模型 Fallback (with_fallbacks)")
    print("=" * 50 + "\n")

    primary = make_ollama()

    # 备用模型（可以换成不同的 provider/model）
    fallback = make_ollama()

    prompt = ChatPromptTemplate.from_template("{input}")
    chain = prompt | primary

    # 绑定 fallback 链
    chain_with_fb = chain.with_fallbacks([prompt | fallback])

    print("  主模型: make_ollama()")
    print("  备用:   make_ollama() (fallback)\n")

    try:
        result = chain_with_fb.invoke({"input": "你好"})
        print(f"  成功回复: {result.content[:50]}...")
    except Exception as e:
        print(f"  所有模型均失败: {e}")


# ========== Part 4: 模拟不稳定 API + 重试 ==========

class UnreliableLLM:
    """模拟不稳定的 LLM（随机失败）"""

    def __init__(self, failure_rate=0.6):
        self.failure_rate = failure_rate
        self.call_count = 0
        self.real_llm = make_ollama()

    def invoke(self, text, config=None):
        self.call_count += 1
        if random.random() < self.failure_rate:
            raise ConnectionError(f"[模拟] API 临时不可用 (第{self.call_count}次)")
        return self.real_llm.invoke(text)


def demo_unstable_api():
    """模拟不稳定 API 展示重试效果"""
    print("\n" + "=" * 50)
    print("Part 4: 不稳定 API + 自动重试")
    print("=" * 50 + "\n")

    from langchain_core.runnables import RunnableLambda

    unreliable = UnreliableLLM(failure_rate=0.7)
    # 用 RunnableLambda 包装，使其获得 .with_retry() 等 Runnable 方法
    unreliable_runnable = RunnableLambda(unreliable.invoke)
    retry_chain = unreliable_runnable.with_retry(stop_after_attempt=5)

    print(f"  模拟 API: 70% 失败率")
    print(f"  重试策略: 最多5次\n")

    start = time.time()
    try:
        result = retry_chain.invoke("说 hello")
        elapsed = time.time() - start
        print(f"  最终成功! 用时 {elapsed:.1f}s, 共调用 {unstable.call_count} 次")
        print(f"  回复: {result.content}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  全部失败 ({elapsed:.1f}s): {e}")


def main():
    demo_manual_retry()
    demo_chain_retry()
    demo_fallback()
    demo_unstable_api()


if __name__ == "__main__":
    main()
