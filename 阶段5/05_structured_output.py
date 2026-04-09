"""
阶段5 - 05_structured_output.py
结构化输出（Structured Output）

核心概念：
- with_structured_output(): 让 LLM 原生输出符合 Schema 的结构化数据
- 与 OutputParser 的区别: with_structured_output 在模型层约束（更可靠）
- 支持两种模式: JSON Mode (dict) / Tool Calling (Pydantic)

LangChain 1.x 推荐的结构化输出方式：
    旧方式: Prompt → LLM → OutputParser (后处理，可能失败)
    新方式: LLM.with_structered_output(schema) (模型原生保证格式)

本示例演示：
- Pydantic 模式：强类型对象输出
- JSON Schema 模式：灵活字典输出
- 多种 schema 切换
- 与 Chain 集成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Literal
from pydantic import BaseModel, Field
from tools import make_ollama


# ========== Part 1: Pydantic 结构化输出 ==========

class ProductAnalysis(BaseModel):
    """产品分析结果"""
    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    price_range: str = Field(description="价格区间")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    score: int = Field(description="综合评分 (1-10)")
    recommendation: Literal["强烈推荐", "推荐", "一般", "不推荐"] = Field(
        description="购买建议"
    )


class SentimentResult(BaseModel):
    """情感分析结果"""
    text: str = Field(description="原始文本")
    sentiment: Literal["正面", "负面", "中性"] = Field(description="情感倾向")
    confidence: float = Field(description="置信度 (0-1)")
    keywords: List[str] = Field(description="关键词列表")


def demo_pydantic_structured():
    """with_structured_output + Pydantic Model"""
    print("=" * 50)
    print("Part 1: Pydantic 结构化输出")
    print("=" * 50 + "\n")

    llm = make_ollama()

    # 绑定 Pydantic schema - 输出自动转为 ProductAnalysis 实例
    structured_llm = llm.with_structured_output(ProductAnalysis)

    query = """
    iPhone 15 Pro Max 是苹果2023年发布的旗舰手机，
    采用钛金属边框、A17 Pro 芯片、4800万像素主摄。
    起售价9999元。优点是性能强劲、拍照出色、生态系统完善；
    缺点是价格昂贵、充电速度慢、没有USB-C（实际已改用）。
    """

    print(f"输入: 关于 iPhone 的描述\n")
    result: ProductAnalysis = structured_llm.invoke(query)

    print("输出 (ProductAnalysis 对象):")
    print(f"  名称:   {result.name}")
    print(f"  类别:   {result.category}")
    print(f"  价格:   {result.price_range}")
    print(f"  优点:   {', '.join(result.pros)}")
    print(f"  缺点:   {', '.join(result.cons)}")
    print(f"  评分:   {result.score}/10")
    print(f"  建议:   {result.recommendation}")
    print(f"\n  类型检查: isinstance(result, ProductAnalysis) = {isinstance(result, ProductAnalysis)}")


def demo_json_structured():
    """with_structured_output + JSON Schema (输出 dict)"""
    print("\n" + "=" * 50)
    print("Part 2: JSON Schema 结构化输出")
    print("=" * 50 + "\n")

    llm = make_ollama()

    # 用 JSON Schema 定义输出结构（返回 dict）
    json_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名"},
            "population_million": {"type": "number", "description": "人口(百万)"},
            "attractions": {"type": "array", "items": {"type": "string"}, "description": "景点列表"},
            "best_visit_season": {"type": "string", "description": "最佳旅游季节"},
        },
        "required": ["city", "population_million", "attractions"],
    }

    structured_llm = llm.with_structured_output(json_schema)

    result: dict = structured_llm.invoke("介绍一下杭州这座城市")
    print("输出 (dict):")
    for key, value in result.items():
        if isinstance(value, list):
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value}")


# ========== Part 3: 情感分析实战 ==========

def demo_sentiment_analysis():
    """批量文本情感分析"""
    print("\n" + "=" * 50)
    print("Part 3: 批量情感分析")
    print("=" * 50 + "\n")

    llm = make_ollama()
    structured_llm = llm.with_structured_output(SentimentResult)

    texts = [
        "这家餐厅的服务太棒了！菜品美味，环境优雅，下次还会来。",
        "产品质量很差，用了两天就坏了，客服态度也不好。",
        "今天天气不错，温度适中，适合出门散步。",
    ]

    for text in texts:
        result: SentimentResult = structured_llm.invoke(text)
        bar = "█" * int(result.confidence * 20) + "░" * (20 - int(result.confidence * 20))
        print(f"  [{result.sentiment}] ({bar} {result.confidence:.0%})")
        print(f"    关键词: {', '.join(result.keywords)}")
        print()


def main():
    demo_pydantic_structured()
    demo_json_structured()
    demo_sentiment_analysis()


if __name__ == "__main__":
    main()
