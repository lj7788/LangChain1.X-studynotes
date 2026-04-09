"""
阶段5 - 02_output_parsers.py
输出解析器（Output Parsers）

核心概念：
- Pydantic Output Parser: 将 LLM 输出强制转换为 Pydantic 对象
- JSON Parser: 解析 LLM 输出的 JSON 格式文本
- XML Parser: 提取 LLM 输出中的 XML 标签内容
- CommaSeparatedListParser: 将输出解析为列表

使用场景：
- 结构化数据提取（如从非结构化文本中提取实体）
- 表单填写、API 参数构建
- 确保输出格式符合下游系统要求

LangChain 1.x 推荐的解析器链：
    Prompt → LLM → OutputParser → 结构化数据
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import List
from pydantic import BaseModel, Field
from tools import make_ollama
from langchain_core.output_parsers import (
    PydanticOutputParser,
    JsonOutputParser,
    XMLOutputParser,
)
from langchain_core.prompts import PromptTemplate


# ========== Part 1: Pydantic 输出解析 ==========

class MovieReview(BaseModel):
    """电影评论的结构化表示"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分 (0-10)")
    genre: str = Field(description="电影类型")
    summary: str = Field(description="一句话剧情简介")
    recommend: bool = Field(description="是否推荐观看")


def demo_pydantic_parser():
    """演示 PydanticOutputParser - 最常用的结构化输出方式"""
    print("=" * 50)
    print("Part 1: Pydantic Output Parser")
    print("=" * 50 + "\n")

    parser = PydanticOutputParser(pydantic_object=MovieReview)

    prompt = PromptTemplate(
        template="分析以下电影描述，按要求提取信息。\n{format_instructions}\n\n电影描述: {query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = make_ollama()
    chain = prompt | llm | parser

    query = """
    《星际穿越》是一部由克里斯托弗·诺兰执导的科幻电影。
    该片讲述了地球环境恶化后，一队宇航员穿越虫洞寻找新家园的故事。
    影片在视觉效果和科学严谨性上获得了极高评价，
    被誉为近年来最好的科幻电影之一。
    """

    print("输入: 电影《星际穿越》的描述\n")
    result = chain.invoke({"query": query})

    print("解析结果 (MovieReview 对象):")
    print(f"  片名:   {result.title}")
    print(f"  评分:   {result.rating}/10")
    print(f"  类型:   {result.genre}")
    print(f"  简介:   {result.summary}")
    print(f"  推荐:   {'是' if result.recommend else '否'}")
    print(f"\n  类型: {type(result).__name__}")
    print(f"  可直接访问属性: result.rating = {result.rating}")


# ========== Part 2: JSON 输出解析 ==========

def demo_json_parser():
    """演示 JsonOutputParser - 灵活的 JSON 结构化输出"""
    print("\n" + "=" * 50)
    print("Part 2: JSON Output Parser")
    print("=" * 50 + "\n")

    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template=(
            "从以下文本中提取关键信息，返回JSON格式。\n"
            "{format_instructions}\n\n"
            "文本内容: {text}"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = make_ollama()
    chain = prompt | llm | parser

    text = "张三，28岁，软件工程师，月薪25000元，住在北京朝阳区。"
    print(f"输入: {text}\n")

    result = chain.invoke({"text": text})
    print("解析结果 (dict):")
    for key, value in result.items():
        print(f"  {key}: {value}")


# ========== Part 3: XML 输出解析 ==========

def demo_xml_parser():
    """演示 XMLOutputParser - 从 XML 标签中提取内容"""
    print("\n" + "=" * 50)
    print("Part 3: XML Output Parser")
    print("=" * 50 + "\n")

    parser = XMLOutputParser(tags=["name", "price", "features"])

    prompt = PromptTemplate(
        template=(
            "将以下产品信息整理为XML格式。\n"
            "{format_instructions}\n\n"
            "产品描述: {description}"
        ),
        input_variables=["description"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = make_ollama()
    chain = prompt | llm | parser

    desc = "MacBook Pro M4芯片，16GB内存，512GB固态硬盘，售价1999美元。支持雷雳5接口和ProMotion屏幕。"
    print(f"输入: {desc}\n")

    result = chain.invoke({"description": desc})
    print("解析结果:")
    print(result)


# ========== Part 4: 列表解析 ==========

def demo_list_parser():
    """演示逗号分隔列表解析器"""
    print("\n" + "=" * 50)
    print("Part 4: 列表输出解析")
    print("=" * 50 + "\n")

    from langchain_core.output_parsers import CommaSeparatedListOutputParser
    parser = CommaSeparatedListOutputParser()

    prompt = PromptTemplate(
        template="列出{topic}的前5个相关技术，用逗号分隔。\n{format_instructions}",
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = make_ollama()
    chain = prompt | llm | parser

    result = chain.invoke({"topic": "人工智能"})
    print("解析结果 (list):")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item}")


def main():
    demo_pydantic_parser()
    demo_json_parser()
    demo_xml_parser()
    demo_list_parser()


if __name__ == "__main__":
    main()
