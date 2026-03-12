from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = parser.invoke("你好，这是一段文本回复")
print("解析结果:", result)
print("类型:", type(result))
