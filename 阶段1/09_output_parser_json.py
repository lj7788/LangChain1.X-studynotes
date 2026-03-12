from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

parser = JsonOutputParser(pydantic_object=Person)

json_str = '{"name": "张三", "age": 25, "city": "北京"}'
result = parser.invoke(json_str)

print("解析结果:", result)
print("类型:", type(result))
print("姓名:", result["name"])
