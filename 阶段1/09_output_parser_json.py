import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

parser = JsonOutputParser(pydantic_object=Person)
json_str = '{"name": "张三", "age": 25, "city": "北京","x":100}'
result = parser.invoke(json_str)

print("解析结果:", result)
print("类型:", type(result))
print("姓名:", result["name"])
print("x:", result.get("xx",0))

class Student(BaseModel):
    id:int

stParse=JsonOutputParser(pydantic_object=Student)

strJson='{"name": "张三", "age": 25, "city": "北京","x":100}'

res=stParse.invoke(strJson)

print(res.get("id",0))


from tools import make_model
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Person(BaseModel):
    name: str
    age: int
    city: str

parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "请描述一个人，名字叫张三，25岁，住在北京\n{format_instructions}"
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

llm = make_model()

chain = prompt | llm | parser

response = chain.invoke({})

print("结构化输出:", response)
print("类型:", type(response))
print("姓名:", response.get("name"))
print("年龄:", response.get("age"))
print("城市:", response.get("city"))

