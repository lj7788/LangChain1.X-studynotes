import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("用一句话介绍 {topic}")

model = make_ollama()

output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke({"topic": "智能体"})

print(result)
