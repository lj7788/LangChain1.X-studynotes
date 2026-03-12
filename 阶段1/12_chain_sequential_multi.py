import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
model = make_model()

chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model | output_parser

chain2_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成法语: {text}"
)
chain2 = chain2_prompt | model | output_parser

overall_chain = RunnableParallel(
    english_text=chain1,
    french_text=chain2
)

result = overall_chain.invoke({"text": "LangChain 是一个 LLM 应用框架"})
print("英文:", result["english_text"])
print("法语:", result["french_text"])
