import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

model = make_model()

chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model

def extract_text(input):
    return {"text": input.content}

chain2_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容: {text}"
)
chain2 = chain2_prompt | model 

overall_chain = chain1 | RunnableLambda(extract_text) | chain2 


result = overall_chain.invoke({"text": "LangChain 是一个用于构建 LLM 应用的框架"})
print("最终结果:", result)
