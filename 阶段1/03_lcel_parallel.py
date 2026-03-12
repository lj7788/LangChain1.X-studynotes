import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

prompt1 = ChatPromptTemplate.from_template("用中文介绍 {topic}")
prompt2 = ChatPromptTemplate.from_template("用英文介绍 {topic}")

model = make_model()

parser = StrOutputParser()

chain = RunnableParallel(
    chinese=prompt1 | model | parser,
    english=prompt2 | model | parser
)

result = chain.invoke({"topic": "LangChain"})
print("中文:", result["chinese"])
print("英文:", result["english"])
