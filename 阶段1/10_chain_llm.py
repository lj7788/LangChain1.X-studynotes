import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_model
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{question}")

model = make_model()

chain = prompt | model

result = chain.invoke({"question": "LangChain 是什么?"})
print("回答:", result.content)
