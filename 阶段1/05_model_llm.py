from tools import make_model
from langchain_core.messages import HumanMessage

model = make_model()

response = model.invoke([HumanMessage(content="请用一句话介绍 LangChain")])
print("模型回复:", response.content)
