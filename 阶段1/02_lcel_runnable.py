from tools import make_model
from langchain_core.runnables import RunnableLambda

def transform_func(data):
    return {"topic": data["topic"].upper()}

chain = RunnableLambda(transform_func)

result = chain.invoke({"topic": "hello"})
print(result)
