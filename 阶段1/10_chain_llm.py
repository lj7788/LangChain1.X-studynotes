from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{question}")

model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.com/v1",
    api_key="your-gitee-ai-api-key"
)

chain = LLMChain(llm=model, prompt=prompt)

result = chain.invoke({"question": "LangChain 是什么?"})
print("回答:", result["text"])
