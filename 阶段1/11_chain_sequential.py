from langchain.chains import SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.com/v1",
    api_key="your-gitee-ai-api-key"
)

chain1_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文: {text}"
)
chain1 = chain1_prompt | model

chain2_prompt = ChatPromptTemplate.from_template(
    "用一句话总结以下内容: {text}"
)
chain2 = chain2_prompt | model

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

result = overall_chain.invoke("LangChain 是一个用于构建 LLM 应用的框架")
print("最终结果:", result)
