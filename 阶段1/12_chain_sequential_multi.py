from langchain.chains import SequentialChain
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
    "将以下内容翻译成法语: {text}"
)
chain2 = chain2_prompt | model

overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["text"],
    output_variables=["english_text", "french_text"],
    verbose=True
)

result = overall_chain.invoke({"text": "LangChain 是一个 LLM 应用框架"})
print("英文:", result["english_text"])
print("法语:", result["french_text"])
