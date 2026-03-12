from langchain_openai import OpenAI

model = OpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    base_url="https://ai.gitee.com/v1",
    api_key="your-gitee-ai-api-key"
)

response = model.invoke("请用一句话介绍 LangChain")
print("模型回复:", response)
