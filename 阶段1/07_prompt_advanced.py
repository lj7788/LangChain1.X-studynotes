from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_prompt = SystemMessagePromptTemplate.from_template(
    "你是一个 {role}，用 {tone} 的语气回答问题"
)
human_prompt = HumanMessagePromptTemplate.from_template("{question}")

prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

result = prompt.invoke({
    "role":"技术专家",
    "tone":"专业",
    "question":"什么是 LangChain?"
})

print("格式化后的提示词:")
for msg in result.messages:
    print(f"- {msg.type}: {msg.content}")
