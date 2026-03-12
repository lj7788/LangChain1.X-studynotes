
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("请用 {language} 介绍 {topic}")

formatted_prompt = prompt.format(language="中文", topic="LangChain")
print("格式化后的提示词:")
print(formatted_prompt)
