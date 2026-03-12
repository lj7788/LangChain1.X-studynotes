from langchain.chains import TransformChain
from langchain_core.runnables import RunnableLambda

def transform_func(inputs):
    text = inputs["text"]
    words = text.split()
    return {"word_count": len(words), "original_text": text}

chain = TransformChain(
    input_variables=["text"],
    output_variables=["word_count", "original_text"],
    transform=transform_func
)

result = chain.invoke({"text": "LangChain 是一个强大的 LLM 应用框架"})
print("单词数:", result["word_count"])
print("原文:", result["original_text"])
