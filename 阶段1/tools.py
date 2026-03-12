import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

from langchain_openai import ChatOpenAI

def make_model(model_name: str = "GLM-4.7-Flash"):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url="https://ai.gitee.com/v1",
        api_key="7LR8ZKEKGICENUPFMB4MQVDDUK5XNE4PCQ4GMG1C"
    )
