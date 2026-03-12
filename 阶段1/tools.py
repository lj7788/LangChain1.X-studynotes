import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

from langchain_openai import ChatOpenAI, OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载上级目录的 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
api_model = os.getenv("OPENAI_API_MODEL", "GLM-4.7-Flash")

def make_model(model_name: str = None):
    if model_name is None:
        model_name = api_model
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url=api_base,
        api_key=api_key
    )

def make_openai(model_name: str = None):
    if model_name is None:
        model_name = api_model
    return OpenAI(
        model=model_name,
        temperature=0,
        base_url=api_base,
        api_key=api_key
    )
