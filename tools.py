import warnings

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")
warnings.filterwarnings("ignore", message="Please see the migration guide")

from langchain_openai import ChatOpenAI, OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
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

def make_ollama(model_name: str = None):
    if model_name is None:
        model_name = "llama3.1:8b"
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
