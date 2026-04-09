import warnings

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")
warnings.filterwarnings("ignore", message="Please see the migration guide")

from langchain_openai import ChatOpenAI, OpenAI
from openai import OpenAI as OpenAIClient
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
        model_name = "gemma4:e4b-it-q4_K_M_opt"
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

class LlamaCppEmbeddings:
    def __init__(self, base_url="http://localhost:11434/v1", model="bge-m3:latest"):
        self.client = OpenAIClient(
            base_url=base_url,
            api_key="ollama"
        )
        self.model = model
    
    def embed_documents(self, texts):
        all_embeddings = []
        batch_size = 5
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def embed_query(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
    
    def __call__(self, text):
        return self.embed_query(text)

def make_embedding(base_url="http://localhost:11434/v1", model="bge-m3:latest"):
    return LlamaCppEmbeddings(base_url=base_url, model=model)
