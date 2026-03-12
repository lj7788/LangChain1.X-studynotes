import warnings
warnings.filterwarnings("ignore")

from langchain_ollama import OllamaEmbeddings

print("=== Ollama 本地 Embeddings ===")
print("模型: nomic-embed-text")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

texts = [
    "今天天气很好",
    "天气晴朗，适合外出",
    "我喜欢吃苹果",
    "苹果是一种水果"
]

results = embeddings.embed_documents(texts)
for i, (text, embedding) in enumerate(zip(texts, results)):
    print(f"\n文本 {i+1}: {text}")
    print(f"向量长度: {len(embedding)}")
    print(f"向量前5位: {embedding[:5]}")

query = "今天适合出去玩吗"
query_embedding = embeddings.embed_query(query)
print(f"\n查询: {query}")
print(f"查询向量长度: {len(query_embedding)}")
