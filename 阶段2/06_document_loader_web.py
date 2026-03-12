from langchain_community.document_loaders import WebBaseLoader
import warnings
warnings.filterwarnings("ignore")

url = "https://www.example.com"

print("=== 网页加载 ===")
print(f"加载URL: {url}")

loader = WebBaseLoader(url)
documents = loader.load()

print(f"文档数量: {len(documents)}")
print(f"内容长度: {len(documents[0].page_content)}")
print(f"元数据: {documents[0].metadata}")
print(f"内容预览: {documents[0].page_content[:200]}...")
