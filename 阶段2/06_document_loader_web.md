# 网页加载

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/06_document_loader_web.py` 中的代码。

---

## 完整代码

```python
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
```

---

## 代码解析

### WebBaseLoader
- 用于加载网页内容
- 自动提取网页的文本内容
- 元数据包含页面标题和 URL 信息
