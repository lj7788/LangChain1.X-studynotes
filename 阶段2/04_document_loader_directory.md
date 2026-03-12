# 目录加载

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/04_document_loader_directory.py` 中的代码。

---

## 完整代码

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

file1 = data_dir / "doc1.txt"
file1.write_text("这是第一个文档的内容。", encoding="utf-8")

file2 = data_dir / "doc2.txt"
file2.write_text("这是第二个文档的内容。", encoding="utf-8")

loader = DirectoryLoader(
    str(data_dir),
    glob="*.txt",
    loader_cls=TextLoader
)
documents = loader.load()

print("=== 目录加载 ===")
print(f"文档数量: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"\n--- 文档 {i+1} ---")
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata.get('source', 'N/A')}")
```

---

## 代码解析

### DirectoryLoader
- 用于加载目录中的所有文件
- `glob`: 指定要加载的文件模式
- `loader_cls`: 指定使用的加载器类
