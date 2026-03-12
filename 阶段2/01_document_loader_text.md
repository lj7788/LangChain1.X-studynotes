# 文本文件加载

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/01_document_loader_text.py` 中的代码。

---

## 完整代码

```python
from langchain_community.document_loaders import TextLoader
from pathlib import Path

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

sample_text = data_dir / "sample.txt"
sample_text.write_text("这是一个示例文本文件。\n这是第二行内容。\n这是第三行内容。")

loader = TextLoader(str(sample_text), encoding="utf-8")
documents = loader.load()

print("=== 文本文件加载 ===")
print(f"文档数量: {len(documents)}")
print(f"内容长度: {len(documents[0].page_content)}")
print(f"元数据: {documents[0].metadata}")
print(f"内容预览: {documents[0].page_content[:100]}")
```

---

## 代码解析

### TextLoader
- 用于加载文本文件
- 参数 `encoding="utf-8"` 指定编码格式
- 返回 `Document` 对象列表

### Document 对象
- `page_content`: 文档内容
- `metadata`: 元数据（如文件路径等）
