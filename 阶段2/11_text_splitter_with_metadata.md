# 文档分割（带元数据）

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/11_text_splitter_with_metadata.py` 中的代码。

---

## 完整代码

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = """这是第一段内容。包含了一些重要的信息。

这是第二段内容。继续添加更多文本。

这是第三段内容。还有更多内容需要处理。

这是第四段内容。最后一段内容。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=10,
    add_start_index=True
)

docs = splitter.split_documents([
    Document(
        page_content=text,
        metadata={"source": "sample.txt", "author": "AI"}
    )
])

print("=== 文档分割（带元数据）===")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
```

---

## 代码解析

### split_documents 方法
- 直接处理 Document 对象列表
- `add_start_index`: 在元数据中添加起始位置索引
- 保留原始文档的元数据信息
- 适合处理已加载的文档
