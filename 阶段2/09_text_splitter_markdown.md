# Markdown 分割器

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/09_text_splitter_markdown.py` 中的代码。

---

## 完整代码

```python
from langchain_text_splitters import MarkdownTextSplitter

markdown_text = """# 第一章

这是第一章的内容。

## 第一节

这是第一节的内容。

# 第二章

这是第二章的内容。

## 第二节

这是第二节的内容。"""

splitter = MarkdownTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = splitter.split_text(markdown_text)

print("=== Markdown 分割器 ===")
print(f"原始文本长度: {len(markdown_text)}")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"内容: {doc.page_content}")
```

---

## 代码解析

### MarkdownTextSplitter
- 专门用于分割 Markdown 文档
- 按 Markdown 标题结构进行分割
- 保留标题信息在元数据中
- 适合处理 Markdown 格式的文档
