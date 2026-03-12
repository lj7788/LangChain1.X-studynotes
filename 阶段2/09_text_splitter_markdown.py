from langchain.text_splitter import MarkdownTextSplitter

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
