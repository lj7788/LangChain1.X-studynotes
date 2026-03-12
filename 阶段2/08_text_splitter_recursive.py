from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """# 标题一
这是第一段内容。这里有很多文本需要处理。

## 标题二
这是第二段内容。包含了一些重要的信息。

### 标题三
这是第三段内容。还有更多内容需要分割。

第四段内容在这里。继续添加更多文本。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

docs = splitter.split_text(text)

print("=== 递归字符分割器 ===")
print(f"原始文本长度: {len(text)}")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"内容: {repr(doc)}")
