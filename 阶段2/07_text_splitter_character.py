from langchain.text_splitter import CharacterTextSplitter

text = """这是一个很长的文档。
我们可以按照字符数量进行分割。

这里有更多的内容。
继续添加更多文本。"""

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separator="\n"
)

docs = splitter.split_text(text)

print("=== 字符分割器 ===")
print(f"原始文本长度: {len(text)}")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"长度: {len(doc)}")
    print(f"内容: {repr(doc)}")
