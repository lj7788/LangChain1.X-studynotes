from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from pathlib import Path
import re

txt_file = Path(__file__).parent / "data" / "sg.txt"

loader = TextLoader(str(txt_file), encoding="utf-8")
documents = loader.load()
text = documents[0].page_content

print(f"原始内容长度: {len(text)}")

# 使用正则按章节分割
pattern = r"(第[一二三四五六七八九十百千]+回)"
parts = re.split(pattern, text)

chapters = []
current_chapter = ""

for i, part in enumerate(parts):
    if re.match(pattern, part):
        # 这是一个章节标题
        if current_chapter:
            chapters.append(current_chapter)
        current_chapter = part
    else:
        current_chapter += part

# 最后一个章节
if current_chapter:
    chapters.append(current_chapter)

print(f"分割后章节数量: {len(chapters)}")

# 创建 Document 列表
docs = [Document(page_content=ch, metadata={"chapter": i+1}) for i, ch in enumerate(chapters)]

print("\n章节列表:")
for i, doc in enumerate(docs):
    content = doc.page_content
    match = re.search(r"(第[一二三四五六七八九十百千]+回[^。\n]*)", content)
    title = match.group(1) if match else f"无标题"
    print(f"  {title} ({len(content)}字符)")
