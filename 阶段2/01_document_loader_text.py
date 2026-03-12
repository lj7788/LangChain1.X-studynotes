from langchain_community.document_loaders import TextLoader
from pathlib import Path
import os

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
