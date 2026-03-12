from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

pdf_path = Path(__file__).parent / "data" / "sample.pdf"

print("=== PDF 文件加载 ===")
print(f"PDF 文件存在: {pdf_path.exists()}")

if pdf_path.exists():
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"文档数量: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"\n--- 第 {i+1} 页 ---")
        print(f"内容预览: {doc.page_content[:200]}...")
else:
    print("请将 PDF 文件放入 data 目录，或使用其他加载器")
