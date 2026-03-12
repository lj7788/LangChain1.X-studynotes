# CSV 文件加载

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/02_document_loader_csv.py` 中的代码。

---

## 完整代码

```python
from langchain_community.document_loaders import CSVLoader
from pathlib import Path

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

sample_csv = data_dir / "sample.csv"
sample_csv.write_text("""姓名,年龄,城市
张三,25,北京
李四,30,上海
王五,28,深圳""", encoding="utf-8")

loader = CSVLoader(file_path=str(sample_csv), encoding="utf-8")
documents = loader.load()

print("=== CSV 文件加载 ===")
print(f"文档数量: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"\n--- 文档 {i+1} ---")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
```

---

## 代码解析

### CSVLoader
- 用于加载 CSV 文件
- 每一行会被加载为一个 Document
- 元数据包含行号信息
