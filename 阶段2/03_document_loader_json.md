# JSON 文件加载

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/03_document_loader_json.py` 中的代码。

---

## 完整代码

```python
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
import json

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

sample_json = data_dir / "sample.json"
sample_json.write_text("""{
    "name": "LangChain",
    "version": "1.2.0",
    "features": ["LCEL", "Chains", "Agents"]
}""", encoding="utf-8")

loader = JSONLoader(
    file_path=str(sample_json),
    jq_schema=".",
    text_content=False
)
documents = loader.load()

print("=== JSON 文件加载 ===")
print(f"文档数量: {len(documents)}")
print(f"内容: {documents[0].page_content}")
print(f"元数据: {documents[0].metadata}")
```

---

## 代码解析

### JSONLoader
- 用于加载 JSON 文件
- `jq_schema`: 指定如何提取内容，`.` 表示提取整个 JSON 对象
- `text_content`: 设为 False 时返回原始 JSON 字符串
