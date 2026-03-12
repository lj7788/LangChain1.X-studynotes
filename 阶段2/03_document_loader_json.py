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
