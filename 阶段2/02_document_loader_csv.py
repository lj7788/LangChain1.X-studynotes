from langchain_community.document_loaders import CSVLoader
from pathlib import Path
from pydantic import BaseModel

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

sample_csv = data_dir / "sample.csv"
sample_csv.write_text("""姓名,年龄,城市
张三,25,北京
李四,30,上海
王五,28,深圳""", encoding="utf-8")

tbHeader={
    "姓名": "name",
    "年龄": "age",
    "城市": "city"
}

loader = CSVLoader(file_path=str(sample_csv), encoding="utf-8")
documents = loader.load()

print("=== CSV 文件加载 ===")
print(f"文档数量: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"\n--- 文档 {i+1} ---")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")

class Person(BaseModel):
    name: str
    age: int
    city: str

print("\n转换对象:")
persons=[]
for doc in documents:
    content = doc.page_content.strip().split("\n")
    pdata={}
    for part in content:
        if ":" in part:
            key, value = part.split(":", 1)
            # key 是中文，需要转换为英文
            key = tbHeader.get(key.strip(), key.strip())
            pdata[key.strip()] = value.strip()
    if pdata:
        try:
            person = Person(**pdata)
            persons.append(person)
        except ValueError:
            print(f"跳过无效数据: {pdata}")

print(f"转换后的对象数量: {len(persons)}")
for person in persons:
    print(person)
