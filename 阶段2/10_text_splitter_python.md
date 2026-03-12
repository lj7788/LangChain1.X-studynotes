# Python 代码分割器

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/10_text_splitter_python.py` 中的代码。

---

## 完整代码

```python
from langchain_text_splitters import PythonCodeTextSplitter

python_code = """
def hello_world():
    print("Hello, World!")

class MyClass:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"

def calculate_sum(a, b):
    return a + b

# 这是一个注释
if __name__ == "__main__":
    obj = MyClass("LangChain")
    print(obj.greet())
"""

splitter = PythonCodeTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

docs = splitter.split_text(python_code)

print("=== Python 代码分割器 ===")
print(f"原始代码长度: {len(python_code)}")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"内容:\n{doc.page_content}")
```

---

## 代码解析

### PythonCodeTextSplitter
- 专门用于分割 Python 代码
- 按函数、类等代码结构分割
- 保留代码的完整性
- 适合处理代码文档
