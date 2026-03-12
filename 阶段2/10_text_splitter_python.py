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
    print(f"内容:\n{doc}")
