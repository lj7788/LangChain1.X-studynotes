# 递归字符分割器

本文档解释 `/Volumes/data/code/me/2026/03/LangChain1.X-/阶段2/08_text_splitter_recursive.py` 中的代码。

---

## 完整代码

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """# 标题一
这是第一段内容。这里有很多文本需要处理。

## 标题二
这是第二段内容。包含了一些重要的信息。

### 标题三
这是第三段内容。还有更多内容需要分割。

第四段内容在这里。继续添加更多文本。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

docs = splitter.split_text(text)

print("=== 递归字符分割器 ===")
print(f"原始文本长度: {len(text)}")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"内容: {repr(doc)}")
```

---

## 代码解析

### RecursiveCharacterTextSplitter
- 递归使用多个分隔符进行分割
- `separators`: 按优先级排列的分隔符列表
- 智能处理不同类型的文本结构
- 推荐的通用文本分割器
