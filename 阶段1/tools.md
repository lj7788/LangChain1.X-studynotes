# 工具模块 - make_model

本文档解释 `/Volumes/data/code/me/2026/03/longchat01/阶段1/tools.py` 中的代码。

---

## 完整代码

```python
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

from langchain_openai import ChatOpenAI

def make_model(model_name: str = "GLM-4.7-Flash"):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url="https://ai.gitee.io/v1",
        api_key="7LR8ZKEKGICENUPFMB4MQVDDUK5XNE4PCQ4GMG1C"
    )
```

---

## 代码逐行解析

### 第 1-2 行：导入并忽略警告
```python
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")
```
- 忽略 Pydantic V1 兼容性警告
- 这是一个常见的依赖兼容性问题

---

### 第 4 行：导入 ChatOpenAI
```python
from langchain_openai import ChatOpenAI
```
- LangChain 提供的 OpenAI 兼容聊天模型类

---

### 第 6-12 行：定义 make_model 函数
```python
def make_model(model_name: str = "GLM-4.7-Flash"):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        base_url="https://ai.gitee.io/v1",
        api_key="7LR8ZKEKGICENUPFMB4MQVDDUK5XNE4PCQ4GMG1C"
    )
```

---

## 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_name | str | "GLM-7-7-Flash" | 模型名称 |

---

## 返回值

返回一个 `ChatOpenAI` 实例，配置如下：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| model | GLM-4.7-Flash | 智谱 GLM 模型 |
| temperature | 0 | 生成确定性结果 |
| base_url | https://ai.gitee.io/v1 | Gitee AI API 端点 |
| api_key | 7LR8ZKEKGICENUPF... | API 密钥 |

---

## 使用示例

```python
from tools import make_model

# 使用默认模型
model = make_model()

# 指定模型
model = make_model("gpt-4")

# 在链中使用
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("{question}")
chain = prompt | make_model()

result = chain.invoke({"question": "你好"})
```

---

## 核心概念

### 为什么需要这个函数？
1. **统一配置**: 集中管理 API 配置
2. **简化代码**: 避免在每个文件中重复配置
3. **易于修改**: 只需修改一处即可更改模型

### Gitee AI
- 国内可用的 OpenAI 兼容 API
- 支持多种开源大模型
- 访问地址: https://ai.gitee.com

### 注意事项
- API 密钥已硬编码，存在安全风险
- 生产环境应使用环境变量
- 建议改为:
  ```python
  import os
  api_key = os.environ.get("GITEE_API_KEY")
  ```
