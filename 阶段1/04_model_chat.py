import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import make_model
from langchain_core.messages import HumanMessage, SystemMessage

model = make_model()

messages = [
    SystemMessage(content="你是一个超人助手"),
    HumanMessage(content="你好，请介绍一下自己")
]

response = model.invoke(messages)
print("模型回复:", response.content)
print("完整响应:", response)
