from langchain_text_splitters import CharacterTextSplitter

text = """这是一个很长的文档。
我们可以按照字符数量进行分割。
LangChain 是智能体工程（agent engineering）的平台。Replit、Clay、Rippling、Cloudflare、Workday 等公司的 AI 团队信赖 LangChain 的产品来工程化可靠的智能体（reliable agents）。

我们开源的框架可帮助您构建智能体：

LangChain：帮助您快速开始使用任何您选择的模型提供商来构建智能体。
LangGraph：允许您通过低级别编排、内存和**人工干预支持（human-in-the-loop support）来控制自定义智能体的每一步。您可以使用持久化执行（durable execution）**来管理长时间运行的任务。
LangSmith 是一个平台，可帮助 AI 团队利用实时生产数据进行持续测试和改进。LangSmith 提供：

可观测性（Observability）：通过详细的跟踪和汇总趋势指标，精确查看您的智能体如何思考和行动。
评估（Evaluation）：在生产数据和离线数据集上测试和评分智能体行为，以实现持续改进。
部署（Deployment）：使用专为长时间运行任务而构建的可扩展基础设施（scalable infrastructure），一键发布您的智能体。
📢 注意： LangGraph Platform 现已更名为 LangSmith Deployment。欲了解更多信息，请查阅 Changelog（更新日志）。


这里有更多的内容。
继续添加更多文本。"""

splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separator="\n"
)

docs = splitter.split_text(text)

print("=== 字符分割器 ===")
print(f"原始文本长度: {len(text)}")
print(f"分割后文档数量: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- 块 {i+1} ---")
    print(f"长度: {len(doc)}")
    print(f"内容: {repr(doc)}")
