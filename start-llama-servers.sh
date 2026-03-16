#!/bin/bash

LLAMA_SERVER="/Volumes/data/soft/llama.cpp/llama-server"
OLLAMA_BLOBS="$HOME/.ollama/models/blobs"

EMBEDDING_MODEL="$OLLAMA_BLOBS/sha256-6069819943bfc69b76cd36ef4a7a5f7fb1c7de0e2baa7df1a23b48ecfa18fd97"
CHAT_MODEL="$OLLAMA_BLOBS/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff"

CHAT_PORT=11434
EMBEDDING_PORT=11435

EMBEDDING_PID=""
CHAT_PID=""

cleanup() {
    echo ""
    echo "正在停止服务..."
    [ -n "$EMBEDDING_PID" ] && kill $EMBEDDING_PID 2>/dev/null
    [ -n "$CHAT_PID" ] && kill $CHAT_PID 2>/dev/null
    echo "服务已停止"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "========================================"
echo "  llama.cpp 多模型服务启动器"
echo "  MacBook Pro M1 Pro (32GB)"
echo "  兼容 Ollama API (端口 11434)"
echo "========================================"
echo ""

if [ ! -f "$EMBEDDING_MODEL" ]; then
    echo "错误: 找不到 embedding 模型文件"
    echo "路径: $EMBEDDING_MODEL"
    exit 1
fi

if [ ! -f "$CHAT_MODEL" ]; then
    echo "错误: 找不到 chat 模型文件"
    echo "路径: $CHAT_MODEL"
    exit 1
fi

if lsof -i :$CHAT_PORT > /dev/null 2>&1; then
    echo "警告: 端口 $CHAT_PORT 已被占用，可能 ollama 正在运行"
    echo "请先停止 ollama: pkill ollama 或 ollama stop"
    echo ""
    read -p "是否尝试自动停止 ollama? (y/n): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        pkill ollama 2>/dev/null
        sleep 2
    else
        exit 1
    fi
fi

echo "[1/2] 启动 Embedding 服务 (Dmeta-embedding-zh:F16)..."
echo "      端口: $EMBEDDING_PORT"
echo "      模型大小: ~197MB"

$LLAMA_SERVER \
    -m "$EMBEDDING_MODEL" \
    --port $EMBEDDING_PORT \
    --host 127.0.0.1 \
    --embeddings \
    --pooling mean \
    -c 512 \
    -ngl 99 \
    --batch-size 512 \
    --ubatch-size 512 \
    --threads 4 \
    --threads-batch 4 \
    --log-disable \
    2>/dev/null &

EMBEDDING_PID=$!
sleep 2

echo ""
echo "[2/2] 启动 Chat 服务 (llama3.2) - 极致优化模式..."
echo "      端口: $CHAT_PORT (Ollama 兼容)"
echo "      模型大小: ~1.9GB"
echo "      优化: Flash Attention + KV缓存量化 + 连续批处理"

$LLAMA_SERVER \
    -m "$CHAT_MODEL" \
    --port $CHAT_PORT \
    --host 0.0.0.0 \
    -c 8192 \
    -ngl 99 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --threads 8 \
    --threads-batch 8 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --mlock \
    --parallel 8 \
    --cont-batching \
    --cache-reuse 256 \
    --cache-ram 4096 \
    --temp 0.7 \
    --top-k 40 \
    --top-p 0.9 \
    --repeat-penalty 1.1 \
    --metrics \
    2>&1 &

CHAT_PID=$!
sleep 3

echo ""
echo "========================================"
echo "  服务已启动! (Ollama 兼容模式)"
echo "========================================"
echo ""
echo "📡 API 端点 (与 Ollama 完全兼容):"
echo ""
echo "  Chat API (llama3.2):"
echo "    http://localhost:$CHAT_PORT/v1/chat/completions"
echo "    http://localhost:$CHAT_PORT/v1/completions"
echo "    http://localhost:$CHAT_PORT/api/chat"
echo "    http://localhost:$CHAT_PORT/api/generate"
echo ""
echo "  Embedding API (Dmeta-embedding-zh):"
echo "    http://localhost:$EMBEDDING_PORT/v1/embeddings"
echo ""
echo "📊 监控面板:"
echo "    http://localhost:$CHAT_PORT"
echo ""
echo "========================================"
echo "  LangChain 使用示例 (Ollama 兼容)"
echo "========================================"

cat << 'PYTHON_EXAMPLE'

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Chat 模型 - 使用 Ollama 默认端口
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3.2"
)

# Embedding 模型
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:11435/v1",
    api_key="ollama",
    model="Dmeta-embedding-zh"
)

# 使用示例
response = llm.invoke("你好!")
print(response.content)

# 获取文本嵌入
vec = embeddings.embed_query("这是一段测试文本")
print(f"向量维度: {len(vec)}")

PYTHON_EXAMPLE

echo ""
echo "========================================"
echo "  OpenAI Python SDK 示例"
echo "========================================"

cat << 'OPENAI_EXAMPLE'

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "你好!"}]
)
print(response.choices[0].message.content)

OPENAI_EXAMPLE

echo ""
echo "按 Ctrl+C 停止所有服务"
echo ""

wait $CHAT_PID
