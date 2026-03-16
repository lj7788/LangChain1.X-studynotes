import time
import requests
import json

print("=" * 60)
print("Ollama (llama3.2) 性能测试")
print("=" * 60)

base_url = "http://localhost:11434"
model = "llama3.2"

# 测试1: 简单补全 (使用原生 API)
print("\n【测试1】简单补全 (100 tokens)")
start = time.time()
response = requests.post(
    f"{base_url}/api/generate",
    json={
        "model": model,
        "prompt": "The meaning of life is",
        "num_predict": 100,
        "temperature": 0.7,
        "stream": False
    }
)
elapsed = time.time() - start
if response.status_code == 200:
    result = response.json()
    tokens = result.get("eval_count", 0)
    print(f"  响应时间: {elapsed:.2f}s")
    print(f"  生成 tokens: {tokens}")
    print(f"  速度: {tokens/elapsed:.2f} tokens/s")
    print(f"  输出: {result.get('response', '')[:100]}...")
else:
    print(f"  错误: {response.status_code} - {response.text}")

# 测试2: 中等长度补全
print("\n【测试2】中等补全 (200 tokens)")
start = time.time()
response = requests.post(
    f"{base_url}/api/generate",
    json={
        "model": model,
        "prompt": "Write a short story about a robot learning to paint:",
        "num_predict": 200,
        "temperature": 0.7,
        "stream": False
    }
)
elapsed = time.time() - start
if response.status_code == 200:
    result = response.json()
    tokens = result.get("eval_count", 0)
    print(f"  响应时间: {elapsed:.2f}s")
    print(f"  生成 tokens: {tokens}")
    print(f"  速度: {tokens/elapsed:.2f} tokens/s")
else:
    print(f"  错误: {response.status_code}")

# 测试3: 流式输出
print("\n【测试3】流式输出测试")
start = time.time()
response = requests.post(
    f"{base_url}/api/generate",
    json={
        "model": model,
        "prompt": "Count from 1 to 20:",
        "num_predict": 100,
        "temperature": 0.1,
        "stream": True
    },
    stream=True
)
first_token_time = None
token_count = 0
for line in response.iter_lines():
    if line:
        try:
            chunk = json.loads(line.decode('utf-8'))
            if chunk.get("response"):
                if first_token_time is None:
                    first_token_time = time.time() - start
                token_count += 1
            if chunk.get("done"):
                break
        except:
            pass
elapsed = time.time() - start
print(f"  首token延迟: {first_token_time:.3f}s" if first_token_time else "  首token延迟: N/A")
print(f"  总时间: {elapsed:.2f}s")
print(f"  生成 tokens: {token_count}")
print(f"  速度: {token_count/elapsed:.2f} tokens/s" if token_count > 0 else "  速度: N/A")

# 测试4: Chat API
print("\n【测试4】Chat API 测试")
start = time.time()
response = requests.post(
    f"{base_url}/api/chat",
    json={
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ],
        "stream": False
    }
)
elapsed = time.time() - start
if response.status_code == 200:
    result = response.json()
    tokens = result.get("eval_count", 0)
    content = result.get("message", {}).get("content", "")
    print(f"  响应时间: {elapsed:.2f}s")
    print(f"  生成 tokens: {tokens}")
    print(f"  速度: {tokens/elapsed:.2f} tokens/s" if tokens > 0 else "  速度: N/A")
    print(f"  输出: {content[:100]}")
else:
    print(f"  错误: {response.status_code}")

print("\n" + "=" * 60)
print("Ollama 测试完成!")
print("=" * 60)
