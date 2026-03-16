import time
import requests
import json

base_url = "http://localhost:11434/v1"

print("=" * 50)
print("llama.cpp 性能测试")
print("=" * 50)

# 测试1: 简单补全
print("\n【测试1】简单补全 (100 tokens)")
start = time.time()
response = requests.post(
    f"{base_url}/completions",
    json={
        "model": "local",
        "prompt": "The meaning of life is",
        "max_tokens": 100,
        "temperature": 0.7
    },
    stream=False
)
elapsed = time.time() - start
if response.status_code == 200:
    result = response.json()
    text = result.get("choices", [{}])[0].get("text", "")
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    print(f"  响应时间: {elapsed:.2f}s")
    print(f"  生成 tokens: {tokens}")
    print(f"  速度: {tokens/elapsed:.2f} tokens/s")
    print(f"  输出: {text[:100]}...")
else:
    print(f"  错误: {response.status_code}")

# 测试2: 中等长度补全
print("\n【测试2】中等补全 (200 tokens)")
start = time.time()
response = requests.post(
    f"{base_url}/completions",
    json={
        "model": "local",
        "prompt": "Write a short story about a robot learning to paint:",
        "max_tokens": 200,
        "temperature": 0.7
    },
    stream=False
)
elapsed = time.time() - start
if response.status_code == 200:
    result = response.json()
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    print(f"  响应时间: {elapsed:.2f}s")
    print(f"  生成 tokens: {tokens}")
    print(f"  速度: {tokens/elapsed:.2f} tokens/s")
else:
    print(f"  错误: {response.status_code}")

# 测试3: 流式输出速度
print("\n【测试3】流式输出测试")
start = time.time()
response = requests.post(
    f"{base_url}/completions",
    json={
        "model": "local",
        "prompt": "Count from 1 to 20:",
        "max_tokens": 100,
        "temperature": 0.1
    },
    stream=True
)
first_token_time = None
token_count = 0
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices", [{}])[0].get("text"):
                        if first_token_time is None:
                            first_token_time = time.time() - start
                        token_count += 1
                except:
                    pass
elapsed = time.time() - start
print(f"  首token延迟: {first_token_time:.3f}s" if first_token_time else "  首token延迟: N/A")
print(f"  总时间: {elapsed:.2f}s")
print(f"  生成 tokens: {token_count}")
print(f"  速度: {token_count/elapsed:.2f} tokens/s")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
