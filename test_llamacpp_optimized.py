import time
import requests
import json

def test_llamacpp():
    print("=" * 60)
    print("llama.cpp 优化版 性能测试")
    print("=" * 60)
    
    base_url = "http://localhost:12434/v1"
    
    results = []
    
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
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        speed = tokens/elapsed if elapsed > 0 else 0
        results.append(("简单补全", speed))
        print(f"  响应时间: {elapsed:.2f}s")
        print(f"  生成 tokens: {tokens}")
        print(f"  速度: {speed:.2f} tokens/s")
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
        speed = tokens/elapsed if elapsed > 0 else 0
        results.append(("中等补全", speed))
        print(f"  响应时间: {elapsed:.2f}s")
        print(f"  生成 tokens: {tokens}")
        print(f"  速度: {speed:.2f} tokens/s")
    else:
        print(f"  错误: {response.status_code}")
    
    # 测试3: 流式输出
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
    speed = token_count/elapsed if elapsed > 0 and token_count > 0 else 0
    results.append(("流式输出", speed))
    print(f"  首token延迟: {first_token_time:.3f}s" if first_token_time else "  首token延迟: N/A")
    print(f"  总时间: {elapsed:.2f}s")
    print(f"  生成 tokens: {token_count}")
    print(f"  速度: {speed:.2f} tokens/s")
    
    # 测试4: Chat API
    print("\n【测试4】Chat API 测试")
    start = time.time()
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 50,
            "temperature": 0.1
        },
        stream=False
    )
    elapsed = time.time() - start
    if response.status_code == 200:
        result = response.json()
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        speed = tokens/elapsed if elapsed > 0 else 0
        results.append(("Chat API", speed))
        print(f"  响应时间: {elapsed:.2f}s")
        print(f"  生成 tokens: {tokens}")
        print(f"  速度: {speed:.2f} tokens/s")
    else:
        print(f"  错误: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    test_llamacpp()
