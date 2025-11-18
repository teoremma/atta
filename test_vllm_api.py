import requests

API_ENDPOINT = "http://ec2-3-90-255-225.compute-1.amazonaws.com:8000"
API_KEY = "hilde"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"

def get_headers():
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

def test_vllm_models():
    headers = get_headers()
    response = requests.get(f"{API_ENDPOINT}/v1/models", headers=headers)

    if response.status_code == 200:
        data = response.json()
        print("=== vLLM Available Models ===")
        for model in data["data"]:
            print(model["id"])
    else:
        print(f"Error {response.status_code}: {response.text}")

def test_vllm():
    headers = get_headers()

#     prompt = """
# ```python
# def longest_common_subsequence(s, t):
#     m = len(s)
#     n = len(t)
# """

    prompt_file = "prompts/dijkstra.txt"
    with open(prompt_file, "r") as f:
        prompt = f.read()

    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 1,
        "stop": ["```", "\n"],
        "n": 10,
    }

    response = requests.post(f"{API_ENDPOINT}/v1/completions", headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        for i, choice in enumerate(data["choices"]):
            if i >= 10:
                break
            print(f"=== vLLM API Response {i + 1} ===")
            print("```")
            print(choice["text"])
            print("```")
            print()
            print()
        # print("=== vLLM API Response ===")
        # print(data["choices"][0]["text"])
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_vllm()
    # test_vllm_models()
