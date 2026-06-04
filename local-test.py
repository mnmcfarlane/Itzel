import requests
import time

start = time.time()

response = requests.post(
    "http://192.168.1.236:11434/api/generate",
    json={
        "model": "llama3.2:1b",
        "prompt": "Reply with one word: hello",
        "stream": False,
        "options": {
            "num_predict": 5
        }
    },
    timeout=60
)

end = time.time()

print(response.status_code)
print(response.json().get("response"))
print(f"Time: {end - start:.2f} seconds")