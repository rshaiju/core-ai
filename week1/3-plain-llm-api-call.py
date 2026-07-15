import os
from dotenv import load_dotenv
import requests


load_dotenv(override=True)

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={"model": "gpt-5-nano", "messages": [{"role": "user", "content": "Hello, how are you?"}]}
)

print(response.json()["choices"][0]["message"]["content"])