import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI()

message="Hello llm, how are you?"

messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": message}
]

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)

print(response.choices[0].message.content)