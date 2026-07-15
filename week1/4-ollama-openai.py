from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

system_message=f'''You are a story teller. 
Tell a short story any topic you are given
Reply with the story alone'''

user_message=f'''
Tell a story about Rabbit
'''

OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"

openai=OpenAI(base_url=OLLAMA_BASE_URL,api_key="foo")

messages=[
    {"role":"system","content":system_message},
    {"role":"user", "content": user_message}
]

response= openai.chat.completions.create(model="gemma3:270m", messages=messages)

print(response.choices[0].message.content)
