import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

gemini_api_key=os.getenv("GEMINI_API_KEY")

system_msg="You are a helpful assistant"

def chat():
    llm=OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_api_key)
    response=llm.chat.completions.create( model="gemini-3.5-flash", messages=[{"role":"system","content": system_msg}, {"role": "user","content":"what is the date today?"}])
    print("Gemini says:" + response.choices[0].message.content)

chat()