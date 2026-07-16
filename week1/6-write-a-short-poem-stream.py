from openai import OpenAI
import openai
from scraper import fetch_site_content
from dotenv import load_dotenv

load_dotenv(override=True)

site_content=fetch_site_content("https://cnn.com")

system_message="""
You are a poet who writes short poems on any given topic
Respond with the poem alone
"""

user_message=f"""
Here are some contents from a website: {site_content}
Write a 5 line poem
"""

messages=[{"role": "system","content": system_message},{"role":"user","content":user_message}]

openai=OpenAI()

stream=openai.chat.completions.create(model="gpt-5-nano",messages=messages, stream=True)

for chunk in stream:
  print(chunk.choices[0].delta.content or '')


