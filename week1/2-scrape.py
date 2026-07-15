import os
from platform import system_alias
from dotenv import load_dotenv
from openai import OpenAI
from scraper import fetch_site_content,get_site_links

load_dotenv(override=True)

url = "https://cnn.com"

content = fetch_site_content(url)

system_message = f"""
You are a helpful assistant who can summarize website content.
The contents may also have meaningless link text, so you need to ignore them.
You need to return the summary in a short paragraph. Respond with only the summary, no other text.
"""

user_message = f"""
Here is the website content:
```
{content}
```
Please summarize in a witty and engaging way.
"""

messages=[
    {"role": "system", "content": system_message},
    {"role": "user", "content": content}
]

client = OpenAI()

response = client.chat.completions.create(model="gpt-5-nano", messages=messages)

print(response.choices[0].message.content)


