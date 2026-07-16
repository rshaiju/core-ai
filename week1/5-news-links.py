from pyexpat import model
import openai
from urllib3 import response
from scraper import fetch_site_content, get_site_links
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

openai=OpenAI()

system_prompt="""
You decipher the probable purpose of the given urls
Respond with json like below
[{
  "url" : "https://google.com",
  "description": "A popular search engine"
}]
"""

links=get_site_links("https://cnn.com")

user_prompt=f'''
Here is the list of urls
{links[:10]}
Respond just with the json
'''

messages=[{
    "role": "system",
    "content": system_prompt
},
{"role": "user",
"content": user_prompt
}]

print(links[:10])

response=openai.chat.completions.create(model="gpt-5-nano",messages=messages, response_format={"type": "json_object"})

print(response.choices[0].message.content)



