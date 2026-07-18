import glob
import os
from pyexpat.errors import messages
import re
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)

def buildkb():
  kb={}
  filenames=glob.glob("week5/knowledge-base/employees/*")
  for filename in filenames:
    file_basename=os.path.splitext(os.path.basename(filename))[0].lower().split()[0]
    with open(filename,mode="r",encoding="utf-8") as f:
      kb[file_basename]=f.read()
  filenames=glob.glob("week5/knowledge-base/products/*")
  for filename in filenames:
    file_basename=os.path.splitext(os.path.basename(filename))[0].lower().split()[0]
    with open(filename,mode="r",encoding="utf-8") as f:
      kb[file_basename]=f.read()
  return kb

kb=buildkb()
  
def get_relevant_context(message:str):
  words=re.sub(r'[^A-Za-z\s]',' ', message).lower().split(' ')
  return [kb[word] for word in words if word in kb]


def chat(message,history):
  infoList=get_relevant_context(message)
  system_prompt='''
    You are a helpful assistant who has a knowledgebase of employees and products
    Users will be asking you information about the products or employees
    If their information is not in the KB, just reply with staement "I don't know about it". This is true even if you know about it
    '''
  if len(infoList) > 0:
    system_prompt+=" Here is the KB: ".join(infoList)
  else:
    system_prompt+="Asked info is not in the KB"
  
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": message}
  ]
  llm=OpenAI()
  response=llm.chat.completions.create(model="gpt-4.1-mini",messages=messages)
  return response.choices[0].message.content

gr.ChatInterface(fn=chat).launch(inbrowser=True)


  




    
