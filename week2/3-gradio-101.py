from cgitb import text
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import os

load_dotenv(override=True)
gemini_api_key=os.getenv("GEMINI_API_KEY")

def chat(msg,model):
  system_prompt="You are a helpful assistant. You keep replies short and sweet"
  llm=OpenAI()
  if (model=="gemini-3.5-flash"):
    llm.api_key=gemini_api_key
    llm.base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
  messages=[{
    "role":"system",
    "content": system_prompt
  },{
    "role": "user",
    "content": msg
  }
  ]
  response=llm.chat.completions.create(model=model,messages=messages)
  return response.choices[0].message.content

msg_input=gr.Textbox(label="Your message")
model_input=gr.Dropdown(["gpt-5-mini","gpt-4","gemini-3.5-flash"],label="Model", value="gpt-5-mini")
msg_output=gr.Textbox(label="Response")

gr.Interface(fn=chat,title="chat",inputs=[msg_input,model_input], outputs=[msg_output], 
  examples=[
    ["What is the date today?","gpt-5-mini"],
    ["Who is the Indian PM?","gpt-4"],
    ["Which is the biggest city in India?","gemini-3.5-flash"]
  ],
  flagging_mode="never").launch(inbrowser=True)



