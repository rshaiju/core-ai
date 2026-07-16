import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

msg_counter=0

def chat(message,history):
    global msg_counter
    msg_counter+=1
    print(f'messages:{msg_counter}')
    messages=[{"role": h["role"],"content":h["content"] }for h in history]
    messages.append({"role": "user","content":message })
    llm=OpenAI()
    stream=llm.chat.completions.create(model="gpt-5-mini", messages=messages, stream=True)
    response=""
    for chunk in stream:
        response+=chunk.choices[0].delta.content or ''
        yield response
    
gr.ChatInterface(fn=chat).launch(inbrowser=True)

    


