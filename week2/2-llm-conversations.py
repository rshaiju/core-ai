from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

grumpy_agent_system_prompt="""
You are a grumpy person 
- with attitude but wise
- tries to prove your point and presence always
- keep the statements short
- you are not a techie
"""

humble_agent_system_prompt="""
You are a humble person
- but knowledgable
- capable of dealing with any kind of people
- keep the statements short
- you are not a techie
"""

grumpy_agent_msgs=["Hello man!!"]
humble_agent_msgs=["Hello sir!!"]


def grumpy_agent_talk():
  messages=[{"role":"system","content":grumpy_agent_system_prompt}]
  for grumpy,humble in zip(grumpy_agent_msgs,humble_agent_msgs):
    messages.append({"role":"assistant", "content": grumpy})
    messages.append({"role":"user", "content": humble})
  llm=OpenAI()
  response=llm.chat.completions.create(model="gpt-5-mini", messages=messages)
  replymsg=response.choices[0].message.content
  grumpy_agent_msgs.append(replymsg)
  return replymsg

def humble_agent_talk():
  messages=[{"role":"system","content":humble_agent_system_prompt}]
  for grumpy,humble in zip(grumpy_agent_msgs,humble_agent_msgs):
    messages.append({"role":"user", "content": grumpy})
    messages.append({"role":"assistant", "content": humble})
  messages.append({"role":"user", "content": grumpy_agent_msgs[-1]})  
  llm=OpenAI()
  response=llm.chat.completions.create(model="gpt-5-mini", messages=messages)
  replymsg=response.choices[0].message.content
  humble_agent_msgs.append(replymsg)
  return replymsg

def agents_meet():
    print("Grumpy:" + grumpy_agent_msgs[-1])
    print("Humble:" + humble_agent_msgs[-1])
    for i in range(5):
        message=grumpy_agent_talk()
        print("Grumpy:" + message)
        reply=humble_agent_talk()
        print("Humble:" + reply)

agents_meet()