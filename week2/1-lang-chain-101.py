from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

llm=ChatOpenAI(model="gpt-5-nano")

response=llm.invoke("Hello,how are you")

print(response.content)


