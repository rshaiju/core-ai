from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import gradio as gr

MODEL="gpt-4.1-mini"
DB_NAME="kb_vectordb"
VECTORIZING_MODEL="all-MiniLM-L6-v2"

def chat(message,history):
    embeddings=HuggingFaceEmbeddings(model_name=VECTORIZING_MODEL)
    vectorstore=Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    retriever=vectorstore.as_retriever()
    docs=retriever.invoke(message)
    kb="\n\n".join(doc.page_content for doc in docs)
    system_prompt=f"""
    You are a helpful assistant representing the company Insurellm with a knowledgebase
    Insurellm's users will be interacting with you
    If the information is not available in KB, just reply with statement "Sorry,I don't know about it". 
    If the information is relevant to the domain, you will help them with details
    KB: {kb}
    """
    messages=[
        SystemMessage(system_prompt),
        HumanMessage(message)
    ]
    llm=ChatOpenAI(model=MODEL, temperature=0)
    response=llm.invoke(messages)
    return response.content

gr.ChatInterface(fn=chat).launch(inbrowser=True)
    