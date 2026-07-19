import glob, shutil
from os import path
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,TextLoader
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

def get_documents():
    documents=[]
    kb_folders=glob.glob("week5/knowledge-base/*")
    for kb_folder in kb_folders:
        doc_type=path.basename(kb_folder)
        loader=DirectoryLoader(kb_folder,"**/*.md",loader_kwargs={'encoding': 'utf-8'}, loader_cls=TextLoader)
        docs=loader.load()
        for doc in docs:
            doc.metadata['doc_type']=doc_type
            documents.append(doc)
    return documents

def split_documents(documents:[]):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_documents(documents)
    return chunks

embeddings=HuggingFaceEmbeddings(model_name=VECTORIZING_MODEL)

def create_kb_vector_db():
    chunks=split_documents(get_documents())
    if(path.exists(DB_NAME)):
        shutil.rmtree(DB_NAME)
    vectorstore=Chroma.from_documents(documents=chunks,persist_directory=DB_NAME, embedding=embeddings)
    print(f"vectorstore created with {vectorstore._collection.count()} documents")

def get_vector_dimensions_info():
    vectorstore=Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    result=vectorstore.get(include=['embeddings'])
    print(f'vectors are of {len(result['embeddings'][0])} dimensions')


def print_vector_2DView():
    vectorstore=Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    result=vectorstore.get(include=['embeddings','documents','metadatas'])
    vectors=np.array(result['embeddings'])
    documents=result['documents']
    metadatas=result['metadatas']
    doc_types=[metadata['doc_type'] for metadata in metadatas ]
    colors=[['blue','green','red','orange'][['employees','contracts','products','company'].index(doc_type)]for doc_type in doc_types]
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 2D scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(title='2D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y'),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()

def print_vector_3DView():
    vectorstore=Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    result=vectorstore.get(include=['embeddings','documents','metadatas'])
    vectors=np.array(result['embeddings'])
    documents=result['documents']
    metadatas=result['metadatas']
    doc_types=[metadata['doc_type'] for metadata in metadatas ]
    colors=[['blue','green','red','orange'][['employees','contracts','products','company'].index(doc_type)]for doc_type in doc_types]
    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])

    fig.update_layout(title='3D Chroma Vector Store Visualization',
        scene=dict(xaxis_title='x',yaxis_title='y',zaxis_title='z'),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40)
    )

    fig.show()

def chat(message,history):
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
    