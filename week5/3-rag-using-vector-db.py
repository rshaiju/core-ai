import glob
from os import path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


MODEL="gpt-4.1-mini"
DB_NAME="kb_vectordb"

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

chunks=split_documents(get_documents())
print(len(chunks))

'''
def create_kb_vector_db():
    chunks=split_documents(get_documents())
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if(path.exists(DB_NAME)):
        Chroma(persist_directory=DB_NAME,embedding_function=embeddings).delete_collection()
    vectorstore=Chroma.from_documents(documents=chunks,persist_directory=DB_NAME, embedding=embeddings)
    print(f"vectorstore created with {vectorstore._collection.count()} documents")

create_kb_vector_db()
'''