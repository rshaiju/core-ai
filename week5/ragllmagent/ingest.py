import glob, shutil
from os import path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

def create_kb_vector_db():
    embeddings=HuggingFaceEmbeddings(model_name=VECTORIZING_MODEL)
    chunks=split_documents(get_documents())
    if(path.exists(DB_NAME)):
        shutil.rmtree(DB_NAME)
    vectorstore=Chroma.from_documents(documents=chunks,persist_directory=DB_NAME, embedding=embeddings)
    print(f"vectorstore created with {vectorstore._collection.count()} documents")

if __name__ == "__main__":
    create_kb_vector_db()