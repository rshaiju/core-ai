import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

DB_NAME="kb_vectordb"
VECTORIZING_MODEL="all-MiniLM-L6-v2"

def get_vector_dimensions_info():
    embeddings=HuggingFaceEmbeddings(model_name=VECTORIZING_MODEL)
    vectorstore=Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
    result=vectorstore.get(include=['embeddings'])
    print(f'vectors are of {len(result['embeddings'][0])} dimensions')

def print_vector_2DView():
    embeddings=HuggingFaceEmbeddings(model_name=VECTORIZING_MODEL)
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
    embeddings=HuggingFaceEmbeddings(model_name=VECTORIZING_MODEL)
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

    
if __name__=="__main__":
    view_type=sys.argv[1] if len(sys.argv)>1 else "3d"
    match view_type:
        case "2d":
            print_vector_2DView()
        case "3d":
            print_vector_3DView()
        case _:
            get_vector_dimensions_info()
