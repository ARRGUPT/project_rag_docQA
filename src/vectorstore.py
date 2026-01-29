from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vectorstore(documents, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", persist_dir="embeddings"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})
    
    # Path to the specific index file FAISS expects
    index_path = os.path.join(persist_dir, "index.faiss")

    # Only attempt to load if the folder AND the index file exist
    if os.path.exists(persist_dir) and os.path.exists(index_path):
        print(f"Loading existing FAISS vectorstore from '{persist_dir}'...")
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        
        if documents:
            print(f"Adding {len(documents)} new documents...")
            vectorstore.add_documents(documents)
            vectorstore.save_local(persist_dir)
    else:
        print("Creating new FAISS vectorstore...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore.save_local(persist_dir)

    return vectorstore


def load_vectorstore(persist_dir="embeddings", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})
    index_path = os.path.join(persist_dir, "index.faiss")
    
    if os.path.exists(index_path):
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    
    print(f"No existing index found at {index_path}")
    return None