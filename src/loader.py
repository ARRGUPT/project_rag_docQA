from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_documents(data_dir="data/"):
    # Load all PDFs in the directory
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs_split = text_splitter.split_documents(documents)
    return docs_split