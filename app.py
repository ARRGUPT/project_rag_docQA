import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from src.vectorstore import create_vectorstore, load_vectorstore
from src.rag_chain import build_rag_chain

# Initialize LLM
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=1000)


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="RAG-Based QA System")

# ------------------ Rate Limiting ------------------ #
REQUEST_COUNT = {}
RATE_LIMIT = 10  # requests
WINDOW_SIZE = 60  # seconds

import time

def rate_limiter(request: Request):
    # Extract client IP address from request
    ip = request.client.host if request.client else "unknown"
    
    current_time = time.time()
    window_start = current_time - WINDOW_SIZE

    if ip not in REQUEST_COUNT:
        REQUEST_COUNT[ip] = []

    REQUEST_COUNT[ip] = [t for t in REQUEST_COUNT[ip] if t > window_start]

    if len(REQUEST_COUNT[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    REQUEST_COUNT[ip].append(current_time)

# ------------------ Request Models ------------------ #

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)

# ------------------ Document Processing ------------------ #

def process_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs_split = splitter.split_documents(documents)

    create_vectorstore(docs_split)  # adds to FAISS

# ------------------ Routes ------------------ #

@app.get("/")
def hello():
    return {'message':'Document QA API'}

@app.post("/upload-documents/")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    allowed_types = [".pdf", ".txt"]

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_types:
            raise HTTPException(status_code=400, detail=f"{file.filename} not supported")

        save_path = os.path.join(DATA_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(process_document, save_path)

    return {"message": "Files uploaded. Processing in background."}


@app.post("/ask/")
async def ask_question(request: QuestionRequest, _: None = Depends(rate_limiter)):
    vectorstore = load_vectorstore()

    if vectorstore is None or not vectorstore.index.ntotal:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload documents first.")

    rag_chain = build_rag_chain(vectorstore, llm)

    answer = rag_chain.invoke(request.question)

    return {"question": request.question, "answer": answer}
