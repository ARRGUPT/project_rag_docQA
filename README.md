# 📄 RAG-Based Document Question Answering System

A powerful document question-answering system that uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on uploaded PDF and text documents.

## 🎯 Features

- **Document Upload**: Support for PDF and TXT files
- **Vector Store**: FAISS-based vector database for efficient document retrieval
- **RAG Pipeline**: Uses LangChain to combine retrieval and generation
- **LLM Integration**: Groq API for fast and accurate responses
- **Web Interface**: Clean Streamlit UI for easy interaction
- **Rate Limiting**: Built-in request rate limiting for API protection
- **Background Processing**: Asynchronous document processing
- **Persistent Storage**: Embeddings saved locally for quick retrieval

## 🏗️ System Architecture

```
┌─────────────┐
│   Document  │
│   Uploader  │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│     Text     │
│  Splitter    │
└──────┬───────┘
       │
    ┌──┴──┬────┬────┐
    ▼     ▼    ▼    ▼
┌────────────────────────┐
│  Embedding Model       │
│  (HuggingFace)         │
└────────┬───────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 ┌──┴──┐   ┌──┴──┐
 │Vec 1│   │Vec N│
 └─────┘   └─────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│  Vector Store   │
│  (FAISS)        │
└────────┬────────┘
         │
         └──────────────┐
                        │
    ┌───────────────────┘
    │
    ▼
┌──────────────┐
│  Retriever   │
└────┬─────────┘
     │
     ▼
┌──────────────────────────┐
│  Most Relevant Chunks    │
│  + User Query            │
└────┬─────────────────────┘
     │
     ▼
┌──────────────────┐
│   LLM Prompt     │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│   LLM (Groq)     │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│    Response      │
└──────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- API Keys:
  - **GROQ_API_KEY**: Get from [Groq Console](https://console.groq.com)
  - **HUGGINGFACEHUB_API_TOKEN**: Get from [HuggingFace](https://huggingface.co/settings/tokens)

## 🚀 Setup Instructions

### 1. Clone or Download the Project

```bash
cd Project_rag_DocQA
```

### 2. Create and Activate Virtual Environment

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory with your API keys:

```env
GROQ_API_KEY="your_groq_api_key_here"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"
GOOGLE_API_KEY="your_google_api_key_here"
EXCHANGERATE_API_KEY="your_exchangerate_api_key_here"
```

**Important**: Never commit the `.env` file to version control (it's in `.gitignore`).

## 💻 Usage Instructions

### Step 1: Start the FastAPI Backend

Open a terminal and run:

```bash
cd Project_rag_DocQA
.\venv\Scripts\activate  # On Windows
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start the Streamlit Frontend (New Terminal)

Open another terminal and run:

```bash
cd Project_rag_DocQA
.\venv\Scripts\activate  # On Windows
streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Step 3: Use the Application

1. **Upload Documents**:
   - Click "Browse files" in the Upload Documents section
   - Select one or more PDF or TXT files
   - Click "Upload" button
   - Wait for "Files uploaded and processing started" message

2. **Ask Questions**:
   - Enter your question in the "Ask a Question" text box
   - Click "Get Answer" button
   - The system will retrieve relevant document chunks and generate an answer

## 📁 Project Structure

```
Project_rag_DocQA/
├── app.py                 # FastAPI backend application
├── main.py                # Streamlit frontend application
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore             # Git ignore file
├── README.md              # This file
│
├── src/
│   ├── loader.py          # Document loading utilities
│   ├── vectorstore.py      # FAISS vector store management
│   └── rag_chain.py        # RAG chain pipeline
│
├── data/                  # Uploaded documents storage
├── embeddings/            # FAISS embeddings persistence
└── venv/                  # Virtual environment (not in git)
```

## 🔌 API Endpoints

### GET `/`
Returns API health check.

**Response:**
```json
{
  "message": "Document QA API"
}
```

### POST `/upload-documents/`
Upload PDF or TXT files for processing.

**Request:**
- Files: Multi-part form data with files
- Supported types: `.pdf`, `.txt`

**Response:**
```json
{
  "message": "Files uploaded. Processing in background."
}
```

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/upload-documents/" \
  -F "files=@document.pdf" \
  -F "files=@another.txt"
```

### POST `/ask/`
Ask a question based on uploaded documents.

**Request:**
```json
{
  "question": "What is mentioned in the documents?"
}
```

**Response:**
```json
{
  "question": "What is mentioned in the documents?",
  "answer": "The documents mention..."
}
```

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/ask/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

## ⚙️ Configuration

### Chunking Parameters (in `src/vectorstore.py`)

```python
chunk_size=1000          # Size of each document chunk
chunk_overlap=200        # Overlap between chunks for context
```

### LLM Parameters (in `app.py`)

```python
model="llama-3.1-8b-instant"  # Groq model
temperature=0.2               # Lower = more deterministic
max_tokens=1000               # Maximum response length
```

### Rate Limiting (in `app.py`)

```python
RATE_LIMIT = 10    # Max requests
WINDOW_SIZE = 60   # Per 60 seconds
```

## 🐛 Troubleshooting

### "No module named 'pypdf'"

**Solution:**
```bash
.\venv\Scripts\python -m pip install pypdf
```

### "GROQ_API_KEY not found"

**Solution:**
- Ensure `.env` file exists in the project root
- Check that `GROQ_API_KEY` is set correctly
- Restart the backend server

### "No documents indexed yet"

**Solution:**
- Upload documents first using the Streamlit UI
- Wait for "Files uploaded and processing started" message
- Processing may take a few moments for large files

### "Rate limit exceeded"

**Solution:**
- Wait 60 seconds before making another request from the same IP
- Increase `RATE_LIMIT` in `app.py` if needed

### Slow document processing

**Solution:**
- Reduce `chunk_size` in `src/vectorstore.py`
- Check available system memory
- Process documents in smaller batches

### Empty FAISS vectorstore error

**Solution:**
- This error has been fixed in the latest version
- Ensure all dependencies are updated: `pip install -r requirements.txt`

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI |
| Frontend | Streamlit |
| RAG Framework | LangChain |
| LLM Provider | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Document Loaders | PyPDFLoader, TextLoader |
| Server | Uvicorn |

## 📊 Example Workflows

### Workflow 1: Customer Support Documents

1. Upload customer service guides and FAQs
2. Ask: "How do I reset my password?"
3. System retrieves relevant guide sections and generates step-by-step answer

### Workflow 2: Technical Documentation

1. Upload API documentation
2. Ask: "What parameters does the POST endpoint accept?"
3. System retrieves endpoint specs and provides detailed response

### Workflow 3: Research Papers

1. Upload research papers
2. Ask: "What are the main findings?"
3. System summarizes and extracts key conclusions

## 📈 Performance Tips

1. **Optimize Chunks**: Adjust `chunk_size` based on document type
2. **Batch Processing**: Upload documents in batches to improve speed
3. **Cache Embeddings**: Embeddings are cached in `embeddings/` folder
4. **Monitor Resources**: Check system memory during processing
5. **Rate Limiting**: Adjust based on your API quota

## 🔐 Security Considerations

⚠️ **Important**: 
- Never commit `.env` file to version control
- Keep API keys private and secure
- Use environment variables, not hardcoded keys
- Rotate API keys regularly
- Restrict file upload types to PDF and TXT only

## 📝 File Size Limits

- **Recommended**: Documents up to 50MB each
- **Optimal**: Multiple documents under 10MB
- **Processing**: Large documents may take several minutes

## 🤝 Contributing

Improvements welcome! Consider:
- Adding support for more file formats (DOCX, PPTX)
- Implementing caching for frequently asked questions
- Adding multi-language support
- Improving error handling

## 📞 Support

For issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review error messages in terminal output
3. Check API logs in uvicorn terminal
4. Verify all dependencies are installed: `pip list`

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com)
- Powered by [Groq](https://groq.com) for fast LLM inference
- Embeddings from [HuggingFace](https://huggingface.co)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- Web interface with [Streamlit](https://streamlit.io)

---

**Last Updated**: January 29, 2026

Happy document querying! 🚀
