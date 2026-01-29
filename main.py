import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Document QA", layout="wide")
st.title("📄 RAG-Based Document Question Answering")

# ---------------- Upload Section ---------------- #

st.header("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Upload"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(f"{API_BASE}/upload-documents/", files=files)

        if response.status_code == 200:
            st.success("Files uploaded and processing started.")
        else:
            st.error(response.text)

# ---------------- Question Section ---------------- #

st.header("Ask a Question")

question = st.text_input("Enter your question based on uploaded documents")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        response = requests.post(
            f"{API_BASE}/ask/",
            json={"question": question}
        )

        if response.status_code == 200:
            answer = response.json()["answer"]
            st.subheader("Answer")
            st.write(answer)
        else:
            st.error(response.text)
