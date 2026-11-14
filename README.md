# AI Document Assistant

A Python + Generative AI application that lets users upload PDF, DOCX, and TXT documents, ask questions, and receive answers grounded in retrieved document passages.

## Features

- PDF, DOCX, and TXT text extraction
- Overlapping text chunking
- Sentence Transformers embeddings
- FAISS semantic search
- Groq LLM answer generation using a RAG workflow
- Streamlit chat interface with source filename and page references

## Requirements

- Python 3.10 or 3.11 recommended
- A Groq API key
- Internet connection for the first embedding-model download and API calls

## Setup on Windows

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your Groq API key.

Run:

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

## How it works

1. Extract text from uploaded files.
2. Split text into overlapping chunks.
3. Convert chunks into vectors using Sentence Transformers.
4. Index vectors in FAISS.
5. Retrieve the most similar chunks for each question.
6. Send the retrieved context to the Groq chat model.
7. Display the answer and retrieved source references.

## Notes and limitations

- The vector index and chat history are kept in the active Streamlit session; they are not persistent.
- Scanned/image-only PDFs require OCR, which is not included in this starter version.
- Answers depend on the quality of extracted text and retrieved chunks. Verify important answers against the original documents.
- Do not upload confidential documents or expose your API key.
- For deployment, add `GROQ_API_KEY` and `GROQ_MODEL` in the hosting provider's secrets settings. Do not commit `.env`.

## GitHub

Before pushing, verify that `.env` is not tracked:

```bash
git status
```

Then:

```bash
git init
git add .
git commit -m "Build AI Document Assistant"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```
