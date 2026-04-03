# rag_utils.py
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_documents(folder_path):
    docs = []
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if fname.endswith(".txt"):
            docs.append(load_txt(path))
        elif fname.endswith(".pdf"):
            docs.append(load_pdf(path))
    return docs

def chunk_documents(docs, strategy="fixed", chunk_size=500, overlap=100):
    chunks = []

    for doc in docs:
        text = doc

        if strategy == "fixed":
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i+chunk_size])

        elif strategy == "overlap":
            step = chunk_size - overlap
            for i in range(0, len(text), step):
                chunks.append(text[i:i+chunk_size])

        elif strategy == "semantic":
            paragraphs = text.split("\n")

            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) < chunk_size:
                    current_chunk += p + "\n"
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = p + "\n"

            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks

def explain_chunks(chunks, query):
    explanations = []
    for chunk in chunks:
        score = sum(word in chunk.lower() for word in query.lower().split())
        explanations.append((chunk, score))
    explanations.sort(key=lambda x: x[1], reverse=True)
    return explanations