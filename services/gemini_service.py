import os
from dotenv import load_dotenv 
from google import genai
import ollama

load_dotenv() 

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(prompt, history_messages=None):
    messages = []

    if history_messages:
        messages.extend(history_messages)

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=messages
    )

    return response["message"]["content"]

def get_embedding(text: str):
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values