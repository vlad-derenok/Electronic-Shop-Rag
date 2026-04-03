import os
from dotenv import load_dotenv 
from google import genai

load_dotenv() 

gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(prompt: str):
    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def get_embedding(text: str):
    response = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values