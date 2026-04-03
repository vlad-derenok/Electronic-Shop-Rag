import weaviate
from sentence_transformers import SentenceTransformer
from services.gemini_service import generate_answer

WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ElectronicsChunk"


class RAGAssistant:
    def __init__(self):
        self.client = weaviate.Client(WEAVIATE_URL)

        if not self.client.is_ready():
            raise RuntimeError("Weaviate not ready")

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def get_embedding(self, text):
        return self.embedder.encode(text).tolist()

    def retrieve(self, query, k=5):
        vector = self.get_embedding(query)

        result = (
            self.client.query
            .get(CLASS_NAME, ["text"])
            .with_near_vector({
                "vector": vector,
                "certainty": 0.6
            })
            .with_limit(k)
            .do()
        )

        if not result or not result.get("data"):
            return []

        return result["data"]["Get"].get(CLASS_NAME, [])

    def ask(self, question: str):
        docs = self.retrieve(question, k=5)

        if not docs:
            return {"answer": "❌ Нет данных в базе", "source_chunk": None}
        top_chunk = docs[0]["text"]
        context = "\n\n".join(d["text"] for d in docs)

        prompt = f"""
Ты отвечаешь только по контексту.
Если ответа нет — скажи "нет информации".

Контекст:
{context}

Вопрос: {question}
Ответ:
"""

        answer = generate_answer(prompt)

        return {
            "answer": answer,
            "source_chunk": top_chunk 
        }