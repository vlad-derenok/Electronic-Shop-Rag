import os
import weaviate
from dotenv import load_dotenv
from fuzzywuzzy import fuzz  
from services.evaluator import precision_at_k_fuzzy
import ollama
from services.prompts import build_rag_prompt, build_system_prompt
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),          
        logging.FileHandler("rag.log")    
    ]
)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
CLASS_NAME = os.getenv("INDEX_NAME")


class RAGAssistant:
    def __init__(self):
        self.client = weaviate.Client(WEAVIATE_URL)

        if not self.client.is_ready():
            raise RuntimeError("Weaviate not ready")

        self.history = [] 

    def normalize_text(self, text: str) -> str:
        """
        Приводим текст к нижнему регистру и убираем лишние пробелы/переносы
        """
        return " ".join(text.lower().split())

    def precision_at_k_fuzzy(retrieved_chunks, relevant_chunks, threshold=70):
        k = len(retrieved_chunks)
        relevant_count = 0

        for r_chunk in retrieved_chunks:
            for rel_chunk in relevant_chunks:
                if fuzz.partial_ratio(r_chunk, rel_chunk) >= threshold:  # было ratio
                    relevant_count += 1
                    break

        return relevant_count / k if k > 0 else 0

    def evaluate(self, question: str, relevant_chunks: list, k: int = 5, threshold=70):
        docs = self.retrieve(question, k=k)

        if not docs:
            return {
                "precision": 0.0,
                "retrieved": [],
                "relevant": relevant_chunks
            }

        retrieved_chunks = [d["text"] for d in docs]

        score = precision_at_k_fuzzy(
            retrieved_chunks,
            relevant_chunks,
            threshold
        )

        return {
            "precision": score,
            "retrieved": retrieved_chunks,
            "relevant": relevant_chunks
        }

    def build_history(self, max_turns=3):
        recent = self.history[-max_turns:]

        history_text = ""
        for item in recent:
            history_text += f"Вопрос: {item['question']}\n"
            history_text += f"Ответ: {item['answer']}\n\n"

        return history_text

    def get_embedding(self, text):
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response["embedding"]

    def retrieve(self, query, k=5, alpha=0.5):
        vector = self.get_embedding(query)

        result = (
            self.client.query
            .get(CLASS_NAME, ["text"])
            .with_hybrid(
                query=query,
                vector=vector,
                alpha=alpha
            )
            .with_limit(k)
            .do()
        )

        if not result or not result.get("data"):
            return []

        return result["data"]["Get"].get(CLASS_NAME, [])
    

    def generate_answer(self, messages):
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages
        )
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        logger.info(f"Tokens — input: {input_tokens}, output: {output_tokens}, total: {input_tokens + output_tokens}")

        return response["message"]["content"]

    def ask(self, question: str, relevant_chunks: list = None):
        history_text = self.build_history()

        history_messages = []
        for item in self.history[-3:]:
            history_messages.append({"role": "user", "content": item["question"]})
            history_messages.append({"role": "assistant", "content": item["answer"]})

        docs = self.retrieve(question, k=5)

        if not docs:
            return {
                "answer": "Нет данных в базе",
                "chunks": [],
                "source_chunk": None,
                "precision": 0.0 if relevant_chunks else None
            }

        chunks = [d["text"] for d in docs]
        top_chunk = chunks[0]
        context = "\n\n".join(chunks)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            *history_messages,
            {"role": "user", "content": build_rag_prompt(context, question, history=history_text)}
        ]

        answer = self.generate_answer(messages)

        self.history.append({"question": question, "answer": answer})

        result = {
            "answer": answer,
            "chunks": chunks,
            "source_chunk": top_chunk
        }

        if relevant_chunks:
            eval_result = self.evaluate(question, relevant_chunks)
            result["precision"] = eval_result["precision"]

        return result