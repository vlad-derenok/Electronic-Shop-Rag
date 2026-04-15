from services.rag_assistant import RAGAssistant
from fuzzywuzzy import fuzz


class RAGEvaluator:
    def __init__(self, question: str, relevant_chunks: list, k: int = 5):
        self.rag = RAGAssistant()
        self.question = question
        self.relevant_chunks = relevant_chunks
        self.k = k

    def debug_scores(self):
        docs = self.rag.retrieve(self.question, k=self.k)
        retrieved_chunks = [d["text"] for d in docs]

        for i, chunk in enumerate(retrieved_chunks):
            score = fuzz.partial_ratio(chunk, self.relevant_chunks[0])
            print(f"Chunk {i+1}: score={score}")
            print(f"Текст: {chunk[:100]}")
            print()

    def run(self):
        self.debug_scores()

        result = self.rag.ask(self.question, relevant_chunks=self.relevant_chunks)

        print("Ответ модели:", result["answer"])
        print("Использованные чанки:", result["chunks"])
        print("Главный чанк:", result["source_chunk"])
        print("Precision@k:", result.get("precision"))

        return result


if __name__ == "__main__":
    evaluator = RAGEvaluator(
        question="Какой срок гарантии у iPhone 17 Pro Max?",
        relevant_chunks=[
            """Смартфон Apple iPhone 17 Pro Max 512GB
            Основные характеристики
            ..."""
        ]
    )
    evaluator.run()