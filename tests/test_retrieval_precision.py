from services.rag_assistant import RAGAssistant
from fuzzywuzzy import fuzz


class RAGEvaluator:
    def __init__(self, question: str, relevant_chunks: list, k: int = 5):
        self.rag = RAGAssistant()
        self.question = question
        self.relevant_chunks = relevant_chunks
        self.k = k

    def mean_reciprocal_rank(self, retrieved_chunks, relevant_chunks, threshold=70):

        for i, chunk in enumerate(retrieved_chunks):
            for rel_chunk in relevant_chunks:
                similarity = fuzz.partial_ratio(chunk, rel_chunk)

                print(f"Chunk {i+1}: similarity={similarity}")

                if similarity >= threshold:
                    print(f"Relevant chunk found at rank {i+1}")
                    return 1 / (i + 1)

        print("Relevant chunk not found")
        return 0

    def debug_scores(self):
        docs = self.rag.retrieve(self.question, k=self.k)
        retrieved_chunks = [d["text"] for d in docs]

        for i, chunk in enumerate(retrieved_chunks):
            score = fuzz.partial_ratio(chunk, self.relevant_chunks[0])
            print(f"Chunk {i+1}: score={score}")
            print(f"Текст: {chunk[:100]}")
            print()

    def average_similarity_score(self, retrieved_chunks, relevant_chunks):

        scores = []

        for chunk in retrieved_chunks:
            best_score = 0

            for rel_chunk in relevant_chunks:
                similarity = fuzz.partial_ratio(chunk, rel_chunk)

                if similarity > best_score:
                    best_score = similarity

            scores.append(best_score)

        if not scores:
            return 0

        return sum(scores) / len(scores)
    
    def answer_quality(self, model_answer: str, expected_answer: str, threshold=85):

        score = fuzz.partial_ratio(
            model_answer.lower(),
            expected_answer.lower()
        )

        return {
            "score": score,
            "correct": score >= threshold
        }

    def run(self, expected_answer: str):
        docs = self.rag.retrieve(self.question, k=self.k)
        retrieved_chunks = [d["text"] for d in docs]

        result = self.rag.ask(
            self.question,
            relevant_chunks=self.relevant_chunks
        )

        model_answer = result["answer"]

        mrr = self.mean_reciprocal_rank(
            retrieved_chunks,
            self.relevant_chunks,
            threshold=70
        )

        avg_similarity = self.average_similarity_score(
            retrieved_chunks,
            self.relevant_chunks
        )

        answer_eval = self.answer_quality(
            model_answer,
            expected_answer
        )

        print("\n=== RESULTS ===")
        print("Answer:", model_answer)
        print("MRR:", mrr)
        print("Avg similarity:", avg_similarity)
        print("Answer score:", answer_eval["score"])
        print("Correct:", answer_eval["correct"])

        return {
            "answer": model_answer,
            "mrr": mrr,
            "avg_similarity": avg_similarity,
            "answer_score": answer_eval["score"],
            "correct": answer_eval["correct"]
        }

if __name__ == "__main__":
    evaluator = RAGEvaluator(
        question="какой iphone поддерживает esim?",
        relevant_chunks=[
            """Смартфон Apple iPhone 17 Pro 256GB\nХарактеристики процессора: Apple A19 Pro , 6 ядерный, 4.26 ГГц\nЭкран: 6.3 \" 1206x2622 пикселей, OLED , 120 Гц\nПамять: ОЗУ 12 ГБ , 256 ГБ\nПоддержка eSIM: Нет\nNFC: Да\nРазрешение основного модуля камеры: 48 Мп\nГарантия: 1 год
            ..."""
        ]
    )
    evaluator.run(expected_answer="Смартфон Apple iPhone 17 Pro Max 512GB поддерживает eSIM")