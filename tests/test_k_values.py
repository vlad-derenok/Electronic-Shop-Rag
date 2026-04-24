from services.rag_assistant import RAGAssistant

assistant = RAGAssistant()

results = assistant.evaluate_k_values(
    question="Какой срок гарантии у iPhone 17 Pro Max?",
    relevant_chunks=[
        "Смартфон Apple iPhone 17 Pro Max ... Гарантия: 1 год"
    ]
)

for r in results:
    print(f"k={r['k']} | precision={r['precision']:.2f}")