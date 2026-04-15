from services.rag_assistant import RAGAssistant

rag = RAGAssistant()

res1 = rag.ask("Сколько ОЗУ у iPhone 17 Pro Max?")
print("Ответ 1:", res1["answer"])

res2 = rag.ask("А сколько памяти у iPhone 17 256GB?")
print("Ответ 2:", res2["answer"])

print("\nИстория чата:")
for turn in rag.history:
    print(f"Q: {turn['question']}")
    print(f"A: {turn['answer']}\n")