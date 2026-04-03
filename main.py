import weaviate
import gradio as gr
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from google import genai

from rag_utils import load_documents, chunk_documents

load_dotenv()

WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ElectronicsChunk"
DATA_FOLDER = "data"

client = weaviate.Client(WEAVIATE_URL)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    return embedder.encode(text).tolist()

def generate_answer(prompt):
    response = gemini.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )
    return response.text

# # ===== CREATE CLASS =====
# if not client.schema.exists(CLASS_NAME):
#     client.schema.create_class({
#         "class": CLASS_NAME,
#         "properties": [{"name": "text", "dataType": ["text"]}],
#         "vectorizer": "none"
#     })

# # ===== LOAD DATA =====
# docs = load_documents(DATA_FOLDER)
# chunks = chunk_documents(docs, strategy="overlap")

# print("Загружаем чанки в Weaviate...")

# for chunk in chunks:
#     vector = get_embedding(chunk)
#     client.data_object.create(
#         data_object={"text": chunk},
#         class_name=CLASS_NAME,
#         vector=vector
#     )

# print("Готово!")

def rerank(query, chunks):
    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = reranker.predict(pairs)  # ✅ теперь работает

    scored = list(zip(chunks, scores))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [item[0] for item in scored]

def retrieve(query, k=5):
    vector = get_embedding(query)

    result = (
        client.query
        .get(CLASS_NAME, ["text"])
        .with_near_vector({
            "vector": vector,
            "certainty": 0.5
        })
        .with_limit(k * 3) 
        .do()
    )

    if not result or not result.get("data"):
        return []

    items = result["data"]["Get"].get(CLASS_NAME, [])

    if not items:
        return []

    seen = set()
    unique = []
    for item in items:
        text = item["text"]
        if text not in seen:
            seen.add(text)
            unique.append(item)

    reranked = rerank(query, unique)

    return reranked[:k]

def answer_question(query):
    results = retrieve(query, k=5)

    if not results:
        return "❌ Нет данных в базе", ""

    chunks = [item["text"] for item in results]

    context = "\n\n".join(chunks)

    prompt = f"""
Ответь на вопрос строго по контексту.

Контекст:
{context}

Вопрос: {query}
Ответ:
"""

    answer = generate_answer(prompt)

    chunks_text = "\n\n---\n\n".join(chunks)

    return answer, chunks_text


def evaluate_manual(query, retrieved_chunks):
    print("\n=== Оценка релевантности ===")

    relevant = 0

    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n[{i+1}] {chunk[:200]}...")
        score = input("Релевантно? (1 = да, 0 = нет): ")

        if score == "1":
            relevant += 1

    precision = relevant / len(retrieved_chunks) if retrieved_chunks else 0

    print(f"\n📊 Precision@{len(retrieved_chunks)} = {precision:.2f}")


def chat_interface(query):
    answer, chunks = answer_question(query)
    return answer, chunks

state = {
    "chunks": [],
    "index": 0,
    "relevant": 0,
    "answer": "" 
}

def start_evaluation(query):
    results = retrieve(query, k=5)

    if not results:
        return "Нет данных", "", "", ""

    chunks = [item["text"] for item in results]

    state["chunks"] = chunks
    state["index"] = 0
    state["relevant"] = 0

    answer, _ = answer_question(query)

    state["answer"] = answer 

    first_chunk = chunks[0]

    return answer, first_chunk, "0", f"Chunk 1 / {len(chunks)}"


def mark_relevant():
    state["relevant"] += 1
    return next_chunk()


def mark_irrelevant():
    return next_chunk()


def next_chunk():
    state["index"] += 1

    if state["index"] >= len(state["chunks"]):
        precision = state["relevant"] / len(state["chunks"])

        return (
            "Оценка завершена",
            "",
            f"{precision:.2f}",
            f"Precision@{len(state['chunks'])}"
        )

    chunk = state["chunks"][state["index"]]

    return (
    state["answer"], 
    chunk,
    "",
    f"Chunk {state['index']+1} / {len(state['chunks'])}"
    )

def rebuild_db(strategy):
    client.schema.delete_all()

    client.schema.create_class({
        "class": CLASS_NAME,
        "properties": [{"name": "text", "dataType": ["text"]}],
        "vectorizer": "none"
    })

    docs = load_documents(DATA_FOLDER)
    chunks = chunk_documents(docs, strategy=strategy)

    for chunk in chunks:
        vector = get_embedding(chunk)
        client.data_object.create(
            data_object={"text": chunk},
            class_name=CLASS_NAME,
            vector=vector
        )

    return f"База пересобрана ({strategy}), chunks: {len(chunks)}"


with gr.Blocks() as demo:
    gr.Markdown("## RAG")

    query = gr.Textbox(label="Вопрос")

    answer_box = gr.Textbox(label="Ответ")

    chunk_box = gr.Textbox(label="Проверяемый chunk")

    precision_box = gr.Textbox(label="Precision@k")

    status = gr.Textbox(label="Статус")

    start_btn = gr.Button("Найти и начать оценку")
    rel_btn = gr.Button("Релевантно")
    irrel_btn = gr.Button("Нерелевантно")
    strategy = gr.Dropdown(    ["fixed", "overlap", "semantic"],    value="overlap",    label="Стратегия чанкинга")
    start_btn.click(
        start_evaluation,
        inputs=query,
        outputs=[answer_box, chunk_box, precision_box, status]
    )

    rel_btn.click(
        mark_relevant,
        outputs=[answer_box, chunk_box, precision_box, status]
    )

    irrel_btn.click(
        mark_irrelevant,
        outputs=[answer_box, chunk_box, precision_box, status]
    )

    rebuild_btn = gr.Button("Пересобрать базу")
    status_box = gr.Textbox(label="Статус")
    rebuild_btn.click(
    rebuild_db,
    inputs=strategy,
    outputs=status_box
    )
if __name__ == "__main__":
    demo.launch()
    