import os
import ollama
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from services.prompts import build_rag_prompt, build_system_prompt, build_router_prompt
from services.rag_data import RAGData
import logging

load_dotenv()

logger = logging.getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: str
    answer: str
    precision: float
    category: str 


class RAGGraph:
    def __init__(self):
        self.rag_data = RAGData()
        self.graph = self._build_graph()

    def _get_embedding(self, text: str) -> list:
        return self.rag_data.get_embedding(text)

    def _retrieve_from_index(self, query: str, index_name: str, k: int = 5) -> list:
        vector = self._get_embedding(query)
        result = (
            self.rag_data.client.query
            .get(index_name, ["text"])
            .with_hybrid(query=query, vector=vector, alpha=0.5)
            .with_limit(k)
            .do()
        )
        if not result or not result.get("data"):
            return []
        return result["data"]["Get"].get(index_name, [])


    def router_node(self, state: RAGState) -> RAGState:
        question = state["question"]

        response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": build_router_prompt()
            },
            {"role": "user", "content": question}
        ]
    )

        category = response["message"]["content"].strip()

        for valid in ["Gadgets", "Headphones", "Laptops", "Smartphones"]:
            if valid.lower() in category.lower():
                category = valid
                break
        else:
            category = "Unknown"

        logger.info(f"Router decision: {category}")
        return {**state, "category": category}

    def gadgets_node(self, state: RAGState) -> RAGState:
        docs = self._retrieve_from_index(state["question"], "Gadgets")
        context = "\n\n".join([d["text"] for d in docs])
        return {**state, "context": context}

    def headphones_node(self, state: RAGState) -> RAGState:
        docs = self._retrieve_from_index(state["question"], "Headphones")
        context = "\n\n".join([d["text"] for d in docs])
        return {**state, "context": context}

    def laptops_node(self, state: RAGState) -> RAGState:
        docs = self._retrieve_from_index(state["question"], "Laptops")
        context = "\n\n".join([d["text"] for d in docs])
        return {**state, "context": context}

    def smartphones_node(self, state: RAGState) -> RAGState:
        docs = self._retrieve_from_index(state["question"], "Smartphones")
        context = "\n\n".join([d["text"] for d in docs])
        return {**state, "context": context}

    def generate_node(self, state: RAGState) -> RAGState:
        history_text = self.rag_data.get_history()

        history_messages = []
        for msg in self.rag_data.state["messages"][-6:]:
            if isinstance(msg, HumanMessage):
                history_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_messages.append({"role": "assistant", "content": msg.content})

        messages = [
            {"role": "system", "content": build_system_prompt()},
            *history_messages,
            {"role": "user", "content": build_rag_prompt(
                state["context"], state["question"], history=history_text
            )}
        ]

        response = ollama.chat(model="qwen2.5:7b", messages=messages)
        answer = response["message"]["content"]

        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)
        logger.info(f"Tokens — input: {input_tokens}, output: {output_tokens}")

        self.rag_data.update_state(
            question=state["question"],
            answer=answer,
            context=state["context"]
        )

        return {**state, "answer": answer}

    def unknown_node(self, state: RAGState) -> RAGState:
        return {**state, "answer": "no information", "context": ""}

    # --- РОУТИНГ ---

    def route_by_category(self, state: RAGState) -> str:
        category = state.get("category", "Unknown")
        routes = {
            "Gadgets": "gadgets",
            "Headphones": "headphones",
            "Laptops": "laptops",
            "Smartphones": "smartphones",
        }
        return routes.get(category, "unknown")


    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("router", self.router_node)
        graph.add_node("gadgets", self.gadgets_node)
        graph.add_node("headphones", self.headphones_node)
        graph.add_node("laptops", self.laptops_node)
        graph.add_node("smartphones", self.smartphones_node)
        graph.add_node("generate", self.generate_node)
        graph.add_node("unknown", self.unknown_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges("router", self.route_by_category, {
            "gadgets": "gadgets",
            "headphones": "headphones",
            "laptops": "laptops",
            "smartphones": "smartphones",
            "unknown": "unknown",
        })

        graph.add_edge("gadgets", "generate")
        graph.add_edge("headphones", "generate")
        graph.add_edge("laptops", "generate")
        graph.add_edge("smartphones", "generate")
        graph.add_edge("generate", END)
        graph.add_edge("unknown", END)

        return graph.compile()

    def ask(self, question: str) -> dict:
        initial_state: RAGState = {
            "messages": self.rag_data.state["messages"],
            "question": question,
            "context": "",
            "answer": "",
            "precision": 0.0,
            "category": ""
        }

        result = self.graph.invoke(initial_state)

        return {
            "answer": result["answer"],
            "category": result["category"],
            "context": result["context"]
        }