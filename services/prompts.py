def build_rag_prompt(context, question, history=""):
    history_block = f"{history}" if history.strip() else "none"

    return f"""
Conversation history:
{history_block}

Knowledge base context:
{context}

Question: {question}

Instructions:
- If the question is about the conversation history (e.g. "what did I ask", "previous question") — answer ONLY based on the conversation history above, ignore the context
- If the question is on topic — answer based on the knowledge base context
- If the answer is not found anywhere — say "no information"

Answer:
"""


def build_history_prompt(history: str, question: str) -> str:
    return f"""
You are an assistant with memory.

Answer ONLY based on the conversation history.
If the answer is not in the history — say "no information".

History:
{history}

Question: {question}

Answer:
"""


def build_system_prompt():
    return """
You are an AI assistant.

Rules:
- Answer briefly and to the point
- Do not make up facts
- If there is no information — say "no information"
- Use the context if it is provided
"""

def build_router_prompt():
    return """You are a router. Based on the user's question, determine which index to search in.

Index contents:
- Gadgets: product specifications, characteristics, prices, warranty info for phones, laptops, headphones and other electronics
- Headphones: guide on how to choose headphones, what parameters to consider
- Laptops: guide on how to choose laptops, what parameters to consider
- Smartphones: guide on how to choose smartphones, what parameters to consider

Rules:
- If the question is about specific product specs, price, warranty, model comparison → Gadgets
- If the question is about how to choose headphones → Headphones
- If the question is about how to choose a laptop → Laptops
- If the question is about how to choose a smartphone → Smartphones
- If none of the above → Unknown

Reply with ONLY one word — the index name."""