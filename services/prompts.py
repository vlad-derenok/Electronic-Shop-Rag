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