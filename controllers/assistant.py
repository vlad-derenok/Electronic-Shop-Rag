from fastapi_controllers import Controller, post
from pydantic import BaseModel

from services.rag_assistant import RAGAssistant

rag_assistant = RAGAssistant()


class Question(BaseModel):
    question: str


class AssistantController(Controller):

    @post("/ask")
    def ask(self, q: Question):
        try:
            answer = rag_assistant.ask(q.question)
            return {"answer": answer}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}