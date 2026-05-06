from fastapi_controllers import Controller, post
from pydantic import BaseModel
from services.rag_graph import RAGGraph

rag_graph = RAGGraph()


class Question(BaseModel):
    question: str


class AssistantController(Controller):

    @post("/ask")
    def ask(self, q: Question):
        try:
            result = rag_graph.ask(q.question)
            return {
                "answer": result["answer"],
                "category": result["category"]
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}