from fastapi_controllers import Controller, post
from services.rag_data import RAGData


class DataController(Controller):

    def __init__(self):
        self.rag_data = RAGData()

    @post("/init")
    def init_data(self):
        try:
            result = self.rag_data.init_data()
            return {
                "status": "ok",
                "details": result,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
            }