from fastapi_controllers import Controller, post
from services.rag_data import RAGData


class DataController(Controller):

    def __init__(self):
        self.rag_data = RAGData()

    @post("/init")
    def init_data(self):
        try:
            results = {}
            results["gadgets"] = self.rag_data.init_index("data/gadgets.txt", "Gadgets")
            results["headphones"] = self.rag_data.init_index("data/headphones.txt", "Headphones")
            results["laptops"] = self.rag_data.init_index("data/laptops.txt", "Laptops")
            results["smartphones"] = self.rag_data.init_index("data/smartphones.txt", "Smartphones")
            return {
                "status": "ok",
                "details": results,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
            }