from fastapi import FastAPI
from controllers.data import DataController
from controllers.assistant import AssistantController

app = FastAPI(title="RAG API")

app.include_router(DataController.create_router())
app.include_router(AssistantController.create_router())


@app.get("/")
def health():
    return {"status": "ok"}