from langchain_core.embeddings import Embeddings
from services.gemini_service import get_embedding


class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)