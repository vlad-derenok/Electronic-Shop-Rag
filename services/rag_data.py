import os
import weaviate
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate as LangChainWeaviate

from services.gemini_langchain_embeddings import GeminiEmbeddings

load_dotenv()

WEAVIATE_URL = "http://localhost:8080"
INDEX_NAME = "ElectronicsChunk"
DATA_FOLDER = "data"


class RAGData:
    def __init__(self, data_folder: str = DATA_FOLDER):
        self.client = weaviate.Client(WEAVIATE_URL)
        self.data_folder = data_folder
        self.vector_store = None
        self.embeddings = GeminiEmbeddings()

        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready")

    def init_data(self):
        if not os.path.exists(self.data_folder):
            raise RuntimeError(f"Data folder not found: {self.data_folder}")

        loader = DirectoryLoader(self.data_folder, glob="*.txt")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = splitter.split_documents(documents)

        self.vector_store = LangChainWeaviate(
            client=self.client,
            index_name=INDEX_NAME,
            text_key="text",
            embedding=self.embeddings
        )

        ids = self.vector_store.add_documents(splits)

        return {
            "documents": len(documents),
            "chunks": len(splits),
            "stored_vectors": len(ids),
        }

    def get_vector_store(self):
        if self.vector_store is None:
            raise RuntimeError("Vector store is not initialized. Run init_data() first.")
        return self.vector_store