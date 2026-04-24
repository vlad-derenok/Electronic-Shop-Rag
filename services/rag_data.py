import os
import weaviate
import logging
import ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag.log")
    ]
)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
INDEX_NAME = os.getenv("INDEX_NAME")
DATA_FOLDER = os.getenv("DATA_FOLDER")

class RAGData:
    def __init__(self, data_folder: str = DATA_FOLDER):
        self.client = weaviate.Client(WEAVIATE_URL)
        self.data_folder = data_folder

        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready")

        logger.info("RAGData initialized, Weaviate connected")

    def get_embedding(self, text: str) -> list:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response["embedding"]

    def init_data(self):
        if not os.path.exists(self.data_folder):
            raise RuntimeError(f"Data folder not found: {self.data_folder}")

        logger.info(f"Loading documents from: {self.data_folder}")

        try:
            self.client.schema.delete_class(INDEX_NAME)
            logger.info(f"Deleted existing index: {INDEX_NAME}")
        except Exception:
            pass

        self.client.schema.create_class({
            "class": INDEX_NAME,
            "properties": [{"name": "text", "dataType": ["text"], "tokenization": "word"}],
            "vectorizer": "none"
        })

        loaders = [
            DirectoryLoader(
                self.data_folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            ),
            DirectoryLoader(
                self.data_folder,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
            ),
        ]

        documents = []
        for loader in loaders:
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents")
            documents.extend(docs)

        logger.info(f"Total documents loaded: {len(documents)}")

        splitter = RecursiveCharacterTextSplitter(
            # Best chunk configuration:
            # chunk_size=800, chunk_overlap=150
            #
            # This setup is the best because:
            # - MRR = 1.0 → correct chunk is always ranked first (perfect retrieval)
            # - Answer score = 100 → model consistently produces correct answers
            # - Average similarity = 55.0 → acceptable semantic match, sufficient for correct grounding
            #
            # Even though similarity is not the highest, this configuration gives the best balance
            # between retrieval accuracy and final answer correctness.
            
            chunk_size=800,
            chunk_overlap=150,
        )
        splits = splitter.split_documents(documents)
        logger.info(f"Total chunks after splitting: {len(splits)}")

        stored = 0
        for doc in splits:
            vector = self.get_embedding(doc.page_content)
            self.client.data_object.create(
                data_object={"text": doc.page_content},
                class_name=INDEX_NAME,
                vector=vector
            )
            stored += 1

        logger.info(f"Stored vectors: {stored}")

        total_chars = sum(len(doc.page_content) for doc in splits)
        estimated_tokens = total_chars // 4
        logger.info(f"Estimated embedding tokens: ~{estimated_tokens}")

        return {
            "documents": len(documents),
            "chunks": len(splits),
            "stored_vectors": stored,
        }

    def get_vector_store(self):
        if self.vector_store is None:
            raise RuntimeError("Vector store is not initialized. Run init_data() first.")
        return self.vector_store