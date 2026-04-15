import os
import weaviate
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate as LangChainWeaviate
from langchain.callbacks import StdOutCallbackHandler
from services.gemini_langchain_embeddings import GeminiEmbeddings

logging.getLogger("httpx").setLevel(logging.WARNING),



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
        self.vector_store = None
        self.embeddings = GeminiEmbeddings()
        self.callback = StdOutCallbackHandler()

        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready")

    logger.info("RAGData initialized, Weaviate connected")

    def init_data(self):
        if not os.path.exists(self.data_folder):
            raise RuntimeError(f"Data folder not found: {self.data_folder}")
        
        logger.info(f"Loading documents from: {self.data_folder}")

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
            logger.info(f"Loaded {len(docs)} documents from {loader}")
            documents.extend(docs)

        logger.info(f"Total documents loaded: {len(documents)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = splitter.split_documents(documents)
        logger.info(f"Total chunks after splitting: {len(splits)}")

        self.vector_store = LangChainWeaviate(
            client=self.client,
            index_name=INDEX_NAME,
            text_key="text",
            embedding=self.embeddings
        )

        ids = self.vector_store.add_documents(splits)
        logger.info(f"Stored vectors: {len(ids)}")

        total_chars = sum(len(doc.page_content) for doc in splits)
        estimated_tokens = total_chars // 4
        logger.info(f"Estimated embedding tokens: ~{estimated_tokens}")

        return {
            "documents": len(documents),
            "chunks": len(splits),
            "stored_vectors": len(ids),
        }

    def get_vector_store(self):
        if self.vector_store is None:
            raise RuntimeError("Vector store is not initialized. Run init_data() first.")
        return self.vector_store