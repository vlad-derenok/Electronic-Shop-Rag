import os
import weaviate
import logging
import ollama
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import TextLoader, PyPDFLoader
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
DATA_FOLDER = os.getenv("DATA_FOLDER")


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: str
    answer: str
    precision: float
    category: str


class RAGData:
    def __init__(self, data_folder: str = DATA_FOLDER):
        self.client = weaviate.Client(WEAVIATE_URL)
        self.data_folder = data_folder
        self.state: RAGState = {
            "messages": [],
            "question": "",
            "context": "",
            "answer": "",
            "precision": 0.0,
            "category": ""
        }

        if not self.client.is_ready():
            raise RuntimeError("Weaviate is not ready")

        logger.info("RAGData initialized, Weaviate connected")

    def update_state(self, question: str, answer: str, context: str, precision: float = 0.0, category: str = ""):
        self.state["question"] = question
        self.state["context"] = context
        self.state["answer"] = answer
        self.state["precision"] = precision
        self.state["category"] = category
        self.state["messages"] = self.state["messages"] + [
            HumanMessage(content=question),
            AIMessage(content=answer)
        ]
        logger.info(f"State updated — messages: {len(self.state['messages'])}, category: {category}")

    def get_history(self, max_turns: int = 3) -> str:
        messages = self.state["messages"][-max_turns * 2:]
        history_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_text += f"Question: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Answer: {msg.content}\n\n"
        return history_text

    def get_embedding(self, text: str) -> list:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response["embedding"]

    def init_index(self, file_path: str, index_name: str):
        try:
            self.client.schema.delete_class(index_name)
            logger.info(f"Deleted existing index: {index_name}")
        except Exception:
            pass

        self.client.schema.create_class({
            "class": index_name,
            "properties": [{"name": "text", "dataType": ["text"], "tokenization": "word"}],
            "vectorizer": "none"
        })

        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {file_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )
        splits = splitter.split_documents(documents)
        logger.info(f"Chunks: {len(splits)}")

        stored = 0
        for doc in splits:
            vector = self.get_embedding(doc.page_content)
            self.client.data_object.create(
                data_object={"text": doc.page_content},
                class_name=index_name,
                vector=vector
            )
            stored += 1

        logger.info(f"Stored {stored} vectors in index: {index_name}")
        return {"chunks": len(splits), "stored": stored}