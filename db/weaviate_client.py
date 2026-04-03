import weaviate

WEAVIATE_URL = "http://localhost:8080"
CLASS_NAME = "ElectronicsChunk"

client = weaviate.Client(WEAVIATE_URL)