# scripts/init_indexes.py
from services.rag_data import RAGData

rag = RAGData()

rag.init_index("data/gadgets.txt", "Gadgets")
rag.init_index("data/headphones.txt", "Headphones")
rag.init_index("data/laptops.txt", "Laptops")
rag.init_index("data/smartphones.txt", "Smartphones")