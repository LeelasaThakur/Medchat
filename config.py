import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/raw"
VECTOR_DB_PATH = "vector_store"

EMBEDDING_MODEL = "BAAI/bge-large-en"
TOP_K = 5

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

SAMBANOVA_BASE_URL = "https://api.sambanova.ai/v1"
SAMBANOVA_MODEL = "Meta-Llama-3.1-70B-Instruct"