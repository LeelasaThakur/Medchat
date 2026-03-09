from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config import VECTOR_DB_PATH, EMBEDDING_MODEL


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

    return db