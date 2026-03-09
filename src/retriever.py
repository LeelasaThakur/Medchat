from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config import VECTOR_DB_PATH, EMBEDDING_MODEL, TOP_K


def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )