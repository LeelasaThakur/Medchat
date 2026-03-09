from openai import OpenAI
from src.retriever import load_retriever
from config import (
    SAMBANOVA_API_KEY,
    SAMBANOVA_BASE_URL,
    SAMBANOVA_MODEL
)

client = OpenAI(
    api_key=SAMBANOVA_API_KEY,
    base_url=SAMBANOVA_BASE_URL
)


def answer_query(query):
    retriever = load_retriever()
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a medical assistant. Use ONLY the context below.
If unsure, say you don't know.

Context:
{context}

Question: {query}
Answer clearly and safely.
"""

    response = client.chat.completions.create(
        model=SAMBANOVA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content