from langchain_community.vectorstores import FAISS


def generate_documents_vectors(documents, cached_embedder):
    vector = FAISS.from_documents(documents, cached_embedder)
    retriever = vector.as_retriever()

    return retriever
