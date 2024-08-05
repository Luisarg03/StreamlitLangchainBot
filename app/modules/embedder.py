from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings


def init_embedder(api_key, **kwargs):
    embeddings = OpenAIEmbeddings(api_key=api_key, **kwargs)
    store = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )

    return cached_embedder
