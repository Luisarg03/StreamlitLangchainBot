from langchain_community.document_loaders import WikipediaLoader

docs = WikipediaLoader(query="que es un motor plano", load_max_docs=2, lang="es", doc_content_chars_max=5000).load()

b = docs[0].page_content  # a content of the Document
for i in docs:
    print(i.metadata['source'])

