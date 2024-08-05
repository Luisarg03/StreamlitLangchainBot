from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_spliter(pdf_name):
    documents = []
    for k, v in pdf_name.items():
        for i in v:
            loader = PDFMinerLoader(f'../pdfs/{i}.pdf')
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )

            splits = text_splitter.split_documents(data)
            documents.append(splits)

    documents = [item for sublist in documents for item in sublist]
    return documents
