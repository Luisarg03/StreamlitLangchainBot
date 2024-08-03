from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0,
    timeout=10,
    max_retries=1
    )

# output_parser = StrOutputParser()
embeddings = OpenAIEmbeddings()

#
# FASE 1 INICIO - Preparar el contexto
#
prompt = ChatPromptTemplate.from_template("""Responde en español y basado en el contexto provisto desde el documento:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

#
# FASE 1 FIN - Preparar el contexto
#

# ###################
# # Retrieval Chain #
# ###################
# SEGMENTACION DE DOCUMENTOS. PARA NO PASAR TODO EL DOCUMENTO A LA API. SOLO PASAR UNA PARTE.
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# SPLITER DE TEXTO
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# OBTENER EMBEDDINGS DE LOS DOCUMENTOS
vector = FAISS.from_documents(documents, embeddings)

# CREAR CADENA DE RECUPERACION
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "De que va el documento?"})
print(response["answer"])
