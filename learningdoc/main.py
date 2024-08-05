import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
import os

# ##########################
# # FASE 0 - Configuración #
# ##########################

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("API_KEY")  # FOR OpenAIEmbeddings Method
os.environ["OPENAI_API_KEY"] = api_key  # FOR ChatOpenAI Method

pdf_name = 'IA.pdf'

# Configuración del modelo LLM (Large Language Model)
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.8,
    timeout=20,
    max_retries=1
)

# Configuración de embeddings y almacenamiento en cache local
embeddings = OpenAIEmbeddings()
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace=embeddings.model
)

# #####################################
# # FASE 1 - PREPARACION DEL CONTEXTO #
# #####################################

# SEGMENTACION DE DOCUMENTOS
# Cargar el documento PDF y dividirlo en partes manejables
loader = PDFMinerLoader(f'../pdfs/{pdf_name}')
data = loader.load()

# Dividir el texto del documento en segmentos más pequeños para su procesamiento
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(data)

# OBTENER EMBEDDINGS DE LOS DOCUMENTOS
# Obtener los embeddings para los documentos segmentados y almacenarlos en un índice vectorial
vector = FAISS.from_documents(documents, cached_embedder)
retriever = vector.as_retriever()

# CREAR CADENA DE RECUPERACION Y RESPUESTA
# Crear un recuperador a partir del índice vectorial y una cadena de recuperación usando la cadena de documentos
# Crear una cadena de recuperación que tenga en cuenta el historial de la conversación...
history_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Dada la conversación anterior, genere una consulta de búsqueda para buscar información relevante para la conversación")
])

retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)

# Crear la cadena de documentos considerando el historial y dandole un toque felino
document_prompt = ChatPromptTemplate.from_messages([
    ("system", '''
    Sos un gato asistente, respondes todas las frases terminando con un "miau!!".
    Eres un gato amigable, cordial y gracioso, y respondes en español basándote en el contexto provisto desde el documento.
    Asegúrate de que tus respuestas sean útiles y tengan un toque de humor y calidez felina.
    Responda las preguntas del usuario según el siguiente contexto
    \n\n
    {context}
    '''),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain_with_history = create_stuff_documents_chain(llm, document_prompt)

# Crear cadena final de recuperación combinando la cadena de recuperación con historial y la cadena de documentos con historial
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain_with_history)


# ######################################################
# # FASE 2 - PREPARACION DEL HISTORIAL DE CONVERSACION #
# ######################################################
# Función para manejar la conversación con el historial incluido
def chat_with_history(retrieval_chain, user_input, chat_history):
    chat_history.append(HumanMessage(content=user_input))
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    answer = response["answer"]
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history


# ################
# # FASE 3 - UI  #
# ################
with st.sidebar:
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Luisarg03/StreamlitLangchainBot)"

st.image("./img/logo_cat.png", width=200)
st.title("💬 SAMI")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response, st.session_state["chat_history"] = chat_with_history(retrieval_chain, prompt, st.session_state["chat_history"])
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
