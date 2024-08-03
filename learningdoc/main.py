# Importar las bibliotecas necesarias
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

# # Cargar variables de entorno
load_dotenv()
api_key = os.getenv("API_KEY")  # # FOR OpenAIEmbeddings Method
os.environ["OPENAI_API_KEY"] = api_key  # FOR ChatOpenAI Method

pdf_name = 'IA.pdf'

# # Configuración del modelo LLM (Large Language Model)
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0,
    timeout=10,
    max_retries=1
)

# # Configuración de embeddings y almacenamiento en cache local
embeddings = OpenAIEmbeddings()
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace=embeddings.model
)


# #####################################
# # FASE 1 - PREPARACION DEL CONTEXTO #
# #####################################
# # Crear una plantilla de prompt para el modelo
prompt = ChatPromptTemplate.from_template('''
    Responde en español y basado en el contexto provisto desde el documento:

    <context>
    {context}
    </context>

    Question: {input}''')

# # Crear una cadena de documentos usando la plantilla de prompt y el modelo LLM
document_chain = create_stuff_documents_chain(llm, prompt)
# # FASE 1 FIN - Preparar el contexto

# # SEGMENTACION DE DOCUMENTOS
# # Cargar el documento PDF y dividirlo en partes manejables
loader = PDFMinerLoader(f'../pdf_context_files/{pdf_name}')
data = loader.load()

# # Dividir el texto del documento en segmentos más pequeños para su procesamiento
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(data)

# # OBTENER EMBEDDINGS DE LOS DOCUMENTOS
# # Obtener los embeddings para los documentos segmentados y almacenarlos en un índice vectorial
vector = FAISS.from_documents(documents, cached_embedder)


# # CREAR CADENA DE RECUPERACION Y RESPUESTA
# # Crear un recuperador a partir del índice vectorial y una cadena de recuperación usando la cadena de documentos
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # Crear una cadena de recuperación que tenga en cuenta el historial de la conversación
history_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, history_prompt)

# # Crear la cadena de documentos considerando el historial
document_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain_with_history = create_stuff_documents_chain(llm, document_prompt)

# # Crear cadena final de recuperación combinando la cadena de recuperación con historial y la cadena de documentos con historial
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain_with_history)


# ######################################################
# # FASE 2 - PREPARACION DEL HISTORIAL DE CONVERSACION #
# ######################################################
# Función para manejar la conversación con el historial incluido
def chat_with_history(retrieval_chain, user_input, chat_history=[]):
    chat_history.append(HumanMessage(content=user_input))
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    answer = response["answer"]
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history


# # Probar la cadena con una conversación inicial
chat_history = []


# #################
# # FASE 3 - TEST #
# #################
# # Bucle para manejar la interacción con el usuario
while True:
    user_input = input("Tu pregunta: ")
    if user_input.lower() in ["exit", "salir"]:
        print("Terminando la conversación.")
        break
    answer, chat_history = chat_with_history(retrieval_chain, user_input, chat_history)
    print(f"AI: {answer}\n")
