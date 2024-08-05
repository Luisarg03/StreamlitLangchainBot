import os
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain

from modules.confs import load_config
from modules.model import init_model
from modules.embedder import init_embedder
from modules.document_processing import pdf_spliter
from modules.generate_vectors import generate_documents_vectors
from modules.personality import personalitybot
from modules.history_prompt import generate_prompt_history
from modules.UI import generate_ui

# CONFS APP
load_dotenv()
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
configs = load_config()

if __name__ == "__main__":
    llm = init_model(api_key=api_key, **configs['modelapi'])
    embedder = init_embedder(api_key=api_key)
    pdf_spliter = pdf_spliter(configs['pdfs'])
    retriever = generate_documents_vectors(pdf_spliter, embedder)
    personality = personalitybot(llm, configs['assistantconf'])
    history_prompt = generate_prompt_history(llm, retriever)

    # # Crea cadena final de recuperación, combinando el historial de chat y la personalidad
    retrieval_chain = create_retrieval_chain(history_prompt, personality)

    generate_ui(retrieval_chain)
