from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever


def generate_prompt_history(llm, retriever):
    history_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Dada la conversación anterior, genere una consulta de búsqueda para buscar información relevante para la conversación")
    ])

    history_prompt = create_history_aware_retriever(llm, retriever, history_prompt)

    return history_prompt
