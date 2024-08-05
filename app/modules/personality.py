from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


def personalitybot(llm, config):
    atributies = f'''
        Tu nombre es {config['name']}
        Sos un {config['description']}
        Hablas en {config['language']}
        Tu personalidad es {config['personality']}
        Tambien tenes estos atributos: {config['other']}
        Responda las preguntas del usuario según el siguiente contexto
        '''
    prompt = atributies + '''
        Responda las preguntas del usuario según el siguiente contexto
        \n\n
        {context}
        '''

    document_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    personality = create_stuff_documents_chain(llm, document_prompt)

    return personality
