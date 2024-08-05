import time
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.document_loaders import WikipediaLoader


def generate_ui(llm, retrieval_chain):
    # ################
    # # AUX FUNCTION #
    # ################
    def chat_with_history(retrieval_chain, user_input, chat_history):

        chat_history.append(HumanMessage(content=user_input))
        # chunks = []
        # for chunk in retrieval_chain.stream({"chat_history": chat_history, "input": user_input}):
        #     chunks.append(chunk)

        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })

        answer = response["answer"]
        chat_history.append(AIMessage(content=answer))

        return answer, chat_history

    def response_generator(response):
        for char in response:
            yield char
            time.sleep(0.01)  # Simular el retardo de escritura

    def context_wikipedia(llm, wikipedia_result):
        wiki_messages = [
            (
                "system",
                '''
                Eres un exeperto resumiendo textos.
                Desglosas los conceptos más importantes y los presentas de manera clara y concisa.
                Si es necesario, en tu criterio, agregas ejemplos simples y claros para ilustrar los conceptos complejos.
                Si el resultado no coincide con la consulta realizada, simplemente ignora la consulta y di "lo siento, la busqueda no arrojo resultados"
                Complementa los resultados con tu conocimiento
                '''
            ),
            ("human", f"{wikipedia_result}")
        ]

        wiki_response = llm.invoke(wiki_messages).content

        for i in wikipedia_result:
            link = i.metadata['source']
            wiki_response = wiki_response + f"\n\nFuente: {link}"

        return wiki_response

    # ######
    # # UI #
    # ######
    sami = "🐯"
    user = "🐱"

    st.title("💬 SAM")
    st.image("./img/logo_cat.png", width=200)
    st.title("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Luisarg03/StreamlitLangchainBot)")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "En que puedo ayudar hoy?"}]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar=sami).write(msg["content"])

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=user).write(prompt)

        if "wikipedia" in prompt.lower():
            prompt = prompt.lower().replace("wikipedia", "")
            st.chat_message("assistant", avatar=sami).write("Un momento, busco informacion en Wikipedia...🐱✨")

            messages = [
                (
                    "system",
                    '''Extrae del texto el concepto, idea o palabras claves.
                    Genera una frase clave para buscar en motores de busqueda.
                    '''
                ),
                ("human", f"{prompt}")
            ]

            concept = llm.invoke(messages).content
            wikipedia_result = WikipediaLoader(query=concept, load_max_docs=4, lang="es", doc_content_chars_max=5000).load()
            wiki_response = context_wikipedia(llm, wikipedia_result)

            st.session_state.messages.append({"role": "assistant", "content": wiki_response})
            st.chat_message("assistant", avatar=sami).write(wiki_response)

        else:
            response, st.session_state["chat_history"] = chat_with_history(retrieval_chain, prompt, st.session_state["chat_history"])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant", avatar=sami).write(response_generator(response))
