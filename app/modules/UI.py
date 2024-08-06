import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import YouTubeSearchTool


def generate_ui(llm, retrieval_chain):
    # ################
    # # AUX FUNCTION #
    # ################
    def context_wikipedia(llm, retrieval_chain, prompt, chat_history):
        chat_history.append(HumanMessage(content=prompt))
        concept_messages = [
            (
                "system",
                '''Extrae del texto el concepto, idea o palabras claves.
                Genera una frase clave para buscar en motores de busqueda.
                '''
            ),
            ("human", f"{prompt}")
        ]

        concept = llm.invoke(concept_messages).content
        wikipedia_result = WikipediaLoader(
            query=concept,
            load_max_docs=3,
            lang="es",
            doc_content_chars_max=2000,
            load_all_available_meta=False).load()

        wiki_messages = [
            (
                "system",
                '''
                Eres un exeperto resumiendo textos.
                Desglosas los conceptos más importantes y los presentas de manera clara y concisa.
                Si es necesario, en tu criterio, agregas ejemplos simples y claros para ilustrar los conceptos complejos.
                Complementa los resultados con tu conocimiento del tema en cuestion.
                Siempre referencia los links de la fuente original.
                '''
            ),
            ("human", f"{wikipedia_result}")
        ]

        chunk = retrieval_chain.stream({"chat_history": chat_history, "input": wiki_messages})

        return chunk

    def context_youtube(llm, retrieval_chain, prompt, chat_history):
        chat_history.append(HumanMessage(content=prompt))
        response = retrieval_chain.invoke({"chat_history": chat_history, "input": prompt})

        concept_messages = [
            (
                "system",
                '''Extrae del texto el concepto, idea o palabras claves.
                Genera una frase clave para buscar en motores de busqueda.
                La frase sera extremaente simple y clara para que el motor de busqueda pueda encontrar resultados precisos.
                Dispondras de los links proporcionados por el motor de busqueda para compartir con el usuario.
                '''
            ),
            ("human", f"{response}")
        ]

        concept = llm.invoke(concept_messages).content
        youtubetool = YouTubeSearchTool()
        links_count = "5"
        results = youtubetool.run(f'''{concept.replace(links_count,"")},{links_count}''')
        results = results.split(',')

        return results

    def get_response(retrieval_chain, user_input, chat_history):
        chat_history.append(HumanMessage(content=user_input))
        chunk = retrieval_chain.stream({"chat_history": chat_history, "input": user_input})
        return chunk

    def stream_response(response):
        for chunk in response:
            if chunk.get("answer") is not None:
                yield chunk.get("answer")

    # ######
    # # UI #
    # ######
    sami = "🐯"
    user = "🐱"

    st.title("💬 SAM")
    st.image("./img/logo_cat.png", width=200)
    st.title("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Luisarg03/StreamlitLangchainBot)")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI, Wikipedia and Youtube")

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
            with st.chat_message("assistant", avatar=sami):
                st.write("Un momento, busco informacion en Wikipedia...🐱📖✨")

                # response = context_wikipedia(llm, retrieval_chain, prompt, st.session_state["chat_history"])
                response = st.write_stream(stream_response(context_wikipedia(llm, retrieval_chain, prompt, st.session_state["chat_history"])))
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.session_state["chat_history"].append(AIMessage(content=response))

        elif "youtube" in prompt.lower():
            with st.chat_message("assistant", avatar=sami):
                st.write("Un momento, busco videos en YouTube...🐱✨▶️")

                pattern = r"[\'\[\]]"
                urls = context_youtube(llm, retrieval_chain, prompt, st.session_state["chat_history"])
                s = ''

                st.session_state["messages"].append({"role": "assistant", "content": urls})

                for i in urls:
                    s += "- " + re.sub(pattern, '', i) + "\n"
                st.markdown(s)

                st.session_state["chat_history"].append(AIMessage(content=s))

        else:
            with st.chat_message("assistant", avatar=sami):
                st.session_state["chat_history"].append(HumanMessage(content=prompt))
                response = st.write_stream(stream_response(get_response(retrieval_chain, prompt, st.session_state["chat_history"])))
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.session_state["chat_history"].append(AIMessage(content=response))
