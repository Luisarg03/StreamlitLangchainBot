import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


def generate_ui(retrieval_chain):
    # ################
    # # AUX FUNCTION #
    # ################
    def chat_with_history(retrieval_chain, user_input, chat_history):
        chat_history.append(HumanMessage(content=user_input))

        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })

        answer = response["answer"]
        chat_history.append(AIMessage(content=answer))

        return answer, chat_history

    # ######
    # # UI #
    # ######
    with st.sidebar:
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Luisarg03/StreamlitLangchainBot)"

    st.image("./img/logo_cat.png", width=200)
    st.title("💬 SAMI")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "En que puedo ayudar hoy?"}]
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
