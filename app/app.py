from openai import OpenAI
import streamlit as st

with st.sidebar:
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Luisarg03/StreamlitLangchainBot)"


st.image("./img/logo_cat.png", width=200)
st.title("💬 SAMI") 
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar='🐱').write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant", avatar='🐱').write(msg)
