from langchain_openai import ChatOpenAI


def init_model(api_key, **kwargs):
    llm = ChatOpenAI(**kwargs)

    return llm
