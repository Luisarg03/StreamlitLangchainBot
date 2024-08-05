from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def init_model(api_key, **kwargs):
    llm = ChatOpenAI(
        **kwargs,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    return llm
