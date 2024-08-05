import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

# ##########################
# # FASE 0 - Configuración #
# ##########################

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("API_KEY")  # FOR OpenAIEmbeddings Method
os.environ["OPENAI_API_KEY"] = api_key  # FOR ChatOpenAI Method

llm = ChatOpenAI(
    openai_api_key=api_key,
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[StreamingStdOutCallbackHandler()]  # ! important
)

# create messages to be passed to chat LLM
messages = [HumanMessage(content="tell me a long story")]

print(llm.invoke(messages))
