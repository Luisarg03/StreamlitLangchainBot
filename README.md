# LangChain Conversation Retrieval Example

Este repositorio proporciona un ejemplo de cómo utilizar `langchain` para crear una cadena de recuperación de conversaciones, permitiendo a un modelo de lenguaje (LLM) responder preguntas de seguimiento teniendo en cuenta el historial completo de la conversación.

### Preparación del Entorno

```bash
    python -m venv venv
```
```bash
    source venv/bin/activate
```
```bash
    pip install -r requeriments.txt
```

### Configuración de la API Key

Crea un archivo `.env` en la raíz del proyecto y agrega tu clave API de OpenAI:

```env
OPENAI_API_KEY=tu_clave_api
```

Carga las variables de entorno en tu script:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
```

## Descripción del Código

### Importación de Librerías

El código comienza con la importación de las librerías necesarias:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
```

### Configuración del LLM y Embeddings

Se carga la clave API desde el archivo `.env` y se configura el modelo de lenguaje (LLM) y los embeddings de OpenAI:

```python
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0,
    timeout=10,
    max_retries=1
)

embeddings = OpenAIEmbeddings()
```

### Preparación del Contexto

Se crea un prompt y una cadena de documentos para preparar el contexto:

```python
prompt = ChatPromptTemplate.from_template("""Responde en español y basado en el contexto provisto desde el documento:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
```

### Cadena de Recuperación

Se cargan los documentos desde una URL, se segmentan, y se crean los embeddings para luego construir la cadena de recuperación:

```python
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

### Ejemplo de Uso

Finalmente, se invoca la cadena de recuperación con una pregunta y se imprime la respuesta:

```python
response = retrieval_chain.invoke({"input": "De que va el documento?"})
print(response["answer"])
```

## Resumen

Este proyecto demuestra cómo usar `langchain` para crear un chatbot que responde preguntas de seguimiento considerando el historial completo de la conversación. Se utilizan técnicas avanzadas como embeddings y almacenamiento vectorial (FAISS) para mejorar la relevancia y precisión de las respuestas generadas por el modelo de lenguaje.

## Contribución

Si tienes sugerencias o mejoras, no dudes en hacer un fork de este repositorio y enviar un pull request. ¡Agradecemos tu colaboración!

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo `LICENSE` para obtener más información.