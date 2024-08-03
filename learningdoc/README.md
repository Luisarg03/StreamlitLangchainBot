## Script DEMO - LangChain

Este directorio contiene un script DEMO que utiliza LangChain para procesar documentos PDF y responder preguntas en español basadas en el contenido de los documentos.

### Descripción del Script

El script realiza las siguientes tareas:

1. **Fase 0 - Configuración**:
    - Carga las variables de entorno necesarias.
    - Configura el modelo de lenguaje grande (LLM) utilizando `ChatOpenAI`.
    - Configura embeddings y almacenamiento en caché local.

2. **Fase 1 - Preparación del Contexto**:
    - Crea una plantilla de prompt para el modelo.
    - Carga un documento PDF y lo divide en segmentos manejables utilizando `RecursiveCharacterTextSplitter`.
    - Obtiene los embeddings para los documentos segmentados y los almacena en un índice vectorial `FAISS`.

3. **Fase 2 - Preparación del Historial de Conversación**:
    - Crea una cadena de recuperación y respuesta que tenga en cuenta el historial de la conversación utilizando `ChatPromptTemplate` y `MessagesPlaceholder`.

4. **Fase 3 - Test**:
    - Función para manejar la conversación con el historial incluido.
    - Bucle para manejar la interacción con el usuario y responder preguntas basadas en el contexto de los documentos PDF.

### Requisitos

- Python
- LangChain
- OpenAI API Key
- PDFMiner
- FAISS
- dotenv

### Uso

1. Clonar el repositorio y navegar al directorio `learningdoc`.
2. Instalar los requisitos.
    
    ```sh
    pip install -r requirements.txt
    ```

3. Ejecutar el script.

    ```sh
    python main.py
    ```