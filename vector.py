from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os


# PDF a procesar
PDF_PATH = "google_privacy_policy.pdf"

# definimos el modelo de embeddings de ollama (texto -> vector numérico)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# Directorio donde se almacenará la base vectorial
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location) 


# Si la base vectorial no existe, se crean los documentos
if add_documents:
    print("Cargando PDF...")

    # Cargar PDF
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # Dividir en fragmentos (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = text_splitter.split_documents(pages)


# Se inicializa el almacen de vectores (bd vectorial) y se le asigna el modelo de embeddings.
# objeto de tipo Chroma
almacen_vectores = Chroma(
    collection_name="google_privacy_policy",
    persist_directory = db_location,
    embedding_function = embeddings
)


# Si no existe ./chrome_langchain_db agrega los datos al almacen de vectores 
# Realiza el embedding automaticamente a cada documento y lo guarda en la base de datos vectorial.
if add_documents:
    almacen_vectores.add_documents(documents)
    print("Base vectorial creada correctamente.")





# Retriever: componente que se encarga de recuperar los documentos relevantes a partir de una consulta.
retriever = almacen_vectores.as_retriever(
    search_kwargs={"k": 5}
)



