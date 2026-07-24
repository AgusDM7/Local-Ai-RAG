from langchain_ollama import OllamaEmbeddings # generar vectores.
from langchain_chroma import Chroma # base de datos vectorial
from langchain_text_splitters import RecursiveCharacterTextSplitter # intenta cortar en lugares "naturales"
from langchain_community.document_loaders import PyPDFLoader 
import os


# Se define la variable con la ruta del PDF
PDF_PATH = "google_privacy_policy.pdf"

# definimos el modelo de embeddings de ollama (texto -> vector numérico)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# Directorio donde se almacenará la base vectorial
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location) 

### si add_documents = True (la carpeta NO existe) ###

if add_documents:
    print("Cargando PDF...")

    # Cargar PDF
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # Dividir en fragmentos (chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #caracteres
        chunk_overlap=200 #solapamiento (transicion entre caracteres)
    )

    documents = text_splitter.split_documents(pages) # split_documents funcion -> dividir documentos





# Se inicializa el almacen de vectores "Constructor" (bd vectorial) y se le asigna el modelo de embeddings.

# A nivel interno Chroma realiza...
# Si la carpeta ya existía, esto ABRE la base de datos existente en disco. 
# Si no existía, esto CREA una base de datos vacía nueva.
almacen_vectores = Chroma(
    collection_name="google_privacy_policy",
    persist_directory = db_location,
    embedding_function = embeddings
)


### si add_documents = True (la carpeta NO existe) ### agrega los datos al almacen de vectores 
# Realiza el embedding automaticamente a cada documento y lo guarda en la base de datos vectorial.
if add_documents:
    almacen_vectores.add_documents(documents)
    print("Base vectorial creada correctamente.")





# Retriever: componente que se encarga de recuperar los documentos relevantes a partir de una consulta.
# Se ejecuta siempre
retriever = almacen_vectores.as_retriever(
    search_kwargs={"k": 5} # k es la cantidad de documentos (chunks) más similares que devuelva la búsqueda por similitud.
)



