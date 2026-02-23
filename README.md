#  Local AI RAG Agent

> Agente de inteligencia artificial local con capacidades de **Retrieval-Augmented Generation (RAG)** para analizar documentos PDF, construido completamente sin APIs externas de pago.

---

## ¿Qué hace este proyecto?

Este agente permite hacerle **preguntas en lenguaje natural a un documento PDF** y obtener respuestas precisas basadas únicamente en su contenido. Todo corre **de forma local** gracias a Ollama, sin enviar datos a servidores externos.

El flujo es el siguiente:

1. Se carga un PDF y se divide en fragmentos (chunks)
2. Cada fragmento se convierte en un vector numérico (embedding)
3. Los vectores se almacenan en una base de datos vectorial local
4. Cuando el usuario hace una pregunta, se buscan los fragmentos más relevantes
5. Cada fragmento recuperado recibe un **score de similitud** que indica qué tan cercano es semánticamente a la pregunta
6. Los fragmentos con mayor relevancia + la pregunta se envían al modelo de lenguaje local
7. El modelo responde basándose **solo** en lo que encontró en el documento

---

## Demo

<img src="assets/Captura%20de%20pantalla%201.png" width="700"/>

<br/><br/>

<img src="assets/Captura%20de%20pantalla%202.png" width="700"/>

<br/><br/><br/>

<img src="assets/Captura%20de%20pantalla%203.png" width="700"/>

<br/><br/>

<img src="assets/Captura%20de%20pantalla%204.png" width="700"/>


---


## Tecnologías utilizadas

| Tecnología | Rol en el proyecto |
|---|---|
| **[Ollama](https://ollama.com/)** | Ejecuta modelos de lenguaje (LLM) de forma local |
| **[LangChain](https://www.langchain.com/)** | Orquesta el pipeline RAG (carga, chunking, retrieval, chain) |
| **[ChromaDB](https://www.trychroma.com/)** | Base de datos vectorial local para almacenar embeddings |
| **[mxbai-embed-large](https://ollama.com/library/mxbai-embed-large)** | Modelo de embeddings para convertir texto en vectores |
| **[llama3.2](https://ollama.com/library/llama3.2)** | Modelo LLM local para generar las respuestas |
| **PyPDF** | Carga y extracción de texto desde archivos PDF |
| **Python 3.10+** | Lenguaje base del proyecto |

---

## Conceptos clave

**RAG (Retrieval-Augmented Generation)**
Técnica que combina búsqueda de información con generación de texto. En lugar de que el modelo "invente" respuestas, primero busca fragmentos relevantes en una fuente de datos real y luego genera la respuesta basándose en ellos. Reduce drásticamente las alucinaciones.

**Embeddings**
Representaciones numéricas (vectores) del texto que capturan su significado semántico. Dos frases con significado similar tendrán vectores cercanos en el espacio vectorial, aunque usen palabras distintas.

**Vector Store (ChromaDB)**
Base de datos especializada en almacenar y consultar vectores de forma eficiente. Permite encontrar los fragmentos de texto más similares a una pregunta en milisegundos.

**Chunking**
Proceso de dividir documentos largos en fragmentos más pequeños (chunks) con cierto solapamiento (overlap) para que el contexto no se pierda entre fragmentos.

**LLM local con Ollama**
Ollama permite descargar y ejecutar modelos de lenguaje directamente en tu máquina, sin depender de APIs externas. Esto garantiza privacidad total de los datos.

**Scores de similitud**
Cada fragmento recuperado viene acompañado de un score numérico que indica qué tan relevante es respecto a la pregunta del usuario. Se calcula mediante distancia vectorial entre el embedding de la pregunta y el de cada chunk almacenado en ChromaDB. Un score más bajo indica mayor similitud (distancia menor en el espacio vectorial). Esto permite evaluar la calidad del retrieval y ajustar parámetros como `k` (cantidad de fragmentos recuperados) para obtener mejores respuestas.

```python
resultados = almacen_vectores.similarity_search_with_score(question, k=5)
```

---

## 🚀 Cómo ejecutarlo

### Requisitos previos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado y corriendo
- Los modelos de Ollama descargados:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AgusDM7/Local-Ai-RAG.git
cd Local-Ai-RAG

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
python main.py
```

La primera vez que se ejecute, se creará automáticamente la base de datos vectorial (`chrome_langchain_db/`) procesando el PDF incluido. Las ejecuciones siguientes reutilizarán esa base de datos.

---

## 📁 Estructura del proyecto

```
Local-Ai-RAG/
├── main.py                    # Lógica principal: prompt, chain y loop de conversación
├── vector.py                  # Carga del PDF, chunking, embeddings y configuración de ChromaDB
├── requirements.txt           # Dependencias del proyecto
├── google_privacy_policy.pdf  # PDF de ejemplo para probar el agente
├── assets/                    # Capturas de pantalla para el README
└── .gitignore
```

---

## Dependencias

```
langchain
langchain-ollama
langchain-chroma
langchain-community
pypdf
```

---

## 👤 Autor

**Agustín Del Monte**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/agustín-del-monte-741776244/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AgusDM7)

---

> *Este proyecto forma parte de mi portfolio personal de IA. Desarrollado con foco en aprendizaje de arquitecturas RAG locales y privadas.*
