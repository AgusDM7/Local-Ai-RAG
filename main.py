from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, almacen_vectores

model = OllamaLLM(model="llama3.2")

template = """
Sos un asistente especializado en analizar políticas de privacidad.

Reglas:
- Respondé ÚNICAMENTE utilizando la información proporcionada abajo.
- Si la respuesta no está en el documento, respondé:
  "No encontré esa información en la política de privacidad."
- No inventes información.
- Respondé de forma clara y profesional en español.

Fragmentos relevantes del documento:
{datos}

Pregunta:
{question}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n-------------------------------")
    question = input("Ingrese su pregunta (q para salir): ")
    print()

    if question.lower() == "q":
        break
    
    

    # Obtener los 5 documentos más similares junto con su score de similitud
    resultados = almacen_vectores.similarity_search_with_score(question, k=5)

    print("Scores de similitud:")
    for i, (doc, score) in enumerate(resultados):
      print(f"{i+1}. Score: {score:.4f}") #muestra el score con 4 decimales.




    # El retriever se encarga de recuperar los documentos relevantes a partir de la pregunta del usuario utilizando "almacen_vectores".
    datos = retriever.invoke(question)
    result = chain.invoke({"datos": datos, "question": question})

    print ("\nRespuesta del asistente:")
    print(result)