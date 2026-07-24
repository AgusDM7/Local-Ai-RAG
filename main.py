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
# Se conecta el prompt con el modelo 


#Bucle principal de ejecución segun usuario 
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




    # El retriever (definido en vector.py, con k=5) busca de nuevo los 5 chunks más relevantes y los guarda en datos
    datos = retriever.invoke(question)

    # Generación de la respuesta
    result = chain.invoke({"datos": datos, "question": question})

    print ("\nRespuesta del asistente:")
    print(result)