from langchain.prompts import PromptTemplate
prompt=PromptTemplate(template="""  Question: {input}
                      You are a helpful Medical assistant.
                      talk like a polite and helpful doctor.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      Use maximum 3 to 4 sentences to answer the input.
     
      {context}
      """,
      input_variables=["context","input"])