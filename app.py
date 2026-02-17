from flask import Flask, request, jsonify, render_template
from src.helper import extract_text_from_pdf, only_need, split_docs, create_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import *
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embeddings=create_embeddings()
index_name = "medical-chatbot"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding=embeddings
)
retriver=vectorstore.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
question_answer=create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain=create_retrieval_chain(retriver,question_answer)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input": msg})
    print("response", response["answer"])
    return str(response["answer"])






if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)