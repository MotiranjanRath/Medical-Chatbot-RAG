from dotenv import load_dotenv
import os
from src.helper import extract_text_from_pdf, only_need, split_docs, create_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

extracted_data=extract_text_from_pdf("C:/Users/motir/OneDrive/Desktop/medicalbot/Medical-Chatbot-RAG/data")
main_docs = only_need(extracted_data)
chunk_docs = split_docs(main_docs)
embeddings = create_embeddings()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))


vectorstore = PineconeVectorStore.from_documents(documents=chunk_docs, embedding=embeddings, index_name=index_name)
