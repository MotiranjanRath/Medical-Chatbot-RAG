from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings

#Extract text from pdf
def extract_text_from_pdf(pdf):
    loader = DirectoryLoader(path=pdf, glob="*.pdf",  loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#extract only that much i need to save in the vector database
from typing import List
from langchain.schema import Document

def only_need(documents: List[Document]) -> List[Document]:
    """ i only need the metadata source and page content"""
    min_docs: List[Document] = []
    for doc in documents:
        src= doc.metadata.get("source")
        page_content = doc.page_content
        mini_doc = Document(page_content, metadata={"source": src})
        min_docs.append(mini_doc)
    return min_docs

#split the documents into smaller chunks
def split_docs(documents: List[Document], chunk_size: int = 500, overlap: int = 30) -> List[Document]:
    """ split the documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_document = text_splitter.split_documents(documents)
    return split_document



#create embeddings for the documents
def create_embeddings():
    """ create embeddings for the documents"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name,encode_kwargs={'normalize_embeddings': True})
    return embedding_model
embeddings = create_embeddings()

