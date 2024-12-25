from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


DATA_PATH = "data/"

#Step 1: Load the PDF document
def load_pdf_document(data):
    loader = DirectoryLoader(data,
                             glob = "*.pdf",
                             loader_cls=PyPDFLoader)
                             
    documents = loader.load()
    return documents

documents = load_pdf_document(data=DATA_PATH)


#Step 2: Split the document into chunks
def create_chunks(extract_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                   chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extract_data)
    return text_chunks

text_chunks = create_chunks(extract_data=documents)


#Step 3: Create vector embeddings for each chunk
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",)
    return embedding_model
  
embedding_model = get_embedding_model()


#Step 4: Store the embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)