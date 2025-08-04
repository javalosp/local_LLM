import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from catalogs import MODELS_CATALOG, EMBEDDING_MODELS_CATALOG
from settings import DATA_PATH, embeddings_model_name, BASE_VECTORSTORE_DIR, VECTORSTORE_PATH, prompt_template


# Preprocess documents

# Read files
def load_sources(DATA_PATH, verbosity=1):
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    if verbosity > 0:    
        print(f"Loaded {len(documents)} pages from PDF files.")
    return documents

# Split Documents
def split_documents(documents, verbosity=1):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if verbosity > 0:
        print(f"Split the documents into {len(texts)} chunks.")
    return texts


# Create vector store database
def create_vector_store(embeddings_model_name, VECTORSTORE_PATH, texts, verbosity=1):
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)   
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTORSTORE_PATH)
    if verbosity > 0:
        print(f"Vector store created and saved to {VECTORSTORE_PATH}.")
        
    return db

def load_vector_store(embeddings_model_name, VECTORSTORE_PATH, texts, verbosity=1):

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    #Check if a store already exists for this model
    if Path(VECTORSTORE_PATH).exists() and Path(VECTORSTORE_PATH).is_dir():
        if verbosity > 0:
            print(f"Loading existing vector store from: {VECTORSTORE_PATH}")
        # Load the saved vector store (so you don't have to re-process documents every time)
        db = FAISS.load_local(
                              folder_path=VECTORSTORE_PATH, 
                              embeddings=embeddings, 
                              allow_dangerous_deserialization=True
                             )
    else:
        print(f"No existing vector store found. Creating a new one at: {VECTORSTORE_PATH}")
        # Create a new vector store (and save it for future runs)
        db = create_vector_store(embeddings_model_name, VECTORSTORE_PATH, texts, verbosity=verbosity)

    return db
