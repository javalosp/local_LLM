import os
import logging
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from catalogs import MODELS_CATALOG, EMBEDDING_MODELS_CATALOG
from settings import DATA_PATH, embeddings_model_name, BASE_VECTORSTORE_DIR, VECTORSTORE_PATH, prompt_template


# Get a logger for this module. It inherits the root configuration.
logger = logging.getLogger(__name__)

# Preprocess documents

# Read files
def load_sources(DATA_PATH, verbosity=1):
    """
    Loads all PDF documents from a specified directory.

    This function iterates through all files in the given directory, identifies
    those with a '.pdf' extension, and uses `PyPDFLoader` to load each one.
    The content of each page in the PDFs is extracted and returned as a list
    of LangChain `Document` objects.

    Args:
        DATA_PATH (str): The path to the directory containing the source
            PDF files.
        verbosity (int, optional): A flag to control console output. If set
            to a value greater than 0, it will print the total number of
            pages loaded. Defaults to 1.

    Returns:
        documents (List[Document]): A list of LangChain `Document` objects,
        where each object represents a single page from the loaded PDF files.
        Returns an empty list if no PDF files are found.
    """
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    if verbosity > 0:    
        logger.info(f"Loaded {len(documents)} pages from PDF files.")
    return documents


# Split Documents
def split_documents(documents, verbosity=1):
    """
    Splits a list of documents into smaller text chunks.

    This function uses LangChain's `RecursiveCharacterTextSplitter` to
    divide a list of `Document` objects into smaller, more manageable
    chunks. This is a crucial pre-processing step for embedding, as it
    ensures that the text segments fit within the context window of the
    embedding model.

    This operation is done with a fixed size and overlap to maintain
    semantic context between the chunks.

    Args:
        documents (List[Document]): A list of LangChain `Document` objects
            to be split.
        verbosity (int, optional): A flag to control console output. If set
            to a value greater than 0, it will print the total number of
            chunks created. Defaults to 1.

    Returns:
        texts (List[Document]): A new list of `Document` objects, where each
        object represents a single text chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if verbosity > 0:
        logger.info(f"Split the documents into {len(texts)} chunks.")
    return texts


# Create vector store database
def create_vector_store(embeddings_model_name, VECTORSTORE_PATH, texts, verbosity=1):
    """
    Creates and saves a FAISS (Facebook AI Similarity Search) vector store
    from document chunks.

    This function takes a list of text chunks, generates their numerical
    embeddings using a specified Hugging Face model, and builds a FAISS
    vector store. The resulting vector store is then saved to a specified
    local directory for later use.

    Args:
        embeddings_model_name (str): The name or path of the sentence-transformer
            model to use for creating embeddings (e.g., 'all-MiniLM-L6-v2').
        VECTORSTORE_PATH (str): The directory path where the created FAISS
            vector store will be saved.
        texts (List[Document]): A list of LangChain `Document` objects, where
            each document represents a text chunk to be embedded.
        verbosity (int, optional): A flag to control console output. If set
            to a value greater than 0, it will print a confirmation message
            after saving the store. Defaults to 1.

    Returns:
        db (FAISS): The newly created FAISS vector store object.
    """
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)   
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTORSTORE_PATH)
    if verbosity > 0:
        logger.info(f"Vector store created and saved to {VECTORSTORE_PATH}.")
        
    return db


def load_vector_store(embeddings_model_name, VECTORSTORE_PATH, texts, verbosity=1):
    """
    Loads a FAISS (Facebook AI Similarity Search) vector store from a local path,
    creating it if it doesn't exist.

    This function acts as a "smart loader." It first checks if a vector store
    already exists at the specified directory path. If it does, the function
    loads it directly from disk. If not, it triggers the creation of a new
    vector store by calling the `create_vector_store` helper function, which
    also saves it for future use.

    Args:
        embeddings_model_name (str): The name or path of the sentence-transformer
            model required to initialise the embeddings.
        VECTORSTORE_PATH (str): The directory path where the FAISS vector
            store is located or will be saved.
        texts (List[Document]): A list of LangChain `Document` objects. This is
            required only if the vector store does not already exist and needs
            to be created from scratch.
        verbosity (int, optional): A flag to control console output. If set
            to a value greater than 0, it will print status messages about
            loading or creating the store. Defaults to 1.

    Returns:
        db (FAISS): The loaded or newly created FAISS vector store object.
    """

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    #Check if a store already exists for this model
    if Path(VECTORSTORE_PATH).exists() and Path(VECTORSTORE_PATH).is_dir():
        if verbosity > 0:
            logger.info(f"Loading existing vector store from: {VECTORSTORE_PATH}")
        # Load the saved vector store (so you don't have to re-process documents every time)
        db = FAISS.load_local(
                              folder_path=VECTORSTORE_PATH, 
                              embeddings=embeddings, 
                              allow_dangerous_deserialization=True
                             )
    else:
        logger.info(f"No existing vector store found. Creating a new one at: {VECTORSTORE_PATH}")
        # Create a new vector store (and save it for future runs)
        db = create_vector_store(embeddings_model_name, VECTORSTORE_PATH, texts, verbosity=verbosity)

    return db
