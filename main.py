import argparse
import logging

# local imports
from catalogs import MODELS_CATALOG, EMBEDDING_MODELS_CATALOG
from settings import DATA_PATH, embeddings_model_name, BASE_VECTORSTORE_DIR, VECTORSTORE_PATH, prompt_template, method, model
from preprocess import load_sources, split_documents, load_vector_store
from llm_model_setup import create_llm, create_retrieval_chain
from utils import setup_logging, get_query_from_file


def main(write_log=True):
    """
    Main function to set up and run the Retrieval-Augmented Generation (RAG) pipeline.

    This function serves as the main entry point for the command-line
    application. It handles the entire end-to-end process:
    1.  Parses command-line arguments to get an optional query file path.
    2.  Initialises the selected language model (LLM).
    3.  Loads, splits, and processes source documents into a vector store.
    4.  Constructs the final Retrieval-Augmented Generation (RAG) chain.
    5.  Executes the chain with a query (from the file or a hardcoded default).
    6.  Prints the generated answer and its source documents to the console.

    The script relies on multiple configuration variables and helper
    functions previously imported.
    """
 

    parser = argparse.ArgumentParser(description="Run a local Retrieval-Augmented Generation (RAG) chain with a specified LLM.")
    parser.add_argument("--query_file", type=str, help="Path to a text file containing the query.")
    parser.add_argument("--log", action="store_true", help="Enable detailed logging to a file and the console.")
    args = parser.parse_args()

    # Initialise logger
    # Conditionally configure logging based on the command-line argument
    if args.log or write_log:
        setup_logging()
    
    # Get a logger for this specific module
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Local LLM application...")

    # Create a local instance of the LLM with the given settings
    # We do this first to make sure that the model is usable
    # before processing the data sources
    llm = create_llm(method, model, MODELS_CATALOG, download_dir="models")

    # Pre-process data sources
    logger.info(f"Loading documents from: {DATA_PATH}")
    documents = load_sources(DATA_PATH)
    
    if not documents:
        logger.info("No documents found in the sources directory. Exiting.")
        return

    texts = split_documents(documents)
    db = load_vector_store(EMBEDDING_MODELS_CATALOG[embeddings_model_name], VECTORSTORE_PATH, texts)

    # Create an instance of a question-answer retrieval chain using the database created and the LLM
    qa_chain = create_retrieval_chain(db, prompt_template, llm)

    # Determine the query
    query = None
    if args.query_file:
        query = get_query_from_file(args.query_file)
        if query is None:
            return # Exit if file not found
    else:
        # Use a hardcoded query
        topic = "data uncertainty handling methods"
        query = f"Based on the provided documents, write a short summary of the key findings on {topic}."

    # Now use the question-answer retrieval chain
    logger.info(f"\nExecuting query: {query}\n")
    result = qa_chain({"query": query})

    # Print the result
    logger.info("\n--- Generated Answer ---\n")
    logger.info(result['result'])

    # Inspect the metadata of the source documents used for the answer to retrieve references
    logger.info("\n--- Source Documents Used ---\n")
    for doc in result['source_documents']:
        logger.info(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()