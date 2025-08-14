import argparse
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr

# local imports
from catalogs import MODELS_CATALOG, EMBEDDING_MODELS_CATALOG
from settings import DATA_PATH, embeddings_model_name, BASE_VECTORSTORE_DIR, VECTORSTORE_PATH, prompt_template, method, model
from preprocess import load_sources, split_documents, load_vector_store
from llm_model_setup import create_llm, create_retrieval_chain
from utils import setup_logging, get_query_from_file, StreamToLogger


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
    parser.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2], help="Set the verbosity level: 0=silent, 1=info, 2=debug.")
    
    args = parser.parse_args()

    # Set a default verbosity value to pass to create_llm function
    verbosity = 1

    # Initialise logger
    # Conditionally configure logging based on the command-line argument
    if args.log or write_log:
        if args.verbosity:
            # Map the integer verbosity level to a logging level
            log_levels = {
                0: logging.WARNING, # Silent: only show warnings and errors
                1: logging.INFO,    # Normal: show info, warnings, and errors
                2: logging.DEBUG,   # Debug: show all messages
            }
            #setup_logging(level=log_levels.get(args.verbosity))
            # Set console level below file level
            if log_levels.get(args.verbosity) < 2:
                setup_logging(console_level=0, file_level=log_levels.get(args.verbosity))
            else:
                setup_logging(console_level=1, file_level=log_levels.get(args.verbosity))
            
            # Update verbosity value
            verbosity = args.verbosity
        else:
            # If no verbosity level is passed, just use the default level:
            # level=logging.INFO
            setup_logging()
    
    # Get a logger for this specific module
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Local LLM application...")
    logger.debug("Using DEBUG logging, i.e. --verbosity 2")

    # Create a local instance of the LLM with the given settings
    # We do this first to make sure that the model is usable
    # before processing the data sources
    llm = create_llm(method, model, MODELS_CATALOG, download_dir="models", verbosity=verbosity)

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
            logger.warning(f"You have passed '--query_file' flag. However, file {args.query_file} was not found. Exiting.")
            return # Exit if file not found
    else:
        # Use a hardcoded query
        topic = "data uncertainty handling methods"
        query = f"Based on the provided documents, write a short summary of the key findings on {topic}."

    # Now use the question-answer retrieval chain
    logger.info(f"\nExecuting query: {query}\n")
    #result = qa_chain({"query": query}) # deprecated
    result = qa_chain.invoke(input=query)

    # Execute the chain with output redirection according to verbosity level
    if args.verbosity == 2:
        # If verbosity is high, redirect C++ library output to our logger
        logger.debug("Using DEBUG logging, i.e. --verbosity 2")
        logger.debug("Redirecting C++ library output to logger.")
        
        # Create stream objects that point to our logger
        stdout_logger = StreamToLogger(logger, logging.DEBUG)
        stderr_logger = StreamToLogger(logger, logging.ERROR)
        
        with redirect_stdout(stdout_logger), redirect_stderr(stderr_logger):
            result = qa_chain.invoke(input=query)
    else:
        result = qa_chain.invoke(input=query)

    # Print the result
    logger.info("\n--- Generated Answer ---\n")
    logger.info(result['result'])

    # Inspect the metadata of the source documents used for the answer to retrieve references
    logger.info("\n--- Source Documents Used ---\n")
    for doc in result['source_documents']:
        logger.info(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()