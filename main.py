import argparse

# local imports
from catalogs import MODELS_CATALOG, EMBEDDING_MODELS_CATALOG
from settings import DATA_PATH, embeddings_model_name, BASE_VECTORSTORE_DIR, VECTORSTORE_PATH, prompt_template, method, model
from preprocess import load_sources, split_documents, load_vector_store
from llm_model_setup import create_llm, create_retrieval_chain


def get_query_from_file(file_path):
    """
    Reads a query from the specified text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Query file not found at {file_path}")
        return None

def main():
    """
    Main function to set up and run the RAG pipeline.
    """
    parser = argparse.ArgumentParser(description="Run a local Retrieval-Augmented Generation (RAG) chain with a specified LLM.")
    parser.add_argument("--query_file", type=str, help="Path to a text file containing the query.")
    args = parser.parse_args()

    # Create a local instance of the LLM with the given settings
    # We do this first to make sure that the model is usable
    # before processing the data sources
    llm = create_llm(method, model, MODELS_CATALOG, download_dir="models")

    # Pre-process data sources
    print(f"Loading documents from: {DATA_PATH}")
    documents = load_sources(DATA_PATH)
    
    if not documents:
        print("No documents found in the sources directory. Exiting.")
        return

    texts = split_documents(documents)
    db = load_vector_store(EMBEDDING_MODELS_CATALOG[embeddings_model_name], VECTORSTORE_PATH, texts, verbosity=1)

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
    print(f"\nExecuting query: {query}\n")
    result = qa_chain({"query": query})

    # Print the result
    print("\n--- Generated Answer ---\n")
    print(result['result'])

    # Inspect the metadata of the source documents used for the answer to retrieve references
    print("\n--- Source Documents Used ---\n")
    for doc in result['source_documents']:
        print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()