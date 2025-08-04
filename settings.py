embeddings_model_name = "minilm"

method = "llamacpp"

#model = "mistral"
#model = "llama" # Gated repo
#model = "nemotron" #Repository not found
#model = "deepseek" #Repository not found
#model = "phi3"
#model = "phi4" #Repository not found
#model = "gemma"
model = "gemma2"  # Gated repo

DATA_PATH = "sources/"

BASE_VECTORSTORE_DIR = "vectorstore/"
VECTORSTORE_PATH = f"{BASE_VECTORSTORE_DIR}{embeddings_model_name}"
# Create a prompt template to guide the LLM
prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """