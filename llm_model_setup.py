import os
import logging
from contextlib import redirect_stdout, redirect_stderr

#from langchain.llms import LlamaCpp, CTransformers  # deprecated
from langchain_community.llms import LlamaCpp, CTransformers

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from huggingface_hub import hf_hub_download

from utils import StreamToLogger

# Get a logger for this module. It inherits the root configuration.
logger = logging.getLogger(__name__)


def get_model_params(method, model=None, use_gpu=None, verbose=False):
    """Generates a dictionary of default parameters for an LLM loading method.

    This function acts as a configuration helper, centralising the default
    parameters for different model loading libraries. It allows for easy
    switching between setups, such as using LlamaCpp for GPU offloading or
    CTransformers for CPU-based inference.

    Args:
        method (str): The loading library to use. Supported values are
            "llamacpp" and "ctransformers".
        model (str, optional): The model type (e.g., 'mistral', 'llama').
            This parameter is required only when the method is "ctransformers".
            Defaults to None.
        use_gpu (bool, optional): A flag to enable or disable GPU offloading.
            This parameter is used only when the method is "llamacpp". If True,
            it sets the model to offload all layers to the GPU. Defaults to None.

    Returns:
        dict: A dictionary containing the default parameters suitable for the
            specified loading method.

    Raises:
        ValueError: If an unsupported method is provided.
    """
    
    if method == "llamacpp":
        logger.info("Loading model parameters using LlamaCpp...")
        if use_gpu:
            gpu_layers = -1 # Offload all possible layers to the GPU
        else:
            gpu_layers = 0 # Set to 0 for CPU-only
        # Define parameters specific to LlamaCpp
        model_params = {
            "n_gpu_layers": gpu_layers,  
            "n_batch": 512,
            "n_ctx": 2048,
            "f16_kv": True, # Important for Mistral, Llama 3, and Phi-3
            "verbose": verbose,
        }
    elif method == "ctransformers":
        logger.info("Loading model parameters using CTransformers...")
        # Define parameters specific to CTransformers
        model_params = {
            "model_type": model,
            "max_new_tokens": 1024,
            "temperature": 0.7,
        }
    return model_params
    

def create_llm(method, model, MODELS_CATALOG, download_dir="models", verbosity=1):
    """Creates a LangChain LLM object, handling model download and configuration.

    This factory function streamlines the process of initialising a local language
    model. It looks up model metadata from a provided catalog, automatically
    downloads the model file from the Hugging Face Hub if it's not already
    present, retrieves the appropriate default parameters using the
    `get_model_params` helper function, and finally instantiates the correct
    LangChain LLM object.

    Args:
        method (str): The loading library to use. Supported values are
            "llamacpp" and "ctransformers".
        model (str): The identifier key for the desired model as defined in the
            MODELS_CATALOG.
        MODELS_CATALOG (dict): A dictionary mapping model identifiers to their
            metadata. Each value should be a dictionary containing keys such as
            'id', 'repo_id', and 'filename'.
        download_dir (str, optional): The root directory for storing and
            finding GGUF model files. Defaults to "models".
        verbosity (int, optional): Verbosity level (0: Silent, 1: Normal, 2: Debug).
            Defaults to "1"

    Returns:
        object: An initialised LangChain LLM object (either `LlamaCpp` or
        `CTransformers`).

    Raises:
        ValueError: If an unsupported method is provided.
        KeyError: If the specified `model` key does not exist in `MODELS_CATALOG`.
        Exception: Re-raises exceptions from `hf_hub_download` if the model
            download fails.
    """
    
    model_info = MODELS_CATALOG[model]
    
    model_path = f"{download_dir}/{model_info['filename']}" # The full path to the GGUF model file.
    model_params = get_model_params(method, model=model, verbose=(verbosity == 2)) # Set verbosity to True only for Debug level

    # Check if the model has been previously downloaded
    # If not found, try to download it from Hugging Face Hub using hf_hub_download
    if os.path.exists(model_path):
        logger.info(f"Model {model_info['id']} found")
    else:
        logger.info(f"Model {model_info['id']} not found at {model_path}. Downloading...")
        try:
            # Download the model from Hugging Face Hub using hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                local_dir=download_dir,
                local_dir_use_symlinks=False # Ensure the actual file is in the directory
            )
            logger.info(f"Model downloaded to: {downloaded_path}")
            logger.info(model_path)
        except Exception as e:
            logger.exception(f"Error downloading model {model_info['filename']} from {model_info['repo_id']}: {e}")
            raise


    # Execute the chain with output redirection according to verbosity level
    if verbosity >= 0:
        # If verbosity is high, redirect C++ library output to our logger
        logger.debug("Using DEBUG logging, i.e. --verbosity 2")
        logger.debug("Redirecting C++ library output to logger.")
        
        # Create stream objects that point to our logger
        stdout_logger = StreamToLogger(logger, logging.DEBUG)
        stderr_logger = StreamToLogger(logger, logging.ERROR)
        
        with redirect_stdout(stdout_logger), redirect_stderr(stderr_logger):
            if method == "llamacpp":
                logger.info("Loading model using LlamaCpp...")
                return LlamaCpp(model_path=model_path, **model_params)
                
            elif method == "ctransformers":
                logger.info("Loading model using CTransformers...")
                return CTransformers(model=model_path, **model_params)
            
            else:
                raise ValueError(f"Unsupported method: '{method}'. Choose 'llamacpp' or 'ctransformers'.")
    else:
        if method == "llamacpp":
            logger.info("Loading model using LlamaCpp...")
            return LlamaCpp(model_path=model_path, **model_params)
            
        elif method == "ctransformers":
            logger.info("Loading model using CTransformers...")
            return CTransformers(model=model_path, **model_params)    
        
        else:
            raise ValueError(f"Unsupported method: '{method}'. Choose 'llamacpp' or 'ctransformers'.")
    

def create_retrieval_chain(db, prompt_template, llm):
    """Constructs and returns a Retrieval-Augmented Generation (RAG) chain.

    This function assembles the core components of a RAG pipeline (a vector
    store, a prompt template, and a language model) into a ready-to-use
    LangChain `RetrievalQA` object.

    The chain is configured to use the "stuff" method, which inserts all
    retrieved document chunks directly into the context of the prompt. It is
    also set to retrieve the top 2 most relevant chunks for a given query.

    Args:
        db (VectorStore): An initialised LangChain VectorStore object (e.g., FAISS)
            that contains the document embeddings and will be used for retrieval.
        prompt_template (str): The string template for the final prompt sent to
            the language model. It must include the placeholders `{context}`
            and `{question}`.
        llm (LLM): An initialised LangChain language model object that will
            generate the final answer based on the prompt.

    Returns:
        qa_chain: A configured LangChain `RetrievalQA` chain object, ready
            to be queried.
    """
    # Set up the Retrieval Chain
    retriever = db.as_retriever(search_kwargs={"k": 2}) # Retrieve the top 2 most relevant chunks
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Create the final chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain