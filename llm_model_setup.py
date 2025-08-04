import os

from langchain.llms import LlamaCpp, CTransformers

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from huggingface_hub import hf_hub_download

def get_model_params(method, model=None, use_gpu=None):
    """
    Args:
        method (str): The library to use, either "llamacpp" or "ctransformers".
    """
    
    if method == "llamacpp":
        print("Loading model parameters using LlamaCpp...")
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
            "verbose": True,
        }
    elif method == "ctransformers":
        print("Loading model parameters using CTransformers...")
        # Define parameters specific to CTransformers
        model_params = {
            "model_type": model,
            "max_new_tokens": 1024,
            "temperature": 0.7,
        }
    return model_params
    

def create_llm(method, model, MODELS_CATALOG, download_dir="models"):
    """
    Dynamically creates and returns a LangChain LLM object based on the specified method.

    Args:
        method (str): The library to use, either "llamacpp" or "ctransformers".
        model (str): Name of the model (key in MODELS_CATALOG)
        MODELS_CATALOG (dict): Dictionary with the information of different models
        download_dir (str): The local directory where models are stored. Defaults to "models".
    Returns:
        A LangChain LLM object, or raises a ValueError if the method is unsupported.
    """
    
    model_info = MODELS_CATALOG[model]
    
    model_path = f"{download_dir}/{model_info['filename']}" # The full path to the GGUF model file.
    model_params = get_model_params(method, model=model)

    # Check if the model has been previously downloaded
    # If not found, try to download it from Hugging Face Hub using hf_hub_download
    if os.path.exists(model_path):
        print(f"Model {model_info['id']} found")
    else:
        print(f"Model {model_info['id']} not found at {model_path}. Downloading...")
        try:
            # Download the model from Hugging Face Hub using hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                local_dir=download_dir,
                local_dir_use_symlinks=False # Ensure the actual file is in the directory
            )
            print(f"Model downloaded to: {downloaded_path}")
            print(model_path)
        except Exception as e:
            print(f"Error downloading model {model_info['filename']} from {model_info['repo_id']}: {e}")
            raise


    if method == "llamacpp":
        print("Loading model using LlamaCpp...")
        return LlamaCpp(model_path=model_path, **model_params)
        
    elif method == "ctransformers":
        print("Loading model using CTransformers...")
        return CTransformers(model=model_path, **model_params)
        
    else:
        raise ValueError(f"Unsupported method: '{method}'. Choose 'llamacpp' or 'ctransformers'.")
    

def create_retrieval_chain(db, prompt_template, llm):
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