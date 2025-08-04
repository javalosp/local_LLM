MODELS_CATALOG = {
    "mistral": {
        "id": "mistral-7b",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                },
    "llama": {
        "id": "llama-3-8b",
        "repo_id": "meta-llama/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    },
    "phi3": {
        "id": "phi-3-mini",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf"
    },
    "gemma": {
        "id": "gemma-7b",
        "repo_id": "google/gemma-1.1-7b-it-gguf",
        "filename": "gemma-1.1-7b-it-Q4_K_M.gguf",
    },
    "nemotron": {
        "id": "nemotron-4-22b-instruct",
        "repo_id": "nvidia/Nemotron-4-22B-Instruct-GGUF", # Assumed GGUF provider
        "filename": "Nemotron-4-22B-Instruct.Q4_K_M.gguf",
    },
    "phi4": {
        "id": "phi-4-medium-instruct",
        "repo_id": "microsoft/Phi-4-medium-14k-instruct-gguf", # Assumed repo
        "filename": "Phi-4-medium-14k-instruct.Q4_K_M.gguf",
    },
    "gemma2": {
        "id": "gemma-2-27b-instruct",
        "repo_id": "google/gemma-2-27b-it-gguf",
        "filename": "gemma-2-27b-it.Q4_K_M.gguf",
    },
    "deepseek": {
        "id": "deepseek-v2-chat",
        "repo_id": "deepseek-ai/DeepSeek-V2-chat-GGUF", # Assumed GGUF provider
        "filename": "deepseek-v2-chat.Q4_K_M.gguf",
    },
}

EMBEDDING_MODELS_CATALOG = {
    "minilm": "all-MiniLM-L6-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "gte-large": "thenlper/gte-large"
}
